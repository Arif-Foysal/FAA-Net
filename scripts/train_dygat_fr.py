#!/usr/bin/env python3
"""
DyGAT-FR Training Script

Train DyGAT-FR on NIDS datasets with incremental learning.

Usage:
    python scripts/train_dygat_fr.py --dataset unsw-nb15 --epochs 50
    python scripts/train_dygat_fr.py --dataset synthetic --incremental --n_increments 5
"""

import argparse
import os
import sys
import json
from datetime import datetime

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dygat_fr import DyGATFR, DyGATFRLoss, DyGATFRTrainer
from core.dygat_fr.data_loader import (
    NIDSDataLoader, 
    create_synthetic_graph,
    compute_graph_statistics,
    TemporalGraphSplitter
)
from core.dygat_fr.utils import (
    set_seed, 
    compute_metrics,
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    format_results_table,
    save_checkpoint
)
from core.dygat_fr.trainer import IncrementalTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train DyGAT-FR for Network Intrusion Detection'
    )
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['unsw-nb15', 'synthetic'],
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Directory containing dataset files')
    
    # Model arguments
    parser.add_argument('--hidden_channels', type=int, default=128,
                        help='Hidden layer dimension')
    parser.add_argument('--out_channels', type=int, default=64,
                        help='Output embedding dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of DyGAT layers')
    parser.add_argument('--heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--n_prototypes', type=int, default=8,
                        help='Number of minority prototypes')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma')
    parser.add_argument('--use_feedback', action='store_true',
                        help='Enable feedback refinement module')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--curriculum_epochs', type=int, default=5,
                        help='Curriculum learning epochs')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Early stopping patience')
    
    # Incremental learning arguments
    parser.add_argument('--incremental', action='store_true',
                        help='Use incremental learning')
    parser.add_argument('--n_increments', type=int, default=5,
                        help='Number of increments')
    parser.add_argument('--minority_drift', action='store_true',
                        help='Simulate minority class drift')
    
    # Synthetic data arguments
    parser.add_argument('--n_nodes', type=int, default=5000,
                        help='Number of nodes for synthetic data')
    parser.add_argument('--n_features', type=int, default=33,
                        help='Number of features')
    parser.add_argument('--minority_ratio', type=float, default=0.2,
                        help='Minority class ratio')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results/dygat_fr',
                        help='Output directory')
    parser.add_argument('--save_model', action='store_true',
                        help='Save trained model')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots')
    
    # Misc arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Get computing device."""
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def load_data(args):
    """Load dataset based on arguments."""
    if args.dataset == 'synthetic':
        print(f"Creating synthetic graph with {args.n_nodes} nodes...")
        data = create_synthetic_graph(
            n_nodes=args.n_nodes,
            n_features=args.n_features,
            minority_ratio=args.minority_ratio,
            random_state=args.seed
        )
        
        # Create train/val/test masks
        n = data.num_nodes
        perm = torch.randperm(n)
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        
        train_mask[perm[:int(0.6*n)]] = True
        val_mask[perm[int(0.6*n):int(0.8*n)]] = True
        test_mask[perm[int(0.8*n):]] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        return data, None
    
    elif args.dataset == 'unsw-nb15':
        print("Loading UNSW-NB15 dataset...")
        loader = NIDSDataLoader(
            dataset_name='unsw-nb15',
            data_dir=args.data_dir
        )
        train_data, test_data = loader.load_unsw_nb15()
        
        # Create validation split from training
        n = train_data.num_nodes
        perm = torch.randperm(n)
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        
        train_mask[perm[:int(0.8*n)]] = True
        val_mask[perm[int(0.8*n):]] = True
        
        train_data.train_mask = train_mask
        train_data.val_mask = val_mask
        
        return train_data, test_data


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load data
    train_data, test_data = load_data(args)
    
    # Print statistics
    stats = compute_graph_statistics(train_data)
    print("\n=== Dataset Statistics ===")
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Features: {stats['num_features']}")
    print(f"Avg Degree: {stats['avg_degree']:.2f}")
    print(f"Class Distribution: {stats['class_distribution']}")
    if 'imbalance_ratio' in stats:
        print(f"Imbalance Ratio: {stats['imbalance_ratio']:.4f}")
    
    # Get input dimension
    in_channels = train_data.x.size(1)
    
    # Create model
    model = DyGATFR(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_classes=2,
        num_layers=args.num_layers,
        heads=args.heads,
        n_prototypes=args.n_prototypes,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        dropout=args.dropout,
        use_feedback=args.use_feedback
    )
    
    print(f"\n=== Model Summary ===")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Layers: {args.num_layers}")
    print(f"Heads: {args.heads}")
    print(f"Prototypes: {args.n_prototypes}")
    print(f"Feedback: {args.use_feedback}")
    
    # Create loss function
    loss_fn = DyGATFRLoss(
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        contrastive_weight=0.1,
        replay_weight=0.1,
        num_classes=2
    )
    
    # Create trainer
    if args.incremental:
        trainer = IncrementalTrainer(
            model=model,
            loss_fn=loss_fn,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            curriculum_epochs=args.curriculum_epochs
        )
    else:
        trainer = DyGATFRTrainer(
            model=model,
            loss_fn=loss_fn,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            curriculum_epochs=args.curriculum_epochs
        )
    
    # Training
    print(f"\n=== Training ===")
    
    if args.incremental:
        # Create increments
        splitter = TemporalGraphSplitter(
            n_increments=args.n_increments,
            minority_drift=args.minority_drift
        )
        increments = splitter.split(train_data, random_state=args.seed)
        
        print(f"Training on {args.n_increments} increments...")
        results = trainer.train_increments(
            increments,
            epochs_per_increment=args.epochs // args.n_increments,
            verbose=args.verbose
        )
        
        history = trainer.history
    else:
        # Standard training
        history = trainer.train_increment(
            train_data,
            epochs=args.epochs,
            train_mask=train_data.train_mask if hasattr(train_data, 'train_mask') else None,
            val_mask=train_data.val_mask if hasattr(train_data, 'val_mask') else None,
            verbose=args.verbose,
            early_stopping_patience=args.early_stopping
        )
    
    # Evaluation
    print(f"\n=== Evaluation ===")
    
    # Evaluate on training data
    train_metrics = trainer.evaluate(
        train_data,
        mask=train_data.train_mask if hasattr(train_data, 'train_mask') else None
    )
    print("\nTraining Metrics:")
    print(format_results_table(train_metrics))
    
    # Evaluate on validation data
    if hasattr(train_data, 'val_mask'):
        val_metrics = trainer.evaluate(train_data, mask=train_data.val_mask)
        print("\nValidation Metrics:")
        print(format_results_table(val_metrics))
    
    # Evaluate on test data if available
    if test_data is not None:
        test_metrics = trainer.evaluate(test_data)
        print("\nTest Metrics:")
        print(format_results_table(test_metrics))
    
    # Save results
    results_dict = {
        'config': vars(args),
        'model_params': model.count_parameters(),
        'train_metrics': train_metrics,
        'history': {k: v for k, v in history.items()},
        'timestamp': timestamp
    }
    
    if hasattr(train_data, 'val_mask'):
        results_dict['val_metrics'] = val_metrics
    if test_data is not None:
        results_dict['test_metrics'] = test_metrics
    
    results_path = os.path.join(args.output_dir, f'results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    
    # Save model
    if args.save_model:
        model_path = os.path.join(args.output_dir, f'model_{timestamp}.pt')
        save_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            epoch=args.epochs,
            metrics=train_metrics,
            path=model_path
        )
        print(f"Model saved to {model_path}")
    
    # Generate plots
    if args.plot:
        print("\nGenerating plots...")
        
        # Training history
        history_path = os.path.join(args.output_dir, f'history_{timestamp}.png')
        plot_training_history(history, save_path=history_path)
        
        # Get predictions for curves
        model.eval()
        with torch.no_grad():
            x = train_data.x.to(device)
            edge_index = train_data.edge_index.to(device)
            logits = model(x, edge_index)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
        y_true = train_data.y.numpy()
        y_pred = (probs > 0.5).astype(int)
        
        # Confusion matrix
        cm_path = os.path.join(args.output_dir, f'confusion_matrix_{timestamp}.png')
        plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
        
        # ROC and PR curves
        curves_path = os.path.join(args.output_dir, f'roc_pr_{timestamp}.png')
        plot_roc_pr_curves(y_true, probs, model_name='DyGAT-FR', save_path=curves_path)
    
    print("\n=== Training Complete ===")


if __name__ == '__main__':
    main()
