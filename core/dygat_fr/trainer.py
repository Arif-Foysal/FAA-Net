"""
DyGAT-FR Trainer

Training pipeline for DyGAT-FR with:
1. Curriculum learning (easy-to-hard sampling)
2. Incremental adaptation (prototype updates, replay)
3. Feedback loop integration
4. Comprehensive evaluation metrics
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)
import time
from collections import defaultdict


class DyGATFRTrainer:
    """
    Trainer for DyGAT-FR with incremental learning support.
    
    Features:
    - Curriculum learning: Start with balanced data, increase imbalance
    - Prototype management: Initialize and update prototypes
    - Memory replay: Prevent catastrophic forgetting
    - Feedback integration: Refine based on errors
    - Comprehensive metrics: F1, Recall, Precision, AUC, etc.
    
    Args:
        model: DyGATFR model instance
        loss_fn: DyGATFRLoss instance
        device: Training device (cuda/cpu)
        lr: Learning rate
        weight_decay: L2 regularization strength
        curriculum_epochs: Number of curriculum learning epochs
        feedback_start_epoch: Epoch to start feedback refinement
        replay_ratio: Ratio of replay samples to batch size
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        curriculum_epochs: int = 5,
        feedback_start_epoch: int = 10,
        replay_ratio: float = 0.1
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.curriculum_epochs = curriculum_epochs
        self.feedback_start_epoch = feedback_start_epoch
        self.replay_ratio = replay_ratio
        
        # Optimizer with weight decay
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Training history
        self.history: Dict[str, List[float]] = defaultdict(list)
        
        # Best model tracking
        self.best_val_f1 = 0.0
        self.best_model_state = None
        
        # Timing
        self.epoch_times: List[float] = []
    
    def curriculum_sample(
        self,
        labels: torch.Tensor,
        epoch: int,
        max_epochs: int,
        minority_label: Optional[int] = None
    ) -> torch.Tensor:
        """
        Curriculum learning: Start with balanced, ramp up imbalance.
        
        Args:
            labels: Node labels (N,)
            epoch: Current epoch
            max_epochs: Total epochs
            minority_label: Label for minority class (auto-detected if None)
        
        Returns:
            Selected indices for training
        """
        # Auto-detect minority class if not specified
        if minority_label is None:
            unique, counts = torch.unique(labels, return_counts=True)
            if len(unique) >= 2:
                minority_label = unique[counts.argmin()].item()
            else:
                minority_label = 1  # fallback
        
        minority_mask = labels == minority_label
        majority_mask = ~minority_mask
        
        minority_idx = torch.where(minority_mask)[0]
        majority_idx = torch.where(majority_mask)[0]
        
        n_minority = len(minority_idx)
        n_majority = len(majority_idx)
        
        # Get device from labels
        device = labels.device
        
        if epoch < self.curriculum_epochs:
            # Early epochs: Nearly balanced sampling
            # Gradually increase imbalance ratio
            balance_factor = 1.0 + (epoch / self.curriculum_epochs) * 2.0
            n_sample_majority = min(
                int(n_minority * balance_factor),
                n_majority
            )
            
            sampled_majority = majority_idx[
                torch.randperm(n_majority, device=device)[:n_sample_majority]
            ]
            selected_idx = torch.cat([minority_idx, sampled_majority])
        else:
            # Later epochs: Use all data
            selected_idx = torch.arange(len(labels), device=device)
        
        # Shuffle
        return selected_idx[torch.randperm(len(selected_idx), device=device)]
    
    def compute_feedback_signals(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute feedback signals for refinement.
        
        Args:
            logits: Model predictions (N, 1)
            labels: Ground truth (N,)
            embeddings: Node embeddings (N, d)
        
        Returns:
            prediction_errors: Per-node error signals (N,)
            loss_values: Per-node loss values (N,)
        """
        # Predictions
        probs = torch.sigmoid(logits.squeeze(-1))
        
        # Prediction errors (binary)
        preds = (probs > 0.5).float()
        errors = (preds != labels.float()).float()
        
        # Per-node loss (BCE without reduction)
        loss_values = nn.functional.binary_cross_entropy(
            probs, labels.float(), reduction='none'
        )
        
        return errors, loss_values
    
    def train_epoch(
        self,
        data: Any,
        epoch: int,
        max_epochs: int,
        train_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            data: PyG Data object with x, edge_index, y
            epoch: Current epoch number
            max_epochs: Total number of epochs
            train_mask: Optional mask for training nodes
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        # Move data to device
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        y = data.y.to(self.device)
        
        # Curriculum sampling
        if train_mask is not None:
            # Move train_mask to same device as y
            train_mask_device = train_mask.to(self.device) if train_mask.device != self.device else train_mask
            available_idx = torch.where(train_mask_device)[0]
            labels_available = y[available_idx]
            selected_relative = self.curriculum_sample(
                labels_available, epoch, max_epochs
            )
            # Ensure selected_relative is on same device
            if selected_relative.device != available_idx.device:
                selected_relative = selected_relative.to(available_idx.device)
            train_idx = available_idx[selected_relative]
        else:
            train_idx = self.curriculum_sample(y, epoch, max_epochs)
        
        # Forward pass
        logits, embeddings = self.model(
            x, edge_index,
            return_embeddings=True
        )
        
        # Get training subset
        train_logits = logits[train_idx]
        train_labels = y[train_idx]
        train_embeddings = embeddings[train_idx]
        
        # Get prototypes for contrastive loss
        prototypes = self.model.get_prototype_keys()
        
        # Compute main loss
        loss = self.loss_fn(
            train_logits, train_labels,
            embeddings=train_embeddings,
            prototypes=prototypes
        )
        
        # Memory replay (after initial epochs)
        if epoch >= self.feedback_start_epoch:
            n_replay = int(len(train_idx) * self.replay_ratio)
            replay_x, replay_y = self.model.sample_replay(
                n_replay, self.device
            )
            
            if replay_x is not None and len(replay_x) > 0:
                # Process replay samples through input projection + classifier
                # (simplified - full version would use graph context)
                replay_proj = self.model.input_proj(replay_x)  # project raw features to hidden dim
                replay_proj = self.model.input_norm(replay_proj)
                replay_embed = self.model.residual_proj(replay_proj)  # project to out_channels
                replay_logits = self.model.classifier(replay_embed)
                
                replay_loss = self.loss_fn.focal_loss(replay_logits, replay_y)
                loss = loss + self.loss_fn.replay_weight * replay_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            probs = torch.sigmoid(train_logits).cpu().numpy()
            preds = (probs > 0.5).astype(int).flatten()
            labels_np = train_labels.cpu().numpy()
            
            metrics = {
                'loss': loss.item(),
                'f1': f1_score(labels_np, preds, zero_division=0),
                'recall': recall_score(labels_np, preds, zero_division=0),
                'precision': precision_score(labels_np, preds, zero_division=0)
            }
            
            try:
                metrics['auc'] = roc_auc_score(labels_np, probs)
            except ValueError:
                metrics['auc'] = 0.0
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        data: Any,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on data.
        
        Args:
            data: PyG Data object
            mask: Optional mask for evaluation nodes
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        # Move data to device
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        y = data.y.to(self.device)
        
        # Forward pass
        logits = self.model(x, edge_index)
        
        # Apply mask
        if mask is not None:
            logits = logits[mask]
            labels = y[mask]
        else:
            labels = y
        
        # Compute loss
        loss = self.loss_fn.focal_loss(logits, labels)
        
        # Compute predictions
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int).flatten()
        labels_np = labels.cpu().numpy()
        
        # Metrics
        metrics = {
            'loss': loss.item(),
            'f1': f1_score(labels_np, preds, zero_division=0),
            'recall': recall_score(labels_np, preds, zero_division=0),
            'precision': precision_score(labels_np, preds, zero_division=0),
            'accuracy': (preds == labels_np).mean()
        }
        
        try:
            metrics['auc'] = roc_auc_score(labels_np, probs)
            metrics['avg_precision'] = average_precision_score(labels_np, probs)
        except ValueError:
            metrics['auc'] = 0.0
            metrics['avg_precision'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(labels_np, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positive'] = int(tp)
            metrics['true_negative'] = int(tn)
            metrics['false_positive'] = int(fp)
            metrics['false_negative'] = int(fn)
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return metrics
    
    def train_increment(
        self,
        data: Any,
        epochs: int = 50,
        train_mask: Optional[torch.Tensor] = None,
        val_mask: Optional[torch.Tensor] = None,
        verbose: bool = True,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train on a new data increment.
        
        Handles:
        1. Prototype initialization/update
        2. Replay buffer management
        3. Training loop with validation
        4. Early stopping
        
        Args:
            data: PyG Data object
            epochs: Number of training epochs
            train_mask: Training node mask
            val_mask: Validation node mask
            verbose: Print progress
            early_stopping_patience: Patience for early stopping
        
        Returns:
            Training history dictionary
        """
        # Move data to device for prototype initialization
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        y = data.y.to(self.device)
        
        # Initialize/update prototypes with minority samples
        # Auto-detect minority class (fewer samples)
        unique, counts = torch.unique(y, return_counts=True)
        if len(unique) >= 2:
            minority_label = unique[counts.argmin()].item()
        else:
            minority_label = 1  # fallback
        
        minority_mask = y == minority_label
        if minority_mask.sum() > 0:
            self.model.eval()
            with torch.no_grad():
                _, embeddings = self.model(
                    x, edge_index,
                    return_embeddings=True
                )
                minority_embeddings = embeddings[minority_mask]
                
                # Initialize or update prototypes
                if not self.model.prototype_attention.initialized:
                    self.model.initialize_prototypes(minority_embeddings)
                    if verbose:
                        print(f"Initialized {self.model.prototype_attention.n_prototypes} "
                              f"prototypes from {minority_mask.sum().item()} minority samples")
                else:
                    self.model.update_for_increment(minority_embeddings)
                    if verbose:
                        print(f"Updated prototypes with {minority_mask.sum().item()} "
                              f"new minority samples")
                
                # Add to replay buffer
                self.model.add_to_replay_buffer(
                    data.x[minority_mask.cpu()],
                    data.y[minority_mask.cpu()]
                )
        
        # Training loop
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(
                data, epoch, epochs, train_mask
            )
            
            # Update scheduler
            self.scheduler.step()
            
            # Record training metrics
            for key, value in train_metrics.items():
                self.history[f'train_{key}'].append(value)
            
            # Validate
            if val_mask is not None:
                val_metrics = self.evaluate(data, val_mask)
                
                for key, value in val_metrics.items():
                    self.history[f'val_{key}'].append(value)
                
                # Early stopping check
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    patience_counter = 0
                    self.best_model_state = {
                        k: v.cpu().clone() 
                        for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
            
            epoch_time = time.time() - start_time
            self.epoch_times.append(epoch_time)
            
            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                log_str = (f"Epoch {epoch+1}/{epochs} | "
                          f"Loss: {train_metrics['loss']:.4f} | "
                          f"F1: {train_metrics['f1']:.4f} | "
                          f"Recall: {train_metrics['recall']:.4f}")
                
                if val_mask is not None:
                    log_str += (f" || Val F1: {val_metrics['f1']:.4f} | "
                               f"Val Recall: {val_metrics['recall']:.4f}")
                
                print(log_str)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return dict(self.history)
    
    def get_per_class_metrics(
        self,
        data: Any,
        mask: Optional[torch.Tensor] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class detection metrics.
        
        Args:
            data: PyG Data object
            mask: Optional evaluation mask
            class_names: Optional class name mapping
        
        Returns:
            Per-class metrics dictionary
        """
        self.model.eval()
        
        with torch.no_grad():
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            y = data.y
            
            logits = self.model(x, edge_index)
            
            if mask is not None:
                logits = logits[mask]
                y = y[mask]
            
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int).flatten()
            labels = y.cpu().numpy()
        
        # Per-class metrics
        unique_classes = np.unique(labels)
        results = {}
        
        for cls in unique_classes:
            cls_mask = labels == cls
            cls_preds = preds[cls_mask]
            cls_labels = labels[cls_mask]
            
            # For binary: compute detection rate
            if cls == 1:  # Minority/attack class
                detection_rate = (cls_preds == 1).mean()
            else:  # Majority/normal class
                detection_rate = (cls_preds == 0).mean()
            
            cls_name = class_names[cls] if class_names else f"Class_{cls}"
            results[cls_name] = {
                'count': int(cls_mask.sum()),
                'detection_rate': float(detection_rate),
                'correct': int((cls_preds == cls_labels).sum())
            }
        
        return results


class IncrementalTrainer(DyGATFRTrainer):
    """
    Extended trainer for multi-increment scenarios.
    
    Manages training across multiple data increments with
    proper prototype updates and replay management.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.increment_history: List[Dict[str, Any]] = []
    
    def train_increments(
        self,
        data_increments: List[Any],
        epochs_per_increment: int = 30,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Train on multiple data increments sequentially.
        
        Args:
            data_increments: List of PyG Data objects (one per increment)
            epochs_per_increment: Training epochs per increment
            verbose: Print progress
        
        Returns:
            List of per-increment results
        """
        all_results = []
        
        for i, data in enumerate(data_increments):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Training Increment {i+1}/{len(data_increments)}")
                print(f"{'='*50}")
            
            # Train on this increment
            history = self.train_increment(
                data,
                epochs=epochs_per_increment,
                verbose=verbose
            )
            
            # Evaluate
            final_metrics = self.evaluate(data)
            
            result = {
                'increment': i + 1,
                'history': history,
                'final_metrics': final_metrics,
                'replay_buffer_size': len(self.model.replay_buffer)
            }
            
            all_results.append(result)
            self.increment_history.append(result)
            
            if verbose:
                print(f"\nIncrement {i+1} Final Metrics:")
                print(f"  F1: {final_metrics['f1']:.4f}")
                print(f"  Recall: {final_metrics['recall']:.4f}")
                print(f"  Precision: {final_metrics['precision']:.4f}")
        
        return all_results
