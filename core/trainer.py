"""
Training loop for EDA-Net with EDT-specific metric logging.
"""

import torch
import torch.nn as nn
import time
import copy
from .utils import evaluate_model
from sklearn.metrics import f1_score, recall_score


def train_model(model, train_loader, val_loader, config, criterion, device,
                use_edt_loss=False):
    """
    Training loop for EDA-Net and ablation models.

    Logs standard metrics (loss, F1, recall) plus EDT-specific metrics
    (mean temperature, mean entropy) when available.

    Args:
        model: nn.Module (EDANet or ablation variant)
        train_loader: training DataLoader
        val_loader: validation DataLoader
        config: dict with training hyperparameters
        criterion: loss function (EDANetLoss or standard)
        device: torch.device
        use_edt_loss: if True, pass EDT info to criterion (for EDANetLoss)
    """
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_recall': [], 'val_recall': [],
        'lr': [],
        'mean_tau': [],       # EDT: average temperature per epoch
        'mean_entropy': [],   # EDT: average entropy per epoch
        'tau_std': [],        # EDT: temperature std (adaptation measure)
    }

    best_val_f1 = 0.0
    best_model_state = None
    epochs_no_improve = 0
    label_smooth = config.get('label_smoothing', 0.0)
    epochs = config.get('epochs', 100)
    patience = config.get('patience', 20)

    print(f"\n🚀 Training model on {device}...")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(epochs):
        # ---- Training Phase ----
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        epoch_taus, epoch_entropies = [], []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # Label smoothing
            if label_smooth > 0:
                y_smooth = y_batch * (1 - label_smooth) + 0.5 * label_smooth
            else:
                y_smooth = y_batch

            outputs = model(X_batch)

            # Compute loss (with optional EDT regularisation + prototype anchoring)
            if use_edt_loss and hasattr(model, 'last_edt_info') and model.last_edt_info is not None:
                anchor_loss = model.prototype_anchor_loss() if hasattr(model, 'prototype_anchor_loss') else None
                loss, _loss_components = criterion(outputs, y_smooth, model.last_edt_info,
                                                   prototype_anchor_loss=anchor_loss)
            else:
                loss = criterion(outputs, y_smooth)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

            # Collect predictions
            if getattr(model, 'output_logits', False):
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().flatten()
            else:
                preds = (outputs > 0.5).cpu().numpy().flatten()

            train_preds.extend(preds)
            train_labels.extend(y_batch.cpu().numpy().flatten())

            # Collect EDT metrics
            if hasattr(model, 'last_edt_info') and model.last_edt_info is not None:
                edt = model.last_edt_info.get('edt_attention', {})
                if 'mean_tau' in edt:
                    epoch_taus.append(edt['mean_tau'].cpu())
                if 'mean_entropy' in edt:
                    epoch_entropies.append(edt['mean_entropy'].cpu())

        scheduler.step()

        train_loss /= len(train_loader.dataset)
        train_f1 = f1_score(train_labels, train_preds, zero_division=0)
        train_recall = recall_score(train_labels, train_preds, zero_division=0)

        # ---- Validation Phase ----
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)

                if use_edt_loss and hasattr(model, 'last_edt_info') and model.last_edt_info is not None:
                    anchor_loss = model.prototype_anchor_loss() if hasattr(model, 'prototype_anchor_loss') else None
                    loss, _ = criterion(outputs, y_batch, model.last_edt_info,
                                        prototype_anchor_loss=anchor_loss)
                else:
                    loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)

                if getattr(model, 'output_logits', False):
                    preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().flatten()
                else:
                    preds = (outputs > 0.5).cpu().numpy().flatten()

                val_preds.extend(preds)
                val_labels.extend(y_batch.cpu().numpy().flatten())

        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)

        # ---- Record History ----
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        history['lr'].append(current_lr)

        # EDT metrics
        if epoch_taus:
            all_tau = torch.cat(epoch_taus, dim=0)
            history['mean_tau'].append(all_tau.mean().item())
            history['tau_std'].append(all_tau.std().item())
        else:
            history['mean_tau'].append(0.0)
            history['tau_std'].append(0.0)

        if epoch_entropies:
            all_ent = torch.cat(epoch_entropies, dim=0)
            history['mean_entropy'].append(all_ent.mean().item())
        else:
            history['mean_entropy'].append(0.0)

        # ---- Early Stopping ----
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            tau_str = f" | τ={history['mean_tau'][-1]:.3f}±{history['tau_std'][-1]:.3f}" if epoch_taus else ""
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"F1: {train_f1:.4f}/{val_f1:.4f} | "
                  f"Recall: {train_recall:.4f}/{val_recall:.4f}{tau_str}")

        if epochs_no_improve >= patience:
            print(f"\n  ⏹️  Early stopping at epoch {epoch+1} (best F1: {best_val_f1:.4f})")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    train_time = time.time() - start_time
    print(f"\n✓ Training completed in {train_time:.2f}s")
    print(f"  Best Validation F1: {best_val_f1:.4f}")

    return model, history
