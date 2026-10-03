"""Dual-task training loop and loss functions."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from dtgat_config import LABEL_SMOOTH, device
from dtgat_model import EarlyStopping


def label_smooth_bce(pred, target, smooth=LABEL_SMOOTH):

    target_smooth = target * (1.0 - smooth) + (1.0 - target) * smooth
    return F.binary_cross_entropy_with_logits(pred, target_smooth)


def train_dual_task(model, X_mask, adj,
                    train_pairs, val_pairs,
                    train_mask, train_true,
                    val_mask, val_true,
                    epochs=1500, lr=1e-3, w_link=1.0, w_recon=1.0):
    """
    Train the two tasks jointly.

    Each epoch optimizes link prediction and expression reconstruction with a
    weighted joint loss. Reconstruction loss uses the training mask, whereas
    validation R2 uses the validation mask.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=50, min_lr=1e-7)

    # Loss functions
    link_criterion = label_smooth_bce
    recon_criterion = nn.MSELoss()
    
    es = EarlyStopping(patience=130, mode='max')

    history = {
        'train_loss': [], 'train_link_loss': [], 'train_recon_loss': [],
        'val_loss': [], 'val_link_loss': [], 'val_recon_loss': [],
        'val_auc': [], 'val_aupr': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [],
        'val_r2': []
    }

    train_tensor = torch.tensor(train_pairs, dtype=torch.long).to(device)
    val_tensor = torch.tensor(val_pairs, dtype=torch.long).to(device)

    best_metrics = {}

    print("\n" + "=" * 70)
    print("Starting joint dual-task training")
    print("=" * 70)
    print(f"Task 1: link prediction (weight={w_link})")
    print(f"Task 2: expression reconstruction (weight={w_recon})")
    print(f"Label smoothing: smooth={LABEL_SMOOTH}")
    print(f"Decoder: {model.type} | Initial learning rate: {optimizer.param_groups[0]['lr']} | Weight decay: {optimizer.param_groups[0]['weight_decay']} | Scheduler factor: {scheduler.factor} | Scheduler patience: {scheduler.patience}")
    print("Link-prediction classification threshold: 0.60 (for Precision/Recall/F1)")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        optimizer.zero_grad()

        # Run both tasks in the same forward pass.
        link_pred, recon_x, embed = model(X_mask, adj, train_tensor[:, :2])

        # Task 1 loss: link prediction
        link_loss = link_criterion(link_pred, train_tensor[:, 2].float())

        # Task 2 loss: expression reconstruction on the training mask
        recon_pred = recon_x[train_mask[0], train_mask[1]]
        recon_loss = recon_criterion(recon_pred, train_true)

        # Weighted joint loss
        total_loss = w_link * link_loss + w_recon * recon_loss

        # Backpropagate through both tasks.
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            # Validation forward pass
            val_link_pred, val_recon_x, _ = model(X_mask, adj, val_tensor[:, :2])

            # Task 1 validation: link prediction
            val_link_loss = link_criterion(val_link_pred, val_tensor[:, 2].float())
            val_prob = torch.sigmoid(val_link_pred).cpu().numpy().reshape(-1)
            val_label = val_tensor[:, 2].cpu().numpy().reshape(-1)
            val_pred_label = (val_prob >= 0.6).astype(int)

            auc = roc_auc_score(val_label, val_prob) if len(np.unique(val_label)) > 1 else 0.5
            aupr = average_precision_score(val_label, val_prob) if len(np.unique(val_label)) > 1 else 0.5
            precision = precision_score(val_label, val_pred_label, zero_division=0)
            recall = recall_score(val_label, val_pred_label, zero_division=0)
            f1 = f1_score(val_label, val_pred_label, zero_division=0)

            # Task 2 validation: expression reconstruction on the validation mask
            val_recon_pred = val_recon_x[val_mask[0], val_mask[1]].cpu().numpy()
            true_np = val_true.cpu().numpy()
            r2 = r2_score(true_np, val_recon_pred)
            val_recon_loss = recon_criterion(torch.tensor(val_recon_pred).to(device), val_true)

            # Total validation loss
            val_total_loss = w_link * val_link_loss + w_recon * val_recon_loss

        # Select the best model by validation AUPR, then AUC, then link loss.
        if epoch == 1:
            update_best = True
            best_aupr = aupr
            best_auc_for_aupr = auc
            best_link_loss_for_aupr = val_link_loss.item()
        else:
            update_best = False
            if aupr > best_aupr + 1e-5:
                update_best = True
            elif abs(aupr - best_aupr) <= 1e-5:
                if auc > best_auc_for_aupr + 1e-5:
                    update_best = True
                elif abs(auc - best_auc_for_aupr) <= 1e-5:
                    if val_link_loss.item() < best_link_loss_for_aupr - 1e-5:
                        update_best = True

        if update_best:
            best_aupr = aupr
            best_auc_for_aupr = auc
            best_link_loss_for_aupr = val_link_loss.item()
            best_aupr_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                'epoch': epoch,
                'auc': auc, 'aupr': aupr,
                'precision': precision, 'recall': recall, 'f1': f1,
                'r2': r2,
                'link_loss': val_link_loss.item(),
                'recon_loss': val_recon_loss.item()
            }

        es(aupr, model)
        scheduler.step(val_link_loss)
        # Record training history.
        history['train_loss'].append(total_loss.item())
        history['train_link_loss'].append(link_loss.item())
        history['train_recon_loss'].append(recon_loss.item())
        history['val_loss'].append(val_total_loss.item())
        history['val_link_loss'].append(val_link_loss.item())
        history['val_recon_loss'].append(val_recon_loss.item())
        history['val_auc'].append(auc)
        history['val_aupr'].append(aupr)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)
        history['val_r2'].append(r2)

        # Print progress for every epoch.
        print(f"Epoch {epoch:4d} | "
              f"Train Loss {total_loss.item():.4f} (Link:{link_loss.item():.4f} Recon:{recon_loss.item():.4f}) | "
              f"Val Loss {val_total_loss.item():.4f} | "
              f"AUC {auc:.4f} AUPR {aupr:.4f} Precision {precision:.4f} Recall {recall:.4f} F1 {f1:.4f} R2 {r2:.4f}")

        # Early stopping
        if es.stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            if best_metrics:
                print(f"Best model selected from epoch {best_metrics['epoch']}:")
                print(f"  Link prediction: AUC={best_metrics['auc']:.4f} AUPR={best_metrics['aupr']:.4f} Precision={best_metrics['precision']:.4f} Recall={best_metrics['recall']:.4f} F1={best_metrics['f1']:.4f}")
                print(f"  Expression reconstruction: R2={best_metrics['r2']:.4f}")
            break

    model.load_state_dict(best_aupr_state)
    print("\nLoaded the best model selected by validation AUPR")
    return history, best_metrics

