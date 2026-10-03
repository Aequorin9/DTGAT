# -*- coding: utf-8 -*-
"""DTGAT Arabidopsis training and evaluation entry point."""

import os
import sys
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split

from data_utils import (
    add_negative_samples,
    generate_adjacency_matrix,
    load_expression,
    load_regulation,
    mask_expression_split,
)
from dtgat_config import SEED, device, reset_seed
from dtgat_model import DTGAT
from evaluation_utils import evaluate_and_save
from plotting_utils import plot_history
from runtime_utils import Logger, print_runtime_environment
from training_utils import train_dual_task


def main():
    # Configuration
    repo_root = Path(__file__).resolve().parent.parent
    expr_path = repo_root / 'data' / 'Arabidopsis' / 'expression_matrix.csv'
    reg_path = repo_root / 'data' / 'Arabidopsis' / 'reference_network.csv'
    out_dir = repo_root / 'output'

    # Create the output directory before initializing the logger.
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Add a timestamp to avoid overwriting earlier logs.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(out_dir, f"training_{timestamp}.log")
    sys.stdout = Logger(log_file, sys.stdout, filter_epoch_progress=True)
    sys.stderr = Logger(log_file, sys.stderr)

    print(f"Log file created: {log_file}")
    print_runtime_environment()
    print("Input data files:")
    print(f"  Expression matrix: {Path(expr_path).name} ({expr_path})")
    print(f"  Regulatory network: {Path(reg_path).name} ({reg_path})")

    print("\n" + "=" * 70)
    print("DTGAT dual-task learning framework")
    print("=" * 70)
    print("Task 1: link prediction")
    print("Task 2: expression reconstruction")
    print("=" * 70 + "\n")

    # Load data.
    X, scaler, gene_names = load_expression(expr_path)
    n_nodes = X.shape[0]
    gene_to_idx = {name: i for i, name in enumerate(gene_names)}
    pos_pairs = load_regulation(reg_path, n_nodes, gene_to_idx)

    # Split positive edges into training, validation, and test sets.
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    train_pos, temp_pos = train_test_split(pos_pairs, test_size=1.0 - train_ratio, random_state=SEED)
    val_pos, test_pos = train_test_split(
        temp_pos,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=SEED
    )

    total_pos = len(pos_pairs)
    actual_train_ratio = len(train_pos) / total_pos if total_pos > 0 else 0
    actual_val_ratio = len(val_pos) / total_pos if total_pos > 0 else 0
    actual_test_ratio = len(test_pos) / total_pos if total_pos > 0 else 0

    print("\nDataset split:")
    print(f"  Requested ratio: train {train_ratio:.0%} / validation {val_ratio:.0%} / test {test_ratio:.0%}")
    print(f"  Actual ratio: train {actual_train_ratio:.2%} / validation {actual_val_ratio:.2%} / test {actual_test_ratio:.2%}")
    print(f"  Training set: {len(train_pos)} positive edges")
    print(f"  Validation set: {len(val_pos)} positive edges")
    print(f"  Test set: {len(test_pos)} positive edges")

    all_pos_set = set(map(tuple, pos_pairs))

    # Build the graph from training edges only.
    adj = generate_adjacency_matrix(train_pos, n_nodes)

    # Sample negative edges.
    train_pairs = add_negative_samples(train_pos, all_pos_set, ratio=1.0, seed=SEED + 11)
    val_pairs = add_negative_samples(val_pos, all_pos_set, ratio=1.0, seed=SEED + 12)
    test_pairs = add_negative_samples(test_pos, all_pos_set, ratio=1.0, seed=SEED + 13)

    print("\nAfter negative sampling:")
    print(f"  Training set: {len(train_pairs)} pairs (positive:negative = 1:1)")
    print(f"  Validation set: {len(val_pairs)} pairs (positive:negative = 1:1)")
    print(f"  Test set: {len(test_pairs)} pairs (positive:negative = 1:1)")

    # Prepare independent masks for expression reconstruction. The training
    # mask defines the reconstruction loss, the validation mask supports model
    # selection, and the test mask is reserved for final evaluation.
    X_mask, train_mask, train_true, val_mask, val_true, test_mask, test_true = \
        mask_expression_split(X, train_frac=0.15, val_frac=0.05, test_frac=0.10, seed=SEED)

    # Initialize the model.
    reset_seed(SEED)
    model = DTGAT(
        input_dim=X.shape[1],
        hidden1_dim=128,
        hidden2_dim=128,
        hidden3_dim=64,
        hidden4_dim=64,
        output_dim=64,
        num_head1=3,
        num_head2=3,
        num_head3=3,
        alpha=0.2,
        type='cosine',
        reduction='mean'
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel information:")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")
    print(f"  Decoder type: {model.type}")
    print(f"  Multi-head reduction: {model.reduction}")

    # Joint dual-task training
    history, best_metrics = train_dual_task(
        model, X_mask, adj,
        train_pairs, val_pairs,
        train_mask, train_true,
        val_mask, val_true,
        epochs=1500,
        lr=2e-3,
        w_link=1.0,
        w_recon=1.0
    )

    # Final evaluation on the 1:1 test set
    final_metrics = evaluate_and_save(
        model, X_mask, adj, test_pairs,
        test_mask, test_true, gene_names,
        train_pos, all_pos_set, out_dir
    )

    # Save the consolidated training figure only.
    plot_history(history, out_dir)

    # Summary
    print("\n" + "=" * 70)
    print("Dual-task training completed")
    print("=" * 70)
    print(f"Output directory: {out_dir}/")
    print("\nGenerated files:")
    print("  1. DTGAT.pth                           - Trained model")
    print("  2. predicted_network_all.csv           - All predicted TF-target relationships")
    print("     Note: filter Known_All=False to obtain novel predictions")
    print("  3. prediction_statistics.csv           - Predicted-network statistics")
    print("  4. reconstructed_expression.csv        - Reconstructed expression matrix")
    print("  5. dual_task_training_curves.png       - Dual-task training curves")
    print("  6. training_YYYYMMDD_HHMMSS.log        - Runtime and metric log")
    print("=" * 70)

    print("\nFinal performance:")
    print("=" * 70)
    if best_metrics:
        print("Best validation checkpoint selected by AUPR:")
        print(f"  Epoch {best_metrics['epoch']}")
        print(f"  Link prediction: AUC={best_metrics['auc']:.4f}, AUPR={best_metrics['aupr']:.4f}, Precision={best_metrics['precision']:.4f}, Recall={best_metrics['recall']:.4f}, F1={best_metrics['f1']:.4f}")
        print(f"  Expression reconstruction: R2={best_metrics['r2']:.4f}")
    print("\nTest set:")
    print(f"  Link prediction (1:1): AUC={final_metrics['auc']:.4f}, AUPR={final_metrics['aupr']:.4f}, Precision={final_metrics['precision']:.4f}, Recall={final_metrics['recall']:.4f}, F1={final_metrics['f1']:.4f}")
    print(f"  Expression reconstruction: R2={final_metrics['r2']:.4f}, Corr={final_metrics['corr']:.4f}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
