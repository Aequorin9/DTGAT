"""Plotting utilities for the consolidated DTGAT training figure."""

from pathlib import Path

import matplotlib.pyplot as plt

from dtgat_config import LABEL_SMOOTH


def set_publication_plot_style():
    """Apply a consistent publication-oriented plotting style."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'grid.color': '#d9d9d9',
        'grid.linestyle': '-',
        'grid.linewidth': 0.7,
        'grid.alpha': 0.55,
        'font.size': 11,
        'axes.titlesize': 15,
        'axes.labelsize': 12,
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#dddddd',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight'
    })


def plot_history(history, output_dir):
    """Plot dual-task training curves with AUC and AUPR ranking metrics."""
    set_publication_plot_style()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Panel 1: total loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Total', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Val Total', linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title(f'Overall Training Loss  [smooth={LABEL_SMOOTH}]')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: task-specific losses
    ax = axes[0, 1]
    ax.plot(epochs, history['train_link_loss'], label='Train Link', linestyle='--', alpha=0.7)
    ax.plot(epochs, history['val_link_loss'], label='Val Link', linestyle='--', alpha=0.7)
    ax.plot(epochs, history['train_recon_loss'], label='Train Recon', linestyle=':', alpha=0.7)
    ax.plot(epochs, history['val_recon_loss'], label='Val Recon', linestyle=':', alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Task Loss')
    ax.set_title('Task-Specific Losses')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: link-prediction metrics

    ax = axes[1, 0]
    ax.plot(epochs, history['val_auc'], label='AUC', color='#2ca02c', linewidth=2.2)
    ax.plot(epochs, history['val_aupr'], label='AUPR', color='#ff7f0e', linewidth=2.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Task 1: Link Prediction Ranking Metrics')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4: expression-reconstruction metric
    ax = axes[1, 1]
    ax.plot(epochs, history['val_r2'], label='R2', color='purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R2 Score')
    ax.set_title('Task 2: Expression Reconstruction')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/dual_task_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {output_dir}/dual_task_training_curves.png")

