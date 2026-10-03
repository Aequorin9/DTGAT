"""Final evaluation and output serialization."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from dtgat_config import LABEL_SMOOTH, device


def evaluate_and_save(model, X_mask, adj, test_pairs, test_mask, test_true,
                      gene_names, train_pos_pairs, all_pos_set, output_dir):
    """Evaluate both tasks; reconstruction metrics use only the test mask."""
    model.eval()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    test_tensor = torch.tensor(test_pairs, dtype=torch.long).to(device)

    with torch.no_grad():
        link_pred, recon_x, embed = model(X_mask, adj, test_tensor[:, :2])

        # Task 1: link prediction
        prob = torch.sigmoid(link_pred).cpu().numpy().reshape(-1)
        label_np = test_tensor[:, 2].cpu().numpy().reshape(-1)
        pred_label = (prob >= 0.6).astype(int)
        auc = roc_auc_score(label_np, prob) if len(np.unique(label_np)) > 1 else 0.5
        aupr = average_precision_score(label_np, prob) if len(np.unique(label_np)) > 1 else 0.5
        precision = precision_score(label_np, pred_label, zero_division=0)
        recall = recall_score(label_np, pred_label, zero_division=0)
        f1 = f1_score(label_np, pred_label, zero_division=0)

        # Task 2: expression reconstruction on the test mask only
        recon_pred = recon_x[test_mask[0], test_mask[1]].cpu().numpy()
        true_np = test_true.cpu().numpy()
        r2 = r2_score(true_np, recon_pred)
        rmse = np.sqrt(mean_squared_error(true_np, recon_pred))
        mae = mean_absolute_error(true_np, recon_pred)
        corr = np.corrcoef(true_np, recon_pred)[0, 1]

    print("\n" + "=" * 70)
    print("Final test-set evaluation (1:1)")
    print("=" * 70)
    print("[Task 1 - Link prediction]")
    print(f"  AUC:       {auc:.4f}")
    print(f"  AUPR:      {aupr:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 score:  {f1:.4f}")
    print("  Classification threshold: 0.60")
    print("\n[Task 2 - Expression reconstruction]")
    print(f"  R2:   {r2:.4f}")
    print(f"  Corr: {corr:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print("=" * 70)

    final_metrics = {
        'auc': auc,
        'aupr': aupr,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'corr': corr
    }
    # Predict the complete network over all TF-target candidates.
    all_pos_list = list(all_pos_set)
    tf_list = np.unique([p[0] for p in all_pos_list])
    target_list = np.unique([p[1] for p in all_pos_list])
    all_pairs = [(tf, tg) for tf in tf_list for tg in target_list if tf != tg]
    pos_pairs_set = set(map(tuple, train_pos_pairs))

    batch_size = 20000
    pred_scores = []
    pred_pairs = []

    with torch.no_grad():
        for i in range(0, len(all_pairs), batch_size):
            batch_pairs = all_pairs[i:i + batch_size]
            batch_tensor = torch.tensor(batch_pairs, dtype=torch.long).to(device)

            logits, _, _ = model(X_mask, adj, batch_tensor)
            pred_scores.extend(torch.sigmoid(logits).cpu().tolist())
            pred_pairs.extend(batch_pairs)

    net_df = pd.DataFrame({
        'TF_idx': [t for t, _ in pred_pairs],
        'Target_idx': [g for _, g in pred_pairs],
        'TF': [gene_names[t] for t, _ in pred_pairs],
        'Target': [gene_names[g] for _, g in pred_pairs],
        'Score': pred_scores
    })

    net_df['Known_Train'] = net_df.apply(lambda row: (row['TF_idx'], row['Target_idx']) in pos_pairs_set, axis=1)
    net_df['Known_All'] = net_df.apply(lambda row: (row['TF_idx'], row['Target_idx']) in all_pos_set, axis=1)
    net_df = net_df.sort_values('Score', ascending=False)
    net_df.to_csv(f"{output_dir}/predicted_network_all.csv", index=False)

    net_df_known_train = net_df[net_df['Known_Train']]
    net_df_known_heldout = net_df[(~net_df['Known_Train']) & net_df['Known_All']]
    net_df_novel = net_df[~net_df['Known_All']]

    # Prediction statistics
    n_total = len(net_df)
    n_known_train = len(net_df_known_train)
    n_known_heldout = len(net_df_known_heldout)
    n_novel = len(net_df_novel)
    print("\n" + "=" * 70)
    print("Predicted network statistics")
    print("=" * 70)
    print(f"All predicted TF-target pairs:                    {n_total:>6}")
    print(f"  Known training edges (Known_Train=True):        {n_known_train:>6} ({n_known_train / n_total * 100:>5.1f}%)")
    print(f"  Known held-out edges (Known_All & ~Train):      {n_known_heldout:>6} ({n_known_heldout / n_total * 100:>5.1f}%)")
    print(f"  Novel predictions (~Known_All):                 {n_novel:>6} ({n_novel / n_total * 100:>5.1f}%)")
    print("=" * 70)

    # Save reconstructed expression values.
    pd.DataFrame(recon_x.cpu().numpy(), index=gene_names).to_csv(
        f"{output_dir}/reconstructed_expression.csv"
    )
    print(f"Reconstructed expression matrix saved to: {output_dir}/reconstructed_expression.csv")

    # Save prediction statistics.
    stats_df = pd.DataFrame({
        'Category': [
            'All predicted TF-target pairs',
            'Known training edges (Known_Train=True)',
            'Known held-out edges (Known_All & ~Train)',
            'Novel predictions (~Known_All)'
        ],
        'Count': [
            n_total,
            n_known_train,
            n_known_heldout,
            n_novel
        ],
        'Percentage_of_total(%)': [
            100.0,
            n_known_train / n_total * 100 if n_total > 0 else 0,
            n_known_heldout / n_total * 100 if n_total > 0 else 0,
            n_novel / n_total * 100 if n_total > 0 else 0
        ],
        'Percentage_within_category(%)': [
            100.0,
            100.0,
            100.0,
            100.0
        ]
    })
    stats_df.to_csv(f"{output_dir}/prediction_statistics.csv", index=False)
    print(f"\nPrediction statistics saved to: {output_dir}/prediction_statistics.csv")

    # Save the trained model.
    torch.save({
        'model_state_dict': model.state_dict(),
        'gene_names': gene_names.tolist(),
        'train_pos_pairs': train_pos_pairs
,
        'label_smooth': LABEL_SMOOTH
    }, f"{output_dir}/DTGAT.pth")

    return final_metrics

