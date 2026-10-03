"""Data loading, masking, graph construction, and negative sampling."""

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler

from dtgat_config import SEED, device


def load_expression(expression_path: str, n_top_genes=5000):

    if not os.path.exists(expression_path):
        raise FileNotFoundError(f"Expression file not found: {expression_path}")

    df = pd.read_csv(expression_path, index_col=0)

    gene_names = df.index.values.astype(str)
    data = df.values

    print(f"Expression data shape: {data.shape}")
    print(f"Loaded {len(gene_names)} gene names from expression file")

    scaler = StandardScaler()
    data = scaler.fit_transform(data).astype(np.float32)

    return torch.tensor(data, dtype=torch.float32).to(device), scaler, gene_names


def load_regulation(regulation_path: str, n_nodes: int, gene_to_idx: dict):

    if not os.path.exists(regulation_path):
        raise FileNotFoundError(f"Regulatory network file not found: {regulation_path}")

    df = pd.read_csv(regulation_path)
    print(f"Loaded {len(df)} raw pairs from regulation CSV.")

    tf_names = df['TF_ID'].values
    tg_names = df['Target_ID'].values

    pairs = []
    valid_names = 0

    for tf, tg in zip(tf_names, tg_names):
        if tf in gene_to_idx and tg in gene_to_idx:
            pairs.append([gene_to_idx[tf], gene_to_idx[tg]])
            valid_names += 1

    print(f"Found {valid_names} pairs with valid gene names in expression data.")

    if not pairs:
        print("Warning: no overlapping gene names were found between the expression matrix and regulatory network.")
        return []

    pairs = np.array(pairs, dtype=int)
    valid = (pairs[:, 0] < n_nodes) & (pairs[:, 1] < n_nodes)
    pairs = pairs[valid]
    print(f"Valid positive pairs after filtering: {len(pairs)}")
    return pairs.tolist()


def mask_expression_split(X, train_frac=0.15, val_frac=0.05, test_frac=0.10, seed=SEED):

    np.random.seed(seed)
    X_np = X.cpu().numpy()
    maskable = np.where(X_np != 0)
    n_total = len(maskable[0])

    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    n_test = int(n_total * test_frac)

    if n_train + n_val + n_test == 0:
        raise ValueError("The total masking fraction is too small or the data contain only zeros.")

    all_idx = np.random.permutation(n_total)
    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:n_train+n_val]
    test_idx = all_idx[n_train+n_val:n_train+n_val+n_test]

    # Convert to tensors
    train_rows = torch.tensor(maskable[0][train_idx], dtype=torch.long, device=device)
    train_cols = torch.tensor(maskable[1][train_idx], dtype=torch.long, device=device)
    val_rows = torch.tensor(maskable[0][val_idx], dtype=torch.long, device=device)
    val_cols = torch.tensor(maskable[1][val_idx], dtype=torch.long, device=device)
    test_rows = torch.tensor(maskable[0][test_idx], dtype=torch.long, device=device)
    test_cols = torch.tensor(maskable[1][test_idx], dtype=torch.long, device=device)

    # Create masked X with ALL masked positions zeroed
    all_rows = torch.cat([train_rows, val_rows, test_rows])
    all_cols = torch.cat([train_cols, val_cols, test_cols])
    masked_X = X.clone()
    masked_X[all_rows, all_cols] = 0.0

    # Get true values from original X (before zeroing)
    train_true = X[train_rows, train_cols].clone()
    val_true = X[val_rows, val_cols].clone()
    test_true = X[test_rows, test_cols].clone()

    train_mask = torch.stack([train_rows, train_cols])
    val_mask = torch.stack([val_rows, val_cols])
    test_mask = torch.stack([test_rows, test_cols])

    print(f"Masked {n_train} train / {n_val} val / {n_test} test entries "
          f"({train_frac:.1%}/{val_frac:.1%}/{test_frac:.1%})")
    return masked_X, train_mask, train_true, val_mask, val_true, test_mask, test_true


def generate_adjacency_matrix(labeled_pairs, num_nodes):
    row, col = [], []
    for tf, tg in labeled_pairs:
        row.extend([tf, tg])
        col.extend([tg, tf])

    if not row:
        raise ValueError("No valid edges were provided.")

    # Add self-loops.
    adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    adj = (adj + sp.eye(num_nodes, format='coo')).tocoo()

    # Convert to a sparse tensor.
    indices = torch.from_numpy(np.vstack([adj.row, adj.col])).long()
    values = torch.FloatTensor(adj.data)
    adj_sparse = torch.sparse_coo_tensor(indices, values, size=adj.shape, device=device)

    print(f"Number of edges (including self-loops): {adj.nnz}")
    return adj_sparse


def add_negative_samples(pos_list, all_pos_set, ratio=1.0, seed=None):
    """Add sampled negative edges to a list of positive edges."""
    if seed is not None:
        np.random.seed(seed)
    labeled = [(tf, tg, 1) for tf, tg in pos_list]

    tf_set = np.unique([p[0] for p in pos_list])
    tg_set = np.unique([p[1] for p in pos_list])

    neg_samples = []
    avoid = all_pos_set.copy()

    n_neg = int(len(pos_list) * ratio)
    attempts = 0
    max_attempts = n_neg * 10

    while len(neg_samples) < n_neg and attempts < max_attempts:
        attempts += 1
        tf = np.random.choice(tf_set)
        tg = np.random.choice(tg_set)
        if tf != tg and (tf, tg) not in avoid:
            neg_samples.append((tf, tg, 0))
            avoid.add((tf, tg))

    return labeled + neg_samples

