"""Shared runtime configuration and reproducibility settings for DTGAT."""

import numpy as np
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
INITIAL_EPOCHS_TO_PRINT = 5
EPOCH_PRINT_INTERVAL = 50
LABEL_SMOOTH = 0.15


def reset_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

