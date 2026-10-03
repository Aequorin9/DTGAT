"""Logging and runtime-environment reporting utilities."""

import os
import platform
import sys
from pathlib import Path

import torch

from dtgat_config import (
    EPOCH_PRINT_INTERVAL,
    INITIAL_EPOCHS_TO_PRINT,
    device,
)


class Logger(object):
    def __init__(self, filename="log.txt", stream=sys.stdout, filter_epoch_progress=False):
        self.terminal = stream
        self.log = open(filename, "a", encoding='utf-8')
        self.filter_epoch_progress = filter_epoch_progress
        self._suppress_next_newline = False

    def _should_echo_to_terminal(self, message):
        if not self.filter_epoch_progress:
            return True

        if self._suppress_next_newline and message == "\n":
            self._suppress_next_newline = False
            return False

        stripped = message.lstrip()
        if stripped.startswith("Epoch"):
            parts = stripped.split()
            if len(parts) >= 2 and parts[1].isdigit():
                epoch = int(parts[1])
                show_epoch = (
                    epoch <= INITIAL_EPOCHS_TO_PRINT
                    or epoch % EPOCH_PRINT_INTERVAL == 0
                )
                if not show_epoch and not message.endswith("\n"):
                    self._suppress_next_newline = True
                return show_epoch

        self._suppress_next_newline = False
        return True

    def write(self, message):
        if self._should_echo_to_terminal(message):
            self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def print_runtime_environment():
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    virtual_env = os.environ.get('VIRTUAL_ENV')
    env_name = conda_env or (Path(virtual_env).name if virtual_env else 'No conda/venv environment detected')

    print("\n" + "=" * 70)
    print("Runtime environment")
    print("=" * 70)
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Environment: {env_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Compute device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        props = torch.cuda.get_device_properties(current_device)
        total_mem_gb = props.total_memory / 1024 ** 3
        print(f"GPU ID: {current_device}")
        print(f"GPU name: {gpu_name}")
        print(f"GPU memory: {total_mem_gb:.2f} GB")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    else:
        print("GPU: unavailable; using CPU")
    print("=" * 70 + "\n")

