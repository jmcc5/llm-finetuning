"""
Utility functions
"""

import torch
from pathlib import Path


def get_project_root() -> Path:
    """Return absolute path to project root. Modify if file is moved from root."""
    return Path(__file__).parent.parent

def print_gpu_memory_usage():
    """GPU memory usage logging for debugging."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)    # Convert bytes to GB

    print(f"Device: {device}")
    print(f"Memory Allocated: {allocated:.2f} GB")
    print(f"Memory Reserved: {reserved:.2f} GB")