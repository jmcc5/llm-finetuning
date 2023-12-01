"""
Utility functions
"""

import torch
import warnings
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
    
def cuda_check():
    """Check local environment for cuda availability and print results."""
    cuda_available = False#torch.cuda.is_available()
    print(f"Cuda available: {cuda_available}")
    if not cuda_available:
        warnings.warn("Cuda not available in this environment. Experiments will run slowly on CPU. Update your pytorch+cuda installation by following the steps at https://pytorch.org/get-started/locally/.")