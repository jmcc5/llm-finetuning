"""
Import functions for datasets
https://huggingface.co/datasets/glue
https://huggingface.co/datasets/hans
"""

# Import libraries
import os
from datasets import load_dataset, load_from_disk

# Import modules
from src.utils import get_project_root

        
def download_dataset(dataset_name):
    """Load specified huggingface dataset and save to disk. MNLI, COLA and HANS supported."""
    if dataset_name == "mnli" or dataset_name == "cola":
        dataset = load_dataset("glue", dataset_name)
    elif dataset_name == "hans":
        dataset = load_dataset(dataset_name)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    filepath = os.path.join(get_project_root(), 'data', dataset_name)
    if not os.path.exists(filepath):
        dataset.save_to_disk(filepath)
    
def get_in_domain(dataset_name='mnli', set_name='train'):
    """Returns huggingface Dataset for In Domain"""
    filepath = os.path.join(get_project_root(), 'data', dataset_name, set_name)
    if not os.path.exists(filepath):
        download_dataset(dataset_name=dataset_name)
    in_domain = load_from_disk(filepath)
    return in_domain
        
def get_out_domain(dataset_name='hans', set_name='validation'):
    """Returns huggingface Dataset for Out of Domain"""
    filepath = os.path.join(get_project_root(), 'data', dataset_name, set_name)
    if not os.path.exists(filepath):
        download_dataset(dataset_name=dataset_name)
    out_domain = load_from_disk(filepath)
    return out_domain
