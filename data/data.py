"""
Import functions for datasets
https://huggingface.co/datasets/glue
https://huggingface.co/datasets/hans
"""

# Import libraries
import os
from datasets import load_dataset, load_from_disk

# Import modules
from utils import get_project_root


def get_mnli():
    """Download MNLI dataset to disk"""
    dataset = load_dataset("glue", "mnli")
    filepath = os.path.join(get_project_root(), 'data/mnli')
    dataset.save_to_disk(filepath)
    
def get_hans():
    """Download HANS dataset to disk"""
    dataset = load_dataset("hans")
    filepath = os.path.join(get_project_root(), 'data/hans')
    dataset.save_to_disk(filepath)
    
def get_cola():
    """Download COLA dataset to disk"""
    dataset = load_dataset("glue", "cola")
    filepath = os.path.join(get_project_root(), 'data/cola')
    dataset.save_to_disk(filepath)
    
def get_in_domain():
    """Returns huggingface Dataset for In Domain"""
    dataset_name = 'mnli'
    set_name = 'train'
    filepath = os.path.join(get_project_root(), 'data', dataset_name, set_name)
    in_domain = load_from_disk(filepath)
    return in_domain

def get_out_domain():
    """Returns hugginface Dataset for Out of Domain"""
    dataset_name = 'hans'
    set_name = 'validation'
    filepath = os.path.join(get_project_root(), 'data', dataset_name, set_name)
    out_domain = load_from_disk(filepath)
    return out_domain

if __name__ == "__main__":
    get_mnli()
    get_hans()
    get_cola()