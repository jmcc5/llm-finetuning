"""
Import functions for glue datasets (https://huggingface.co/datasets/glue)
"""

# Import libraries
import os
from datasets import load_dataset

# Import modules
from utils import get_project_root


def get_mnli():
    dataset = load_dataset("glue", "mnli")
    filepath = os.path.join(get_project_root(), 'data/mnli')
    dataset.save_to_disk(filepath)
    
def get_hans():
    dataset = load_dataset("hans")
    filepath = os.path.join(get_project_root(), 'data/hans')
    dataset.save_to_disk(filepath)
    
def get_mnli():
    dataset = load_dataset("glue", "cola")
    filepath = os.path.join(get_project_root(), 'data/cola')
    dataset.save_to_disk(filepath)

if __name__ == "__main__":
    get_mnli()
    get_hans()
    get_mnli()