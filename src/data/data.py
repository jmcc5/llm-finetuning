"""
Import functions for datasets
https://huggingface.co/datasets/glue
https://huggingface.co/datasets/hans

In Domain: MNLI and RTE
Out of Domain: HANS lexical overlap
"""

#TODO: remove neutral labels from MNLI (1)

# Import libraries
import os
from datasets import load_dataset, load_from_disk

# Import modules
from src.utils import get_project_root
from src.data.utils import remove_neutral_labels, isolate_lexical_overlap

        
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
    
    # Data processing
    in_domain = remove_neutral_labels(in_domain)
    #TODO: randomly select 1000?
    
    return in_domain
        
def get_out_domain(dataset_name='hans', set_name='validation'):
    """Returns huggingface Dataset for Out of Domain"""
    
    filepath = os.path.join(get_project_root(), 'data', dataset_name, set_name)
    if not os.path.exists(filepath):
        download_dataset(dataset_name=dataset_name)
    out_domain = load_from_disk(filepath)
    
    # Data processing
    out_domain = isolate_lexical_overlap(out_domain)
    
    return out_domain
