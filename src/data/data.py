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
import numpy as np
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
    """Returns huggingface Datasets for In Domain. Default dataset is MNLI. 80/20 train/test split."""
    
    filepath_train = os.path.join(get_project_root(), 'data', dataset_name, 'train')
    filepath_val = os.path.join(get_project_root(), 'data', dataset_name, 'validation_matched')
    if not os.path.exists(filepath_train):
        download_dataset(dataset_name=dataset_name)
    in_domain_train = load_from_disk(filepath_train)
    in_domain_val = load_from_disk(filepath_val)
    
    # Data processing
    in_domain_train = remove_neutral_labels(in_domain_train)
    in_domain_val = remove_neutral_labels(in_domain_val)
    
    return in_domain_train, in_domain_val
        
def get_out_domain(dataset_name='hans', set_name='validation'):
    """Returns huggingface Dataset for Out of Domain"""
    
    filepath = os.path.join(get_project_root(), 'data', dataset_name, set_name)
    if not os.path.exists(filepath):
        download_dataset(dataset_name=dataset_name)
    out_domain = load_from_disk(filepath)
    
    # Data processing
    out_domain = isolate_lexical_overlap(out_domain)
    
    return out_domain

def get_random_subsets(train_dataset, eval_dataset_in, eval_dataset_out, train_sample_sizes=[2, 16, 32, 64, 128], num_trials=10, eval_sample_size=50):
    """Returns a dictionary of a list of randomly sampled datasets, indexed by sample_size"""
    #TODO: need to seed np.random somewhere
    
    train_datasets = {}

    # Loop through each sample size
    for size in train_sample_sizes:
        train_datasets[size] = []

        # Loop to create 'num_samples' random train_datasets
        for _ in range(num_trials):
            random_indices = np.random.choice(range(len(train_dataset)), size, replace=False)
            subset = train_dataset.select(random_indices)
            train_datasets[size].append(subset)
            
    # Generate eval datasets
    random_indices = np.random.choice(range(len(eval_dataset_in)), eval_sample_size, replace=False)
    eval_in = eval_dataset_in.select(random_indices)
    
    random_indices = np.random.choice(range(len(eval_dataset_out)), eval_sample_size, replace=False)
    eval_out = eval_dataset_out.select(random_indices)

    return train_datasets, eval_in, eval_out
