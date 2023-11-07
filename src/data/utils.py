"""
Utility functions for data loading and processing
"""

# Import Libraries
import random


def remove_neutral_labels(dataset):
    """Remove all rows from the MNLI dataset where the label is 1."""
    
    def filter_labels(batch):
        return [label != 1 for label in batch['label']]
    
    def remap_labels(batch):
        return {'label': [1 if label == 2 else label for label in batch['label']]}
    
    filtered_dataset = dataset.filter(filter_labels, batched=True)  # Remove neutral labels 
    relabeled_dataset = filtered_dataset.map(remap_labels, batched=True)    # Map 2 to 1 to create binary distribution
    
    return relabeled_dataset

def isolate_lexical_overlap(dataset):
    """Isolate rows with heuristic 'lexical_overlap' in HANS dataset."""
    
    def filter_heuristic(batch):
        return [heuristic == 'lexical_overlap' for heuristic in batch['heuristic']]
    
    filtered_dataset = dataset.filter(filter_heuristic, batched=True)
    
    return filtered_dataset

def get_random_subsets(dataset, sample_sizes=[2, 16, 32, 64, 128], num_batches=10):
    """Returns a dictionary of randomly sampled datasets, indexed by sample_size"""
    
    subsets = {}

    # Loop through each sample size
    for size in sample_sizes:
        subsets[size] = []

        # Loop to create 'num_samples' random subsets
        for _ in range(num_batches):
            random_indices = random.sample(range(len(dataset)), size)
            subset = dataset.select(random_indices)
            subsets[size].append(subset)

    return subsets