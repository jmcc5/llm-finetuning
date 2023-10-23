"""
Utility functions for data loading and processing
"""

# Import Libraries


def remove_neutral_labels(dataset):
    """Remove all rows from the MNLI dataset where the label is 1."""
    
    filtered_dataset = dataset.filter(lambda example: example['label'] != 1)
    
    return filtered_dataset

def isolate_lexical_overlap(dataset):
    """Isolate rows with heuristic 'lexical_overlap' in HANS dataset."""
    
    filtered_dataset = dataset.filter(lambda example: example['heuristic'] == 'lexical_overlap')
    
    return filtered_dataset