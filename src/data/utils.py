"""
Utility functions for data loading and processing
"""


def remove_neutral_labels(dataset):
    """Remove all rows from the MNLI dataset where the label is 1."""
    
    def filter_batch(batch):
        return [label != 1 for label in batch['label']]
    
    def remap_batch(batch):
        return {'label': [1 if label == 2 else label for label in batch['label']]}
    
    filtered_dataset = dataset.filter(filter_batch, batched=True)  # Remove neutral labels 
    relabeled_dataset = filtered_dataset.map(remap_batch, batched=True)    # Map 2 to 1 to create binary distribution
    
    return relabeled_dataset

def isolate_lexical_overlap(dataset):
    """Isolate rows with heuristic 'lexical_overlap' in HANS dataset."""
    
    def filter_batch(batch):
        return [heuristic == 'lexical_overlap' for heuristic in batch['heuristic']]
    
    filtered_dataset = dataset.filter(filter_batch, batched=True)
    
    return filtered_dataset
