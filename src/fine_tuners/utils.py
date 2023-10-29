"""
Utility functions for fine-tuning
"""

def apply_minimal_pattern(batch):
    """Apply the minimal pattern {premise} {hypothesis}. Currently supports MNLI."""
    batch['text'] = batch['premise'] + " " + batch['hypothesis'] + "?"
    # batch['label'] = "Yes" if batch['label'] == 1 else "No" TODO: do we need this?
    return batch