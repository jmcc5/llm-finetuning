"""
Utility functions for fine-tuning
"""

def tokenize_function(tokenizer, dataset):
    return tokenizer(dataset["text"], padding="max_length", padding=True, truncation=True)

def apply_minimal_pattern(batch):
    """Apply the minimal pattern {premise} {hypothesis}"""
    batch['text'] = batch['premise'] + " " + batch['hypothesis'] + "?"
    # batch['label'] = "Yes" if batch['label'] == 1 else "No" TODO: do we need this?
    return batch