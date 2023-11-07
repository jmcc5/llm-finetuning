"""
Utility functions for fine-tuning
"""

# Import Libraries
import numpy as np
import evaluate


def apply_minimal_pattern(batch):
    """Apply the minimal pattern {premise} {hypothesis}. Currently supports MNLI."""
    batch['text'] = batch['premise'] + " " + batch['hypothesis'] + "?"
    # batch['label'] = "Yes" if batch['label'] == 1 else "No" TODO: do we need this?
    return batch

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)