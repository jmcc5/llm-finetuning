"""
Utility functions for fine-tuning
"""

# Import Libraries
import numpy as np
import evaluate


def apply_minimal_pattern(dataset):
    """Apply the minimal pattern '{premise} {hypothesis}?'. Currently supports MNLI."""
    dataset['text'] = [premise + " " + hypothesis + "?" for premise, hypothesis in zip(dataset['premise'], dataset['hypothesis'])]
    # dataset['text'] = dataset['premise'] + " " + dataset['hypothesis'] + "?"
    # dataset['label'] = "Yes" if dataset['label'] == 1 else "No" TODO: do we need this?
    return dataset

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)