"""
Utility functions for fine-tuning
"""

# Import Libraries
import numpy as np
import evaluate


def apply_minimal_pattern(dataset):
    """Apply the minimal pattern '{premise} {hypothesis}?'. Currently supports MNLI."""
    
    def format_batch(batch):
        # Apply the minimal pattern to the entire batch and return the modified batch
        batch['text'] = [premise + " " + hypothesis + "?" for premise, hypothesis in zip(batch['premise'], batch['hypothesis'])]
        return batch

    dataset = dataset.map(format_batch, batched=True)
    
    return dataset

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)