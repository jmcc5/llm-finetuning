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

def tokenize_dataset(dataset, tokenizer, max_length=512):
    """Tokenize input dataset. Designed for use after minimal pattern is applied."""
    def tokenize_function(examples):
        tokenized_examples = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
        return tokenized_examples
    
    dataset = dataset.map(tokenize_function, batched=True)
    
    return dataset

def compute_metrics(eval_pred):
    """Compute validation metrics."""
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
