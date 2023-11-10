"""
Utility functions for fine-tuning
"""

# Import Libraries
import os
import csv
import numpy as np
import evaluate

# Import Modules
from datasets.utils import disable_progress_bar
from src.utils import get_project_root


def apply_minimal_pattern(dataset):
    """Apply the minimal pattern '{premise} {hypothesis}?'. Currently supports MNLI."""   
    def format_batch(batch):
        batch['text'] = [premise + " " + hypothesis + "?" for premise, hypothesis in zip(batch['premise'], batch['hypothesis'])]
        return batch
    
    disable_progress_bar() 
    dataset = dataset.map(format_batch, batched=True)
    
    return dataset

def tokenize_dataset(dataset, tokenizer, max_length=512):
    """Tokenize input dataset. Designed for use after minimal pattern is applied."""
    def tokenize_function(examples):
        tokenized_examples = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
        return tokenized_examples
    
    disable_progress_bar() 
    dataset = dataset.map(tokenize_function, batched=True)
    
    return dataset

def compute_metrics(predictions):
    """Compute validation metrics."""
    metric = evaluate.load("accuracy")
    logits, labels = predictions
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def metrics_to_csv(metrics_dict, model_name, finetuning_method):
    """Write a dictionary of metrics to a csv."""
    
    filepath = os.path.join(get_project_root(), 'logs', f"{model_name}_{finetuning_method}")
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Header
        headers = ['model_name', 'sample_size']
        sample_size_keys = list(metrics_dict[next(iter(metrics_dict))][0].keys())
        headers.extend(sample_size_keys)
        writer.writerow(headers)

        # Rows
        for shots, results in metrics_dict.items():
            for result in results:
                row = [model_name, shots]
                row.extend(result.values())
                writer.writerow(row)