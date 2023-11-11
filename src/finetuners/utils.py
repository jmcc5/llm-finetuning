"""
Utility functions for fine-tuning
"""

# Import Libraries
import os
import csv
import numpy as np
import torch
import evaluate
from transformers import TrainerCallback

# Import Modules
from datasets.utils import disable_progress_bar
from src.utils import get_project_root

class MemoryUsageCallback(TrainerCallback):
    """Callback class to add GPU memory usage metrics to metric dicts."""
    
    def __init__(self):
        self.using_cuda = torch.cuda.is_available()
        self.reset_memory_stats()
        self.last_call = None

    def reset_memory_stats(self):
        if self.using_cuda:
            torch.cuda.reset_peak_memory_stats()

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_call = 'train'
        self.reset_memory_stats()
        
    def on_prediction_step(self, args, state, control, **kwargs):
        self.last_call = 'eval'

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.using_cuda:
            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # Bytes to GB
            prefix = f"{self.last_call}_"
            logs[prefix + "peak_memory_gb"] = peak_memory
            self.reset_memory_stats()
            
class ReformatEvalMetricsCallback(TrainerCallback):
    """Callback class to reformat eval metrics labels."""
    
    def __init__(self):
        self.last_call = None
        self.log_call_count = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_call = 'train'
        
    def on_prediction_step(self, args, state, control, **kwargs):
        self.last_call = 'eval'

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.log_call_count += 1
        if self.last_call == 'eval':
            if self.log_call_count == 2:
                infix = "in"
            if self.log_call_count == 3:
                infix = "out"
            logs = reformat_eval_metrics(logs, infix)

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
    """Compute evaluation metrics."""
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
                
def reformat_eval_metrics(logs, infix):
    """Reformats the metrics dict to 'eval_in_' or 'eval_out_'"""
    keys_to_modify = [k for k in logs.keys() if k.startswith('eval')]
    for key in keys_to_modify:
        new_key = f"{key[:4]}_{infix}{key[4:]}"
        logs[new_key] = logs.pop(key)
