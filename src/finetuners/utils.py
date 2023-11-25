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
from datasets.utils import disable_progress_bar
from transformers.generation.beam_constraints import DisjunctiveConstraint

# Import Modules
from src.utils import get_project_root

class MemoryUsageCallback(TrainerCallback):
    """Callback class to add GPU memory usage metrics to metric dicts."""
    #BUG: memory metric is the same for eval in/out domain
    
    def __init__(self, val_in_training=False):
        self.using_cuda = torch.cuda.is_available()
        self.reset_memory_stats()
        self.eval_started = False
        self.is_training = False
        self.eval_count = 0
        self.val_in_training = val_in_training

    def reset_memory_stats(self):
        if self.using_cuda:
            torch.cuda.reset_peak_memory_stats()

    def on_train_begin(self, args, state, control, **kwargs):
        self.reset_memory_stats()
        self.is_training = True
    
    def on_train_end(self, args, state, control, **kwargs):
        self.is_training = False
        
    def on_prediction_step(self, args, state, control, **kwargs):
        # Reset memory stats at first eval step, after training is complete
        if not self.eval_started and not self.is_training:
            self.reset_memory_stats()
            self.eval_started = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.using_cuda:
            # Determine if still in training phase
            if not self.is_training or control.should_training_stop == True:
                self.eval_count += 1
                
                # Check if validating during training
                if self.val_in_training:
                    is_final_train_log = self.eval_count == 3   # Final training log
                    is_final_eval_log = self.eval_count > 3 # In and out of domain eval logs
                else:
                    is_final_train_log = self.eval_count == 1
                    is_final_eval_log = self.eval_count > 1
                
                peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # Bytes to GB
                
                if is_final_train_log:
                    prefix = "train"
                elif is_final_eval_log:
                    prefix = "eval"
                    
                # Skip the last eval step during training
                if is_final_train_log or is_final_eval_log:
                    logs[prefix + "_peak_memory_gb"] = peak_memory
                    
                self.eval_started = False
            
class ReformatEvalMetricsCallback(TrainerCallback):
    """Callback class to reformat eval metrics labels."""
    
    def __init__(self):
        self.last_call = None
        self.is_training = False
        self.eval_count = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.is_training = True
                
    def on_train_end(self, args, state, control, **kwargs):
        self.is_training = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Determine if still in training phase
        if not self.is_training:
            # This assumes that the 1st eval after training is in domain and the 2nd is out of domain
            self.eval_count += 1
            if self.eval_count == 1:
                infix = "in"
            elif self.eval_count == 2:
                infix = "out"
            logs = reformat_eval_metrics(logs, infix)
            
def reformat_eval_metrics(logs, infix):
    """Reformats the metrics dict to 'eval_in_' or 'eval_out_'"""
    for key in list(logs.keys()):
        if key.startswith('eval'):
            new_key = key.replace('eval', f'eval_{infix}')
            logs[new_key] = logs.pop(key)

def apply_minimal_pattern(dataset):
    """Apply the minimal pattern '{premise} {hypothesis}?'. Currently supports MNLI and HANS."""
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

def compute_metrics_causal(predictions, tokenizer, label_map):
    """Compute evaluation metrics for a causal language model."""
    predicted_token_ids, labels = predictions
    labels = labels.squeeze()

    decoded_predictions = [tokenizer.decode(pred_ids, skip_special_tokens=True) for pred_ids in predicted_token_ids]
    interpreted_predictions = [decoded_pred.strip().lower() for decoded_pred in decoded_predictions]
    predicted_labels = [label_map.get(pred.lower(), -1) for pred in interpreted_predictions]

    accuracy_metric = evaluate.load("accuracy")

    return accuracy_metric.compute(predictions=predicted_labels, references=labels)

def metrics_to_csv(metrics_dict, model_name, finetuning_method):
    """Write a dictionary of metrics to a csv."""
    filepath = os.path.join(get_project_root(), 'logs', f"{model_name}_{finetuning_method}_metrics.csv")
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
                
def training_histories_to_csv(training_histories, model_name, finetuning_method):
    """Write training histories to a csv."""
    filepath = os.path.join(get_project_root(), 'logs', f"{model_name}_{finetuning_method}_training_history.csv")
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Header
        headers = ['model_name', 'sample_size', 'epoch', 'train_loss', 'val_loss']
        writer.writerow(headers)

        # Rows
        for sample_size, trials in training_histories.items():
            for trial in trials:
                for epoch in range(len(trial['train_loss'])):
                    row = [model_name, sample_size]
                    row.extend([epoch + 1,
                                trial['train_loss'][epoch],
                                trial['val_loss'][epoch]])
                    writer.writerow(row)
                    
def get_yes_no_constraint(tokenizer):
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)
    no_token_id = tokenizer.encode("No", add_special_tokens=False)
    force_words_ids = [yes_token_id, no_token_id]
    constraint = DisjunctiveConstraint(nested_token_ids=force_words_ids)
    return constraint