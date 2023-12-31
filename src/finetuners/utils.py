"""
Utility functions for fine-tuning
"""

# Import Libraries
import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import warnings
import evaluate
from transformers import TrainerCallback
from datasets.utils import disable_progress_bar
from transformers.generation.beam_constraints import DisjunctiveConstraint

# Import Modules
from src.utils import get_project_root


class MemoryUsageCallback(TrainerCallback):
    """Callback class to add GPU memory usage metrics to metric dicts."""
    #BUG: memory metric is the same for eval in/out domain
    #SOLVED: memory is highly dependent on batch size, not number of tokens
    
    def __init__(self, val_in_training=False):
        self.using_cuda = torch.cuda.is_available()
        self.reset_memory_stats()
        self.eval_started = False
        self.is_training = False
        self.log_count = 0
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
                self.log_count += 1
                
                # Check if validating during training
                if self.val_in_training:
                    is_final_train_log = self.log_count == 3   # Final training log
                    is_final_eval_log = self.log_count > 3 # In and out of domain eval logs
                else:
                    is_final_train_log = self.log_count == 1
                    is_final_eval_log = self.log_count > 1
                
                peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # Bytes to GB
                
                if is_final_train_log:
                    prefix = "train"
                elif is_final_eval_log:
                    # This assumes that the 1st eval after training is in domain and the 2nd is out of domain
                    if self.eval_count == 0:
                        prefix = "eval_in"
                    else:
                        prefix = "eval_out"
                    self.eval_count += 1
                    
                # Skip the last eval step during training
                if is_final_train_log or is_final_eval_log:
                    logs[prefix + "_peak_memory_gb"] = peak_memory
                    
                self.eval_started = False
            
class ReformatEvalMetricsCallback(TrainerCallback):
    """Callback class to reformat eval metrics labels."""
    
    def __init__(self):
        warnings.warn("ReformatEvalMetricsCallback should not be used. Use the built-in parameter for the evaluate method, 'metric_key_value', to add prefixes to eval sets.",
                      stacklevel=5)
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

def apply_minimal_pattern(dataset, context = ''):
    """Apply the minimal pattern '{premise} {hypothesis}?'. Currently supports MNLI and HANS."""
    def format_batch(batch):
        batch['text'] = [context + premise + " " + hypothesis + "?" for premise, hypothesis in zip(batch['premise'], batch['hypothesis'])]
        return batch
    
    # Add context
    if not context == '':
        context = context + " "
    disable_progress_bar()
    dataset = dataset.map(format_batch, batched=True)
    
    return dataset

def tokenize_dataset(dataset, tokenizer, max_length=512, padding_side=None):
    """Tokenize input dataset. Designed for use after minimal pattern is applied."""
    def tokenize_function(examples):
        tokenized_examples = tokenizer(examples['text'], 
                                       truncation=True, 
                                       padding='max_length', 
                                       max_length=max_length)
        return tokenized_examples
    
    # Set padding
    original_padding_side = tokenizer.padding_side
    if padding_side is not None:
        tokenizer.padding_side = padding_side
    
    disable_progress_bar() 
    dataset = dataset.map(tokenize_function, batched=True)
    tokenizer.padding_side = original_padding_side  # Restore original padding side
    
    return dataset

def compute_metrics(predictions):
    """Compute evaluation metrics."""
    metric = evaluate.load("accuracy")
    logits, labels = predictions
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def compute_metrics_causal(predicted_labels, actual_labels):
    """Compute accuracy and loss for CausalLM predictions."""
    total_loss = 0
    correct_predictions = 0

    for predicted_label, actual_label in zip(predicted_labels, actual_labels):
        # Convert to tensors
        predicted_tensor = torch.tensor([predicted_label], dtype=torch.float32)
        actual_tensor = torch.tensor([actual_label], dtype=torch.float32)

        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(predicted_tensor, actual_tensor)
        total_loss += loss.item()
        correct_predictions += int(predicted_label == actual_label)

    accuracy = correct_predictions / len(actual_labels)
    avg_loss = total_loss / len(actual_labels)

    return avg_loss, accuracy

def metrics_to_csv(metrics, finetuning_method, exp_label=None):
    """Write a list of metrics dictionaries to a csv."""
    if exp_label is not None:
        exp_label = '_' + exp_label
    else:
        exp_label = ''
    filepath = os.path.join(get_project_root(), 'logs', f"{finetuning_method}_metrics{exp_label}.csv")
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Header
        headers = metrics[0].keys()
        writer.writerow(headers)

        # Rows
        for metrics in metrics:
            writer.writerow(metrics.values())

def training_histories_to_csv(training_histories, finetuning_method, exp_label=None):
    """Write training histories to a csv."""
    if exp_label is not None:
        exp_label = '_' + exp_label
    else:
        exp_label = ''
    filepath = os.path.join(get_project_root(), 'logs', f"{finetuning_method}_training_history{exp_label}.csv")
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Header
        headers = ['model_name', 'sample_size', 'epoch', 'train_loss', 'val_loss']
        writer.writerow(headers)

        # Rows
        for trial in training_histories:
            model_name = trial['model_name']
            sample_size = trial['sample_size']
            for epoch, (train_loss, val_loss) in enumerate(zip(trial['train_loss'], trial['val_loss']), start=1):
                row = [model_name, sample_size, epoch, train_loss, val_loss]
                writer.writerow(row)
                
def get_yes_no_constraint(tokenizer):
    """Return a DisjunctiveConstraint constraining text generation to 'Yes' or 'No'."""
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)
    no_token_id = tokenizer.encode("No", add_special_tokens=False)
    force_words_ids = [yes_token_id, no_token_id]
    constraint = DisjunctiveConstraint(nested_token_ids=force_words_ids)
    return constraint

def interpret_generated_texts(generated_texts, actual_labels):
    """Interpret a list of decoded predictions."""
    predicted_labels = []

    for text, actual_label in zip(generated_texts, actual_labels):
        cleaned_text = text.strip().lower().rstrip(',')
        
        if 'yes' in cleaned_text:
            predicted_labels.append(0)  # Yes = entailment
        elif 'no' in cleaned_text:
            predicted_labels.append(1)  # No = contradiction
        else:
            predicted_labels.append(1-actual_label) # Unknown output = incorrect label

    return predicted_labels

def reformat_eval_metrics(logs, infix):
    """Reformats the metrics dict to 'eval_in_' or 'eval_out_'"""
    keys_to_modify = [k for k in logs.keys() if k.startswith('eval')]
    for key in keys_to_modify:
        new_key = f"{key[:4]}_{infix}{key[4:]}"
        logs[new_key] = logs.pop(key)

def select_random_subset(dataset, num_shots, seed=123):
    """Not in use"""

    if num_shots < 1:
        return [], []
        
    indices = np.random.choice(range(len(dataset)), size=num_shots, replace=False)

    return select_subset_by_idx(dataset, indices), indices

def select_subset_by_idx(dataset, indices):
    """Not in use"""
    subset = dataset.select(indices)
    return subset

def reset_memory_stats():
    """Reset cuda GPU memory stats"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
def get_peak_memory():
    """Return peak GPU memory usage"""
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # Bytes to GB
    else:
        peak_memory = 0

    return peak_memory

def get_teacher_context(dataset):
    question_prompt = """The question you are trying to solve is whether the premise and hypothesis are a contradiction or an entailment. 
The answer is No when the premise and hypothesis are a contradiction and the answer is Yes when the hypothesis and 
premise are an entailment \n """
    contradictory_explanation_prompt = """Explanation: The answer is No because the premise and hypothesis are a contradiction. \n """
    entailment_explanation_prompt = """Explanation: The answer is Yes because the premise and hypothesis are an entailment. \n """

    # labels 0, 1
    contradictory_example = select_random_example_by_label(dataset, quantity=1, label=1) # should be a dict?
    #print(contradictory_example)
    contradictory_input = "Example input: " + contradictory_example['premise'][0] + " " + contradictory_example['hypothesis'][0] + "?\n"
    contradictory_output = "Example output: No \n "
    entailment_example = select_random_example_by_label(dataset, quantity=1, label=0)
    entailment_input = "Example input: " + entailment_example['premise'][0] + " " + entailment_example['hypothesis'][0] + "?\n"
    entailment_output = "Example output: Yes \n " 
    end = "Now determine whether the following is a contradiction or an entailment.\n"

    context = question_prompt + contradictory_input + contradictory_output + contradictory_explanation_prompt + \
                                entailment_input + entailment_output + entailment_explanation_prompt + end

    return context

def select_random_example_by_label(dataset, quantity, label):
    if quantity < 1:
        return [], []
    
    disable_progress_bar()
    filtered_dataset = dataset.filter(lambda x: x["label"] == label)

    indices = np.random.choice(range(len(filtered_dataset)), size=quantity, replace=False)
    random_example = filtered_dataset.select(indices)
    #print(indices)
    return random_example
    
def distillation_loss(teacher_logits, student_logits, temp=1):
    kldivloss_func = torch.nn.KLDivLoss(reduction='batchmean')
    loss = temp ** 2 * kldivloss_func(
                F.log_softmax(student_logits / temp, dim=-1),
                F.softmax(teacher_logits / temp, dim=-1))
    
    return loss
