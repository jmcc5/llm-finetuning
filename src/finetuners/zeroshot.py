"""
Zero-shot model inference as a baseline for comparison with fine-tuned models.

- Each inference instance should be on a single example
"""

# Import Libraries
import os
import time
import torch
import numpy as np
from transformers import Seq2SeqTrainingArguments, TrainingArguments, Seq2SeqTrainer, Trainer, PrinterCallback, DisjunctiveConstraint
from tqdm.autonotebook import tqdm

# Import Modules
from src.finetuners.utils import apply_minimal_pattern, tokenize_dataset, compute_metrics_causal, metrics_to_csv, training_histories_to_csv, get_yes_no_constraint, interpret_generated_texts, MemoryUsageCallback, ReformatEvalMetricsCallback
from src.model.model import save_model, get_model
from src.utils import get_project_root


def evaluate(model, tokenizer, eval_dataset_in, eval_dataset_out, verbose=True, disable_tqdm=None):
    """Zero shot inference."""
    def evaluate_dataset(model, tokenizer, dataset, batch_size):
        start_time = time.time()
        predicted_labels = []
        
        yes_no_constraint = get_yes_no_constraint(tokenizer)
        
        progress_bar = tqdm(range(0, len(dataset), batch_size), disable=disable_tqdm)

        for i in progress_bar:
            # Verbalize and tokenize batch
            batch_indices = range(i, min(i + batch_size, len(dataset)))
            batch = dataset.select(batch_indices)
            batch = apply_minimal_pattern(batch)
            tokenized_batch = tokenize_dataset(batch, tokenizer, max_length=512, padding_side='left')   # Use left padding for text generation (OPT is decoder only)
            
            # Convert to tensors
            input_ids = torch.tensor(tokenized_batch['input_ids'])
            attention_mask = torch.tensor(tokenized_batch['attention_mask'])

            gen_tokens = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                constraints=[yes_no_constraint],
                num_beams=2
            )

            generated_texts = tokenizer.batch_decode(gen_tokens[:, input_ids.shape[1]:], skip_special_tokens=True)  # Decode only the predicted label
            predicted_labels_batch = interpret_generated_texts(generated_texts)   # Transform predictions into integers (1: Yes, 2: No)

            predicted_labels.extend(predicted_labels_batch)

        # Calculate metrics for all batches
        actual_labels = [item['label'] for item in dataset]   # Get actual labels from dataset
        avg_loss, accuracy = compute_metrics_causal(predicted_labels, actual_labels)
        end_time = time.time()
        runtime = end_time - start_time
        samples_per_second = len(dataset) / runtime
        metrics = {"loss": avg_loss, 
                   "accuracy": accuracy, 
                   "runtime": runtime, 
                   "samples_per_second": samples_per_second}
        return metrics

    
    # Evaluate
    eval_metrics_in = evaluate_dataset(model, tokenizer, eval_dataset_in, batch_size=32)    # In domain
    if verbose:
        print(f"In domain eval metrics:\n{eval_metrics_in}")
    eval_metrics_out = evaluate_dataset(model, tokenizer, eval_dataset_out, batch_size=32)  # OOD
    if verbose:
        print(f"Out of domain eval metrics:\n{eval_metrics_out}")
    
    combined_metrics = {f'eval_in_{k}': v for k, v in eval_metrics_in.items()}
    combined_metrics.update({f'eval_out_{k}': v for k, v in eval_metrics_out.items()})
    
    return combined_metrics
    
def batch_fine_tune(model_name, train_datasets, eval_dataset_in, eval_dataset_out, save_trials=False):
    """Function to perform few-shot fine-tuning with certain sized samples of a certain number of trials"""
    
    results = {size: [] for size in train_datasets.keys()}
    training_histories = {size: [] for size in train_datasets.keys()}
    
    # Iterate over few-shot trials
    for sample_size, trials in train_datasets.items():
        progress_bar = tqdm(trials, desc=f"{sample_size}-shot")
        
        for trial_num, dataset in enumerate(progress_bar):
            model, tokenizer = get_model(model_name, 'SequenceClassification')  # Load original model from disk
            metrics, full_training_history = fine_tune(model=model, tokenizer=tokenizer, train_dataset=dataset, eval_dataset_in=eval_dataset_in, eval_dataset_out=eval_dataset_out, val_in_training=True, verbose=False) # Fine-tune
            
            results[sample_size].append(metrics)   # Log results
            
            # Extract losses from training histories
            train_loss = [entry['loss'] for entry in full_training_history if 'eval_loss' not in entry]
            val_loss = [entry['eval_loss'] for entry in full_training_history if 'eval_loss' in entry]
            
            # Add training histories to dict
            trial_history = {
                'train_loss': train_loss,
                'val_loss': val_loss
                }
            training_histories[sample_size].append(trial_history)
            
            # Save fine-tuned model to disk
            if save_trials:
                trial_label = f"{model_name}/{sample_size}-shot/{model_name}_{sample_size}-shot_{trial_num}"
                save_model(model, trial_label)
            
            progress_bar.set_postfix(results[sample_size][trial_num])   # Update progress bar postfix
        
    # Write results to csv
    metrics_to_csv(metrics_dict=results, model_name=model_name, finetuning_method='fewshot')
    training_histories_to_csv(training_histories=training_histories, model_name=model_name, finetuning_method='fewshot')

    return results, training_histories