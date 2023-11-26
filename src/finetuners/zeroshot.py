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


def evaluate(model, tokenizer, eval_dataset_in, eval_dataset_out, max_retries=3, verbose=True, disable_tqdm=None):
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
            input_ids = torch.tensor(tokenized_batch['input_ids'], device=model.device)
            attention_mask = torch.tensor(tokenized_batch['attention_mask'], device=model.device)
            
            is_label_error = False
            for _ in range(max_retries):
                generated_tokens = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=2,
                    constraints=[yes_no_constraint],
                    num_beams=2
                )

                generated_texts = tokenizer.batch_decode(generated_tokens[:, input_ids.shape[1]:], skip_special_tokens=True)
                try:
                    predicted_labels_batch = interpret_generated_texts(generated_texts)
                    predicted_labels.extend(predicted_labels_batch)
                    if is_label_error:
                        print(f"Successful retry with labels {predicted_labels_batch}")
                    break
                except ValueError as e: # If generated label does not match 'Yes' or 'No', retry inference
                    is_label_error = True
                    if verbose:
                        print(f"Retry due to error: {e}")
                    continue

        # Calculate metrics for all batches
        actual_labels = [item['label'] for item in dataset]   # Get actual labels from dataset
        avg_loss, accuracy = compute_metrics_causal(predicted_labels, actual_labels)
        end_time = time.time()
        runtime = end_time - start_time
        samples_per_second = len(dataset) / runtime
        metrics = {
            "loss": avg_loss, 
            "accuracy": accuracy, 
            "runtime": runtime, 
            "samples_per_second": samples_per_second
        }
        return metrics
    
    # Evaluate - batch size = 8 due to GPU memory constraints
    eval_metrics_in = evaluate_dataset(model, tokenizer, eval_dataset_in, batch_size=8)    # In domain
    if verbose:
        print(f"In domain eval metrics:\n{eval_metrics_in}")
    eval_metrics_out = evaluate_dataset(model, tokenizer, eval_dataset_out, batch_size=8)  # OOD
    if verbose:
        print(f"Out of domain eval metrics:\n{eval_metrics_out}")
    
    combined_metrics = {f'eval_in_{k}': v for k, v in eval_metrics_in.items()}
    combined_metrics.update({f'eval_out_{k}': v for k, v in eval_metrics_out.items()})
    
    return combined_metrics
    
def batch_evaluate(model_name, eval_dataset_in, eval_dataset_out):
    """Function to perform zero-shot evaluation and log results."""
    # Load the model and tokenizer
    model, tokenizer = get_model(model_name, 'CausalLM', pretrained=True)

    # Evaluate the model
    eval_metrics = evaluate(model, tokenizer, eval_dataset_in, eval_dataset_out, verbose=False)

    # Create results dict
    sample_size = str(len(eval_dataset_in))
    metrics_dict = {
        sample_size: [eval_metrics]
    }
    
    # Write results to csv
    metrics_to_csv(metrics_dict=metrics_dict, model_name=model_name, finetuning_method='zeroshot')

    return eval_metrics