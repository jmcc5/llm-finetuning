"""
In Context Learning fine-tuning method from “Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation”, Mosbach et al.
https://aclanthology.org/2023.findings-acl.779.pdf
https://huggingface.co/docs/transformers/training

In-Context Learning (ICL):
- Few-shot: 16 demonstrations mainly, additional experiments with 2 and 32 demonstrations.
- Verbalizer: same as for fine-tuning.
- Prediction correctness: higher probability for correct label's verbalizer token compared to other.
"""

# Import Libraries
import os
import time
import torch
import numpy as np
from transformers import Seq2SeqTrainingArguments, TrainingArguments, Seq2SeqTrainer, Trainer, PrinterCallback, DisjunctiveConstraint
from tqdm.autonotebook import tqdm

# Import Modules
from src.finetuners.utils import apply_minimal_pattern, tokenize_dataset, compute_metrics_causal, metrics_to_csv, select_random_subset, select_subset_by_idx, get_yes_no_constraint, interpret_generated_texts, MemoryUsageCallback, ReformatEvalMetricsCallback
from src.model.model import save_model, get_model
from src.utils import get_project_root

def evaluate(model, tokenizer, eval_dataset_in, eval_dataset_out, batch_size=8, verbose=True, disable_tqdm=None):
    """In-context learning base method."""
    def evaluate_dataset(model, tokenizer, dataset, batch_size):
        start_time = time.time()
        predicted_labels = []
        yes_no_constraint = get_yes_no_constraint(tokenizer)
        
        progress_bar = tqdm(range(0, len(dataset), batch_size), disable=disable_tqdm)

        for i in progress_bar:
            # Create in-context learning prompt from training data

            context, context_indices = create_few_shot_context(eval_dataset_in)

            # Verbalize and tokenize batch
            batch_indices = range(i, min(i + batch_size, len(dataset)))
            batch = dataset.select(batch_indices)
            batch = apply_minimal_pattern(batch, context)
            tokenized_batch = tokenize_dataset(batch, tokenizer, max_length=512, padding_side='left')   # Use left padding for text generation (OPT is decoder only)
            
            # Convert to tensors
            input_ids = torch.tensor(tokenized_batch['input_ids'], device=model.device)
            attention_mask = torch.tensor(tokenized_batch['attention_mask'], device=model.device)
            
            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=3,   # Max tokens to generate in output
                constraints=[yes_no_constraint],    # Constrain output to 'Yes' or 'No'
                num_beams=2   # Use minimum number of beams to save compute
            )

            generated_texts = tokenizer.batch_decode(generated_tokens[:, input_ids.shape[1]:], skip_special_tokens=True)    # Decode generated tokens
            actual_labels_batch = [item['label'] for item in batch]   # Get actual labels from batch
            predicted_labels_batch = interpret_generated_texts(generated_texts, actual_labels_batch)    # Translate 'Yes'/'No' to 0 and 1
            predicted_labels.extend(predicted_labels_batch) # Log all predicted labels

        # Calculate metrics for all batches
        actual_labels = [item['label'] for item in dataset]   # Get actual labels from dataset
        avg_loss, accuracy = compute_metrics_causal(predicted_labels, actual_labels)    # Compute loss and accuracy for predictions
        end_time = time.time()
        runtime = end_time - start_time
        samples_per_second = len(dataset) / runtime
        
        # Log metrics
        metrics = {
            "loss": avg_loss, 
            "accuracy": accuracy, 
            "runtime": runtime, 
            "samples_per_second": samples_per_second
        }
        return metrics

    # Evaluate - batch size = 8 due to GPU memory constraints
    eval_metrics_in = evaluate_dataset(model, tokenizer, eval_dataset_in, batch_size=batch_size)    # In domain
    if verbose:
        print(f"In domain eval metrics:\n{eval_metrics_in}")
    eval_metrics_out = evaluate_dataset(model, tokenizer, eval_dataset_out, batch_size=batch_size)  # OOD
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

def create_few_shot_context(dataset, description=None, seperator=",", from_indices=None):
    """Create context for dataset."""
    # Hardcoded verbalizer for lables:
    id_to_token = ['entailment', 'contradiction']
    # select samples to construct context from
    if from_indices is None:
        demos, indices = select_random_subset(dataset, 8)
    else:
        demos, indices = select_subset_by_idx(dataset, from_indices)

    # create context
    context = "" if description == None else f"{description}{seperator}"

    for sample in demos:
        formatted_sample = f"{sample['premise']} {sample['hypothesis']} ?"
        context += f"{formatted_sample}{id_to_token[sample['label']]}{seperator}"

    return context, indices
