"""
Zero-shot model inference using a randomly initialized sequence classification head. WIP.
"""

# Import Libraries
import os
import numpy as np
from transformers import Seq2SeqTrainingArguments, TrainingArguments, Seq2SeqTrainer, Trainer, PrinterCallback, DisjunctiveConstraint
from tqdm.autonotebook import tqdm

# Import Modules
from src.finetuners.utils import apply_minimal_pattern, tokenize_dataset, compute_metrics, metrics_to_csv, training_histories_to_csv, get_yes_no_constraint, MemoryUsageCallback, ReformatEvalMetricsCallback
from src.model.model import save_model, get_model
from src.utils import get_project_root


def evaluate(model, tokenizer, eval_dataset_in, eval_dataset_out, verbose=True, disable_tqdm=None):
    """Zero shot inference."""
    #TODO: make a zeroshot_niave that just uses SequenceClassification
    # Verbalize and tokenize
    eval_dataset_in = apply_minimal_pattern(eval_dataset_in)    #TODO: add a new function to handle adding all 
    eval_dataset_in = tokenize_dataset(eval_dataset_in, tokenizer, max_length=512)
    
    eval_dataset_out = apply_minimal_pattern(eval_dataset_out)
    eval_dataset_out = tokenize_dataset(eval_dataset_out, tokenizer, max_length=512)

    # Fine tuning arguments (Mosbach et al.)
    output_dir = os.path.join(get_project_root(), 'logs')
    if disable_tqdm is None:
        disable_tqdm = not verbose
        
    training_args = Seq2SeqTrainingArguments(
        log_level='error' if not verbose else 'passive',
        disable_tqdm=disable_tqdm,
        output_dir=output_dir,
        per_device_eval_batch_size=32,
        predict_with_generate=True, # Enable text generation
        auto_find_batch_size=True,
        seed=42,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        # compute_metrics=compute_metrics,
    )
    
    if not verbose:
        trainer.remove_callback(PrinterCallback)
        
    yes_no_constraint = get_yes_no_constraint(tokenizer)    # Get constraint
    
    # Evaluate on in domain
    eval_metrics_in = trainer.evaluate(eval_dataset=eval_dataset_in.with_format("torch"),
                                      num_beams=2,
                                      max_new_tokens=3,  
                                      # temperature=0.5,
                                      constraints=[yes_no_constraint])
    
    # Evaluate on OOD
    eval_metrics_out = trainer.evaluate(eval_dataset=eval_dataset_out.with_format("torch"),
                                       num_beams=2,
                                       max_new_tokens=3,  
                                       # temperature=0.5,
                                       constraints=[yes_no_constraint])
    
    combined_metrics = {**eval_metrics_in, **eval_metrics_out}
    
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