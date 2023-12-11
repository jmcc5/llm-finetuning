"""
Few-shot fine-tuning method from “Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation”, Mosbach et al.
https://aclanthology.org/2023.findings-acl.779.pdf
"""

# Import Libraries
import os
import torch
from transformers import TrainingArguments, Trainer, PrinterCallback
from tqdm.autonotebook import tqdm

# Import Modules
from src.finetuners.utils import apply_minimal_pattern, tokenize_dataset, compute_metrics, metrics_to_csv, training_histories_to_csv, MemoryUsageCallback
from src.model.model import save_model, get_model
from src.utils import get_project_root


def fine_tune(model, tokenizer, train_dataset, eval_dataset_in, eval_dataset_out, batch_size=8, val_in_training=True, verbose=True, disable_tqdm=None):
    """Few shot finetuning base method. Model parameters are updated."""
    # Verbalize and tokenize    
    train_dataset = apply_minimal_pattern(train_dataset)  # Apply minimal pattern
    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=512)  # Tokenize
    
    eval_dataset_in = apply_minimal_pattern(eval_dataset_in)
    eval_dataset_in = tokenize_dataset(eval_dataset_in, tokenizer, max_length=512)
    
    eval_dataset_out = apply_minimal_pattern(eval_dataset_out)
    eval_dataset_out = tokenize_dataset(eval_dataset_out, tokenizer, max_length=512)
    
    # Validation
    if len(eval_dataset_in) >= 50:
        val_samples_size = 10
    else:
        val_samples_size = len(eval_dataset_in)
    validation_dataset = eval_dataset_in.shuffle().select(range(val_samples_size))

    # Fine tuning arguments (Mosbach et al.)
    output_dir = os.path.join(get_project_root(), 'logs')
    if disable_tqdm is None:
        disable_tqdm = not verbose
        
    training_args = TrainingArguments(
        log_level='error' if not verbose else 'passive',
        disable_tqdm=disable_tqdm,
        output_dir=output_dir,
        num_train_epochs=40,
        learning_rate=1e-5,
        lr_scheduler_type='linear',
        warmup_ratio = 0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy='epoch' if val_in_training else 'no',
        logging_steps=1 if val_in_training else 500
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        callbacks=[MemoryUsageCallback(val_in_training)],
    )
    
    if not verbose:
        trainer.remove_callback(PrinterCallback)

    # Train on in domain
    train_output = trainer.train()
    train_metrics = train_output.metrics
    
    # Evaluate on in domain
    eval_metrics_in = trainer.evaluate(eval_dataset=eval_dataset_in, metric_key_prefix='eval_in')
    
    # Evaluate on OOD
    eval_metrics_out = trainer.evaluate(eval_dataset=eval_dataset_out, metric_key_prefix='eval_out')
    
    combined_metrics = {**train_metrics, **eval_metrics_in, **eval_metrics_out}
    training_history = trainer.state.log_history[:-3]
    
    return combined_metrics, training_history
    
def batch_fine_tune(model_names, train_datasets, eval_dataset_in, eval_dataset_out, exp_label=None, save_trials=False):
    """Few-shot fine-tuning for multiple models over dictionary of train_datasets with varying sample size."""
    
    metrics = []
    training_histories = []
    
    # Iterate over models
    for model_name in model_names:
        torch.cuda.empty_cache()
        # Iterate over few-shot trials
        for sample_size, trials in train_datasets.items():
            progress_bar = tqdm(trials, desc=f"{model_name} {sample_size}-shot")
            # Dynamic batch sizing
            if model_name == 'opt-350m' and sample_size >= 8:
                batch_size = int(32/sample_size)
            else:
                batch_size = int(8)
            
            for trial_num, dataset in enumerate(progress_bar):
                model, tokenizer = get_model(model_name, 'SequenceClassification')  # Load original model from disk
                metrics_trial, full_training_history = fine_tune(model=model, 
                                                                 tokenizer=tokenizer, 
                                                                 train_dataset=dataset, 
                                                                 eval_dataset_in=eval_dataset_in, 
                                                                 eval_dataset_out=eval_dataset_out, 
                                                                 batch_size=batch_size,
                                                                 val_in_training=True, 
                                                                 verbose=False) # Fine-tune
                
                metrics_trial = {'model_name': model_name,
                                 'sample_size': sample_size,
                                 **metrics_trial}
                metrics.append(metrics_trial)
                
                # Extract losses from training histories
                train_loss = [entry['loss'] for entry in full_training_history if 'eval_loss' not in entry]
                val_loss = [entry['eval_loss'] for entry in full_training_history if 'eval_loss' in entry]
                
                # Add training histories to dict
                training_history_trial = {
                    'model_name': model_name,
                    'sample_size': sample_size,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }
                training_histories.append(training_history_trial)
                
                # Save fine-tuned model to disk
                if save_trials:
                    trial_label = f"{model_name}/{sample_size}-shot/{model_name}_{sample_size}-shot_{trial_num}"
                    save_model(model, trial_label)
                
                progress_bar.set_postfix(metrics_trial)   # Update progress bar postfix
        
    # Write to csv
    metrics_to_csv(metrics=metrics, finetuning_method='fewshot', exp_label=exp_label)
    training_histories_to_csv(training_histories=training_histories, finetuning_method='fewshot', exp_label=exp_label)

    return metrics, training_histories