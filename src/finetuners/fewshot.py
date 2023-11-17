"""
Few-shot fine-tuning method from “Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation”, Mosbach et al.
https://aclanthology.org/2023.findings-acl.779.pdf
https://huggingface.co/docs/transformers/training

Few-Shot Fine-tuning (FT):
- Few-shot: randomly sampled n in {2, 16, 32, 64, 128} examples.
- Minimal pattern: append question mark to each example.
- Verbalizer: "Yes" and "No" labels for NLI and QQP tasks.
- Fine-tuning: 40 epochs, learning rate of 1e-5, linear increase for initial 10% of steps, then constant.
"""

# Import Libraries
import os
import numpy as np
from transformers import TrainingArguments, Trainer, PrinterCallback
from tqdm.autonotebook import tqdm

# Import Modules
from src.finetuners.utils import apply_minimal_pattern, tokenize_dataset, compute_metrics, metrics_to_csv, training_histories_to_csv, MemoryUsageCallback, ReformatEvalMetricsCallback
from src.model.model import save_model, get_model
from src.utils import get_project_root


def fine_tune(model, tokenizer, train_dataset, eval_dataset_in, eval_dataset_out, val_in_training=True, verbose=True, disable_tqdm=None):
    """Few shot finetuning base method. Modifies model passed in."""
    # Verbalize and tokenize    
    train_dataset = apply_minimal_pattern(train_dataset)  # Apply minimal pattern
    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=512)  # Tokenize
    
    eval_dataset_in = apply_minimal_pattern(eval_dataset_in)
    eval_dataset_in = tokenize_dataset(eval_dataset_in, tokenizer, max_length=512)
    
    eval_dataset_out = apply_minimal_pattern(eval_dataset_out)
    eval_dataset_out = tokenize_dataset(eval_dataset_out, tokenizer, max_length=512)
    
    # Validation
    if len(eval_dataset_out) >= 50:
        val_samples_size = 10
    else:
        val_samples_size = len(eval_dataset_out)
    validation_dataset = eval_dataset_out.shuffle().select(range(val_samples_size))

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
        per_device_train_batch_size=32,
        evaluation_strategy='epoch' if val_in_training else 'no',
        logging_steps=1 if val_in_training else 500,
        seed=42,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        callbacks=[MemoryUsageCallback(val_in_training), ReformatEvalMetricsCallback],
    )
    
    if not verbose:
        trainer.remove_callback(PrinterCallback)

    # Train on in domain
    train_output = trainer.train()
    train_metrics = train_output.metrics
    
    # Evaluate on in domain
    eval_metrics_in = trainer.evaluate(eval_dataset=eval_dataset_in)
    
    # Evaluate on OOD
    eval_metrics_out = trainer.evaluate(eval_dataset=eval_dataset_out)
    
    combined_metrics = {**train_metrics, **eval_metrics_in, **eval_metrics_out}
    training_history = trainer.state.log_history[:-3]
    
    return combined_metrics, training_history
    
def batch_fine_tune(model_name, train_datasets, eval_dataset_in, eval_dataset_out, save_trials=False):
    """Function to perform few-shot fine-tuning with certain sized samples of a certain number of trials"""
    
    results = {size: [] for size in train_datasets.keys()}
    training_histories = {size: {'train_loss': [], 'val_loss': []} for size in train_datasets.keys()}
    
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
                'trial': trial_num,
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