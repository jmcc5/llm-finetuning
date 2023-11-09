"""
Few-shot fine-tuning method from “Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation”, Mosbach et al.
https://aclanthology.org/2023.findings-acl.779.pdf
https://huggingface.co/docs/transformers/training

Few-Shot Fine-tuning (FT):
- Few-shot: randomly sampled n in {2, 16, 32, 64, 128} examples.
- Minimal pattern: append question mark to each example.
- Verbalizer: "Yes" and "No" labels for NLI and QQP tasks.
- Fine-tuning: 40 epochs, learning rate of 1e-5, linear increase for initial 10% of steps, then constant.

Randomly sample 10x subsets of examples with sizes in {2, 16, 32, 64, 128}.
30 runs for each sample size
Experiment with 3 different patterns for each set?
"""

# Import Libraries
import os
import torch
import numpy as np
from transformers import TrainingArguments, Trainer

# Import Modules
from src.finetuners.utils import apply_minimal_pattern, tokenize_dataset, compute_metrics
from src.data.utils import get_random_subsets
from src.model.model import save_model, get_model
from src.utils import get_project_root


def fine_tune(model, tokenizer, train_dataset, eval_dataset):
    """Few shot finetuning base method. Modifies model passed in."""
    # Verbalize and tokenize    
    train_dataset = apply_minimal_pattern(train_dataset)  # Apply minimal pattern
    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=512)  # Tokenize
    
    eval_dataset = apply_minimal_pattern(eval_dataset)
    eval_dataset = tokenize_dataset(eval_dataset, tokenizer, max_length=512)

    # Fine tuning arguments (Mosbach et al.)
    output_dir = os.path.join(get_project_root(), 'logs')
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=40,
        learning_rate=1e-5,
        lr_scheduler_type='linear',
        warmup_ratio = 0.1,
        per_device_train_batch_size=len(train_dataset)
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    eval_results = evaluate_model(trainer, eval_dataset)
    
    return eval_results
        
    
def evaluate_model(trainer, eval_dataset):
    """Evaluate fine-tuned model on out of domain dataset."""

    torch.cuda.reset_peak_memory_stats()
    
    prediction_output = trainer.predict(eval_dataset)    # Perform inference
    
    peak_memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)  # bytes to GB
    metrics = prediction_output.metrics
    accuracy = metrics['test_accuracy'] # Accuracy
    total_inference_time = metrics['test_runtime']  # Inference time

    eval_results = {
        'accuracy': accuracy,
        'total_inference_time': total_inference_time,  # Total time for the entire dataset
        'average_inference_time_per_sample': total_inference_time / len(eval_dataset),  # Average time per sample
        'peak_memory_usage_gb': peak_memory_usage,  # Peak memory usage in GB
    }
    
    #TODO: add training loss, eval loss
    
    return eval_results
    
    
def batch_fine_tune(model_name, train_dataset, eval_dataset, sample_sizes, num_trials, save_trials=False):
    """Function to perform few-shot fine-tuning with certain sized samples of a certain number of trials"""
    
    train_datasets = get_random_subsets(train_dataset, sample_sizes, num_trials)
    
    results = {size: [] for size in sample_sizes}
    avg_results = {}
    
    #TODO: add progress bar and suppress Trainer progress bars/logging
    # Iterate over few-shot trials
    for sample_size, trials in train_datasets.items():
        for trial_num, dataset in enumerate(trials):
            model, tokenizer = get_model(model_name)    # Load original model from disk            
            eval_results = fine_tune(model=model, tokenizer=tokenizer, train_dataset=dataset, eval_dataset=eval_dataset) # Fine-tune
            
            # Save trials to disk
            if save_trials:
                trial_label = f"{model_name}/{sample_size}-shot/{trial_num}"
                save_model(model, trial_label)
                
            results[sample_size].append(eval_results)   # Log results
            
        avg_results[sample_size] = {
            'accuracy': np.mean(result['accuracy'] for result in results[sample_size]),
            'total_inference_time': np.mean(result['total_inference_time'] for result in results[sample_size]),
            'average_inference_time_per_sample': np.mean(result['average_inference_time_per_sample'] for result in results[sample_size]),
            'peak_memory_usage_gb': np.mean(result['peak_memory_usage_gb'] for result in results[sample_size]),
        }
        
    return results, avg_results