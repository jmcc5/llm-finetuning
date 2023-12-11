"""
Context distillation fine-tuning.
"""

# Import Libraries
from transformers import get_scheduler
from tqdm.autonotebook import tqdm
import torch
from torch.utils.data import DataLoader
import time
from datasets import Dataset

# Import Modules
from src.finetuners.utils import get_yes_no_constraint, get_teacher_context, apply_minimal_pattern, tokenize_dataset, metrics_to_csv, interpret_generated_texts, distillation_loss, compute_metrics_causal, reset_memory_stats, get_peak_memory
from src.model.model import get_model


def batch_context_distillation(model_names, in_domain_dataset, train_datasets, eval_dataset_in, eval_dataset_out, batch_size=4, exp_label=None):
    """Function to perform context distillation fine-tuning for each model in model_names."""
    metrics = []
    
    # Loop over models, sample sizes
    for model_name in model_names:
        torch.cuda.empty_cache()
        for sample_size, train_dataset in train_datasets.items():
            # Dynamic batch sizing
            if model_name == 'opt-350m':
                batch_size = int(2)
            else:
                batch_size = int(4)
                
            # Load student and teacher models
            student_model, tokenizer = get_model(model_name, 'CausalLM')
            teacher_model, _ = get_model(model_name, 'CausalLM')
            
            # Don't track gradients in teacher model
            for param in teacher_model.parameters():
                param.requires_grad = False
            
            metrics_trial = context_distillation(student_model=student_model,
                                                 teacher_model=teacher_model,
                                                 tokenizer=tokenizer,
                                                 dataset=in_domain_dataset,
                                                 train_dataset=train_dataset,
                                                 eval_dataset_in=eval_dataset_in,
                                                 eval_dataset_out=eval_dataset_out,
                                                 num_epochs=1,
                                                 model_name=model_name,
                                                 batch_size=batch_size)
            
            # Log metrics
            metrics_trial = {'model_name': model_name,
                            'sample_size': sample_size,
                            **metrics_trial}
            metrics.append(metrics_trial)
        
    metrics_to_csv(metrics=metrics, finetuning_method='context_distillation', exp_label=exp_label)

def context_distillation(student_model, teacher_model, tokenizer, dataset, train_dataset, num_epochs, eval_dataset_in, eval_dataset_out, model_name, batch_size=8):
    device = student_model.device
    
    dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)    # data_collator could solve issues
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
    num_training_steps = num_epochs * len(dataloader)

    lr_schedulizer = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    sample_size = len(train_dataset)
    progress_bar = tqdm(range(num_training_steps+1), desc=f"{model_name} {sample_size}-shot")

    for epoch in range(num_epochs):
        for batch in dataloader:
            # Get teacher logits
            teacher_context = get_teacher_context(dataset)
            teacher_batch = Dataset.from_dict(batch)
            teacher_dataset = apply_minimal_pattern(teacher_batch, teacher_context)
            teacher_dataset = tokenize_dataset(teacher_dataset, tokenizer)
            teacher_input_ids = torch.tensor(teacher_dataset['input_ids'], device=device)
            teacher_mask = torch.tensor(teacher_dataset['attention_mask'], device=device)
            teacher_logits = teacher_model(teacher_input_ids, teacher_mask).logits
            
            # Get student logits
            student_batch = Dataset.from_dict(batch)
            student_batch = apply_minimal_pattern(student_batch, "")
            student_dataset = tokenize_dataset(student_batch, tokenizer)
            student_input_ids = torch.tensor(student_dataset['input_ids'], device=device)
            student_mask = torch.tensor(student_dataset['attention_mask'], device=device)
            student_logits = student_model(student_input_ids, student_mask).logits
            
            # Backwards pass and optimizer step
            loss = distillation_loss(teacher_logits, student_logits)    # KL divergence loss
            loss.backward()
            optimizer.step()
            lr_schedulizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    progress_bar.set_postfix_str("Evaluating...")
    metrics = evaluate(student_model, tokenizer, eval_dataset_in, eval_dataset_out, batch_size=8, verbose=False, disable_tqdm=True)
    progress_bar.update(1)
    progress_bar.set_postfix(metrics)   # Update progress bar postfix
    
    return metrics


def evaluate(model, tokenizer, eval_dataset_in, eval_dataset_out, batch_size=8, verbose=False, disable_tqdm=False): # no context needed for eval
    """Context Distillation student model learning base method."""
    def evaluate_dataset(model, tokenizer, dataset, batch_size):
        reset_memory_stats()
        start_time = time.time()
        predicted_labels = []
        yes_no_constraint = get_yes_no_constraint(tokenizer)
        
        progress_bar = tqdm(range(0, len(dataset), batch_size), disable=disable_tqdm)

        for i in progress_bar:
            # Verbalize and tokenize batch
            batch_indices = range(i, min(i + batch_size, len(dataset)))
            batch = dataset.select(batch_indices)
            batch = apply_minimal_pattern(batch, context="")
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
            "samples_per_second": samples_per_second,
            "peak_memory_gb": get_peak_memory()
        }
        return metrics

    # Evaluate - batch size = 8 due to GPU memory constraints
    combined_metrics = {}
    eval_metrics_in = evaluate_dataset(model, tokenizer, eval_dataset_in, batch_size=batch_size)    # In domain
    if verbose:
        print(f"In domain eval metrics:\n{eval_metrics_in}")
    eval_metrics_out = evaluate_dataset(model, tokenizer, eval_dataset_out, batch_size=batch_size)  # OOD
    if verbose:
        print(f"Out of domain eval metrics:\n{eval_metrics_out}")

    combined_metrics.update({f'eval_in_{k}': v for k, v in eval_metrics_in.items()})
    combined_metrics.update({f'eval_out_{k}': v for k, v in eval_metrics_out.items()})
    
    return combined_metrics
