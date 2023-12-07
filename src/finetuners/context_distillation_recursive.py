"""
Context distillation fine-tuning
"""

import os
from transformers import TrainingArguments, Trainer, PrinterCallback, get_scheduler, AdamW
from tqdm.autonotebook import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from datasets import Dataset




# Import Modules
from src.finetuners.utils import get_yes_no_constraint, get_teacher_context, apply_minimal_pattern, tokenize_dataset, compute_metrics, metrics_to_csv, interpret_generated_texts, MemoryUsageCallback, ReformatEvalMetricsCallback, compute_metrics_causal 
from src.model.model import save_model, get_model
from src.utils import get_project_root

def distillation_loss(teacher_logits, student_logits, temp = 1):
    kldivloss_func = torch.nn.KLDivLoss(reduction='batchmean')
    loss = temp ** 2 * kldivloss_func(
                F.log_softmax(student_logits / temp, dim=-1),
                F.softmax(teacher_logits / temp, dim=-1))
    
    return loss


def recursive_context_distillation(student_model, tokenizer, dataset,num_epochs,eval_dataset_in, eval_dataset_out, batch_size=8, model_name='opt-125m'):
    #datasets should come in pre tokenized with context in teacher datatset?
    device = student_model.device
    # may need to use collate_fn = data_collator with data_collator = transformers.DataCollatorWithPadding(tokenizer)

    student_data_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    #to do: test dataloader

    optimizer = optimizer = AdamW(student_model.parameters(), lr=5e-5) # lr maybe changed follows paper

    num_training_steps = num_epochs * len(student_data_loader)

    lr_schedulizer = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        for student_batch in student_data_loader:
            #get teacher logits
            teacher_context = get_teacher_context(dataset)
            teacher_batch = Dataset.from_dict(student_batch)
            teacher_dataset = apply_minimal_pattern(teacher_batch, teacher_context)
            teacher_dataset = tokenize_dataset(teacher_dataset, tokenizer)
            teacher_input_ids = torch.tensor(teacher_dataset['input_ids'])
            teacher_mask = torch.tensor(teacher_dataset['attention_mask'])
            teacher_logits = student_model(teacher_input_ids.to(device), teacher_mask.to(device)).logits
            #get student logits
            student_batch = Dataset.from_dict(student_batch)
            student_batch = apply_minimal_pattern(student_batch, "")
            student_dataset = tokenize_dataset(student_batch, tokenizer)
            student_input_ids = torch.tensor(student_dataset['input_ids'])
            student_mask = torch.tensor(student_dataset['attention_mask'])
            student_logits = student_model(student_input_ids.to(device), student_mask.to(device)).logits
            
            loss = distillation_loss(teacher_logits, student_logits)
            loss.backward()

            optimizer.step()
            lr_schedulizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        #todo eval on student model
    return evaluate(student_model, tokenizer, eval_dataset_in, eval_dataset_out, model_name=model_name)
# Evalute post training

def evaluate(model, tokenizer, eval_dataset_in, eval_dataset_out, batch_size=8, verbose=True, disable_tqdm=None, model_name='opt-125m'): # no context needed for eval

    """Context Distillation student model learning base method."""
    def evaluate_dataset(model, tokenizer, dataset, batch_size):
        torch.cuda.reset_peak_memory_stats(device=model.device)
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
            "peak memory": torch.cuda.max_memory_allocated(device=model.device) / (1024 ** 3)
        }
        return metrics

    # Evaluate - batch size = 8 due to GPU memory constraints
    eval_metrics_in = evaluate_dataset(model, tokenizer, eval_dataset_in, batch_size=batch_size)    # In domain
    if verbose:
        print(f"In domain eval metrics:\n{eval_metrics_in}")
    eval_metrics_out = evaluate_dataset(model, tokenizer, eval_dataset_out, batch_size=batch_size)  # OOD
    if verbose:
        print(f"Out of domain eval metrics:\n{eval_metrics_out}")
    combined_metrics = {"model_name": model_name}
    combined_metrics.update({f'eval_in_{k}': v for k, v in eval_metrics_in.items()})
    combined_metrics.update({f'eval_out_{k}': v for k, v in eval_metrics_out.items()})
    metrics_to_csv(metrics=[combined_metrics], finetuning_method='recursive_context_distillation')
    return combined_metrics

