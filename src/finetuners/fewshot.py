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
from transformers import TrainingArguments, Trainer

# Import Modules
from src.finetuners.utils import apply_minimal_pattern, tokenize_dataset, compute_metrics
from src.utils import get_project_root


def fine_tune(model, tokenizer, train_dataset):
    """Few shot finetuning base method. Modifies model and tokenizer passed in."""
    # Verbalize and tokenize    
    train_dataset = apply_minimal_pattern(train_dataset)  # Apply minimal pattern
    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=512)  # Tokenize
    # val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Debug
    # print("Sample data after tokenization:", train_dataset[0])
    # input_ids = train_dataset[0]['input_ids']
    # labels = train_dataset[0]['labels']
    # print("Input IDs length:", len(input_ids))
    # # print("Labels length:", len(labels))
    # # Check for correct alignment in a batch
    batch_size = 2  # Change this to your actual batch size
    # batch = train_dataset.select(range(batch_size))  # Using 'select' to create a smaller dataset for debugging
    # input_ids_batch = batch['input_ids']
    # labels_batch = batch['labels']
    # print("Batch input IDs length:", len(input_ids_batch))
    # print("Batch labels length:", len(labels_batch))
    # if len(input_ids_batch) != len(labels_batch):
    #     raise ValueError("Mismatched input and label batch sizes.")

    # Fine tuning arguments (Mosbach et al.)
    output_dir = os.path.join(get_project_root(), 'logs')
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=40,
        learning_rate=1e-5,
        evaluation_strategy="epoch",
        lr_scheduler_type='linear',
        warmup_ratio = 0.1,
        per_device_train_batch_size=batch_size
        # dropout=0.1,
        # batch_size=32,
        # total_steps=
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        # compute_metrics=compute_metrics,
    )

    trainer.train()
    
    #TODO: save model?
    
    
def batch_fine_tune(sample_size=[2, 16, 32, 64, 128], num_batches=10):
    """Function to perform few-shot fine-tuning with certain sized samples of a certain number of batches"""
    pass