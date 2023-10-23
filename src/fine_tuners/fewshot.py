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
from transformers import TrainingArguments, Trainer

# Import Modules



def fine_tune(model, tokenizer, train_dataset, val_dataset):
    #TODO: Process data - append question mark to each example - utils

    #TODO: Verbalizer? Same as tokenizer?
    
    #TODO: Randomly select {2, 16, 32, 64, 128} samples from dataset

    # Fine tuning arguments
    training_args = TrainingArguments(
        num_train_epochs=40,
        learning_rate=1e-5,
        evaluation_strategy="epoch",
        
    )
    
    #TODO: write compute_metrics function for validation during training
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()