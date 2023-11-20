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
from transformers import TrainingArguments, Trainer, PrinterCallback
from tqdm.autonotebook import tqdm

# Import Modules
from src.finetuners.utils import apply_minimal_pattern, tokenize_dataset, compute_metrics, metrics_to_csv, MemoryUsageCallback, ReformatEvalMetricsCallback
from src.model.model import save_model, get_model
from src.utils import get_project_root

def fine_tune(model, tokenizer, train_dataset, eval_dataset_in, eval_dataset_out, verbose=True):
    """In-context learning base method."""
    # Verbalize and Tokenize
    train_dataset = apply_minimal_pattern(train_dataset)  # Apply minimal pattern
    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=512)  # Tokenize
    
    eval_dataset_in = apply_minimal_pattern(eval_dataset_in)
    eval_dataset_in = tokenize_dataset(eval_dataset_in, tokenizer, max_length=512)
    
    eval_dataset_out = apply_minimal_pattern(eval_dataset_out)
    eval_dataset_out = tokenize_dataset(eval_dataset_out, tokenizer, max_length=512)

    # Create in-context learning prompt from training data

def create_few_shot_context(dataset, pattern, description=None, seperator=",", from_indices=None):
    # Hardcoded values for keys:
    task_keys = ["premise", "hypothesis"]

    # select samples to construct context from
    if from_indices is None:
        demos, indices = select_random_subset(dataset)
    else:
        demos, indices = select_subset_by_index(dataset, from_indices)

    # create context
    context = "" if description == None else f"{description}{seperator}"

    for sample in demos:
        formatted_sample = f"{task_keys[0]} {task_keys[1]} ?"
        verbalized_label = dataset
        context += f"{formatted_sample}{verbalized_label}{seperator}"

    return context, indices
