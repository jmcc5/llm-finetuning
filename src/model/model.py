"""
Methods for model download and use.
https://huggingface.co/facebook/opt-125m
https://huggingface.co/facebook/opt-350m
"""

# Import Libraries
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

# Import modules
from src.utils import get_project_root


def download_model(model_name, model_type):
    """Load specified huggingface model and save to disk. model_name should be 'opt-125m' and 'opt-250m'. 
    model_type should be 'SequenceClassification' or 'CausalLM'."""
    if model_type == 'SequenceClassification':
        model_class = AutoModelForSequenceClassification
    elif model_type == 'CausalLM':
        model_class = AutoModelForCausalLM
    else:
        raise ValueError(f"Model Type {model_type} not supported. Try 'SequenceClassification' or 'CausalLM'.")
    
    if model_name == "opt-125m" or model_name == "opt-350m":
        tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
        model = model_class.from_pretrained(f"facebook/{model_name}")
    else:
        raise ValueError(f"Model {model_name} not supported. Try 'opt-125m' or 'opt-350m'.")
    
    filepath = os.path.join(get_project_root(), 'models', 'pretrained', model_name) #TODO: modify get_project_root to handle buiding the path
    filepath_tokenizer = os.path.join(filepath, 'tokenizer')
    filepath_model = os.path.join(filepath, 'model', model_type)
    
    tokenizer.save_pretrained(filepath_tokenizer)
    model.save_pretrained(filepath_model)
        
def get_model(model_name, model_type, pretrained=True):
    """Returns baseline pre-trained tokenizer and model. model_name should be 'opt-125m' and 'opt-250m'. 
    model_type should be 'SequenceClassification' or 'CausalLM'. If pretrained is False, will load from /models/finetuned."""
    if model_type == 'SequenceClassification':
        model_class = AutoModelForSequenceClassification
    elif model_type == 'CausalLM':
        model_class = AutoModelForCausalLM
    else:
        raise ValueError(f"Model Type {model_type} not supported. Try 'SequenceClassification' or 'CausalLM'.")
    
    if pretrained:
        filepath = os.path.join(get_project_root(), 'models', 'pretrained', model_name)
        filepath_tokenizer = os.path.join(filepath, 'tokenizer')
        filepath_model = os.path.join(filepath, 'model', model_type)
        if not os.path.exists(filepath_model):
            download_model(model_name, model_type)
    else:
        filepath_tokenizer = os.path.join(get_project_root(), 'models', 'pretrained', model_name, 'tokenizer')
        filepath_model = os.path.join(get_project_root(), 'models', 'finetuned', model_name)
        
    tokenizer = AutoTokenizer.from_pretrained(filepath_tokenizer)
    model = model_class.from_pretrained(filepath_model)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, tokenizer

def save_model(model, model_name):
    """Saves model to /models/finetuned. model_name should be descriptive of the fine-tuned model ('opt-125m_2')"""
    filepath = os.path.join(get_project_root(), 'models', 'finetuned', model_name)
    model.save_pretrained(filepath)
    # tokenizer.save_pretrained(filepath)