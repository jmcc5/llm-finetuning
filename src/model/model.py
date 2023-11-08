"""
Methods for model download and use.
https://huggingface.co/facebook/opt-125m
https://huggingface.co/facebook/opt-350m
"""

# Import Libraries
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import modules
from src.utils import get_project_root


def download_model(model_name):
    """Load specified huggingface model and save to disk. opt-125m and opt-250m supported."""
    if model_name == "opt-125m" or model_name == "opt-350m":
        tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(f"facebook/{model_name}")
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    filepath = os.path.join(get_project_root(), 'models', 'pretrained', model_name)
    filepath_tokenizer = os.path.join(filepath, 'tokenizer')
    filepath_model = os.path.join(filepath, 'model')
    
    if not os.path.exists(filepath):
        tokenizer.save_pretrained(filepath_tokenizer)
        model.save_pretrained(filepath_model)
        
def get_model(model_name, pretrained=True):
    """Returns baseline pre-trained tokenizer and model. Only opt-125m and opt-350m supported. If pretrained is False, will load from /models/finetuned."""
    if pretrained:
        filepath = os.path.join(get_project_root(), 'models', 'pretrained', model_name)
        filepath_tokenizer = os.path.join(filepath, 'tokenizer')
        filepath_model = os.path.join(filepath, 'model')
    else:
        # filepath = os.path.join(get_project_root(), 'models', 'finetuned', model_name)
        filepath_tokenizer = os.path.join(get_project_root(), 'models', 'pretrained', model_name, 'tokenizer')
        filepath_model = os.path.join(get_project_root(), 'models', 'finetuned', model_name)
        
    if not os.path.exists(filepath):
        download_model(model_name)
        
    tokenizer = AutoTokenizer.from_pretrained(filepath_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(filepath_model)
    return tokenizer, model

def save_model(model, model_name):
    """Saves model to /models/finetuned. model_name should be descriptive of the fine-tuned model ('opt-125m_2')"""
    filepath = os.path.join(get_project_root(), 'models', 'finetuned', model_name)
    model.save_pretrained(filepath)
    # tokenizer.save_pretrained(filepath)