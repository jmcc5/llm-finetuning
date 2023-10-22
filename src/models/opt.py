"""
Wrapper classes for facebook opt model inference and fine-tuning 
https://huggingface.co/facebook/opt-125m
https://huggingface.co/facebook/opt-350m
"""

# Import Libraries
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import modules
from src.utils import get_project_root

# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("text-generation", model="facebook/opt-125m")

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
# model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

def download_model(model_name):
    """Load specified huggingface model and save to disk. opt-125m and opt-250m supported."""
    if model_name == "opt-125m" or model_name == "opt-350m":
        tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
        model = AutoModelForCausalLM.from_pretrained(f"facebook/{model_name}")
    else:
        raise ValueError(f"Model {model_name} not supported.")
    filepath = os.path.join(get_project_root(), 'models', model_name)
    filepath_tokenizer = os.path.join(filepath, 'tokenizer')
    filepath_model = os.path.join(filepath, 'model')
    if not os.path.exists(filepath):
        tokenizer.save_pretrained(filepath_tokenizer)
        model.save_pretrained(filepath_model)
        
def get_opt125():
    """Returns baseline opt-125m tokenizer and model"""
    filepath = os.path.join(get_project_root(), 'models', 'opt125')
    if not os.path.exists(filepath):
        download_model('opt-125m')
    filepath_tokenizer = os.path.join(filepath, 'tokenizer')
    filepath_model = os.path.join(filepath, 'model')
    tokenizer = AutoTokenizer.from_pretrained(filepath_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(filepath_model)
    return tokenizer, model

def get_opt350():
    """Returns baseline opt-350m tokenizer and model"""
    filepath = os.path.join(get_project_root(), 'models', 'opt350')
    if not os.path.exists(filepath):
        download_model('opt-350m')
    filepath_tokenizer = os.path.join(filepath, 'tokenizer')
    filepath_model = os.path.join(filepath, 'model')
    tokenizer = AutoTokenizer.from_pretrained(filepath_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(filepath_model)
    return tokenizer, model