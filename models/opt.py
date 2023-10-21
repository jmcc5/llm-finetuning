"""
Wrapper classes for facebook opt model inference and fine-tuning (https://huggingface.co/facebook/opt-125m)
"""

# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("text-generation", model="facebook/opt-125m")

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
# model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")