"""
Fewshot fine-tuning method from “Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation”, Mosbach et al.
https://aclanthology.org/2023.findings-acl.779.pdf
https://huggingface.co/docs/transformers/training

Few-Shot Fine-tuning (FT):
- Few-shot: randomly sampled n in {2, 16, 32, 64, 128} examples.
- Minimal pattern: append question mark to each example.
- Verbalizer: "Yes" and "No" labels for NLI and QQP tasks.
- Fine-tuning: 40 epochs, learning rate of 1e-5, linear increase for initial 10% of steps, then constant.
"""


