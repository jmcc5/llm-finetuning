"""
In Context Learning fine-tuning method from “Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation”, Mosbach et al.
https://aclanthology.org/2023.findings-acl.779.pdf
https://huggingface.co/docs/transformers/training

In-Context Learning (ICL):
- Few-shot: 16 demonstrations mainly, additional experiments with 2 and 32 demonstrations.
- Verbalizer: same as for fine-tuning.
- Prediction correctness: higher probability for correct label's verbalizer token compared to other.
"""


