# llm-finetuning

## Outline

- `data`: datasets from huggingface
- [`experiments`](https://github.com/jmcc5/llm-finetuning/tree/main/experiments): notebooks for running experiments
- `logs`: csv output from fine-tuning
- `models`: saved pre-trained and fine-tuned models
- [`src/`](https://github.com/jmcc5/llm-finetuning/tree/main/src): main project directory
  - [`data`](https://github.com/jmcc5/llm-finetuning/tree/main/src/data): functions for fetching and loading datasets
  - [`finetuners`](https://github.com/jmcc5/llm-finetuning/tree/main/src/finetuners): fine-tuning methods
  - [`models`](https://github.com/jmcc5/llm-finetuning/tree/main/src/model): functions for fetching and loading models
  - [`visualization`](https://github.com/jmcc5/llm-finetuning/tree/main/src/visualization): functions for graphing fine-tuning output

## Summary
This Python project aims to explore LLM fine-tuning and context based methods in an accessible format. We build on work in [uds-lsv/llmft](https://github.com/uds-lsv/llmft) to implement few-shot fine-tuning and in-context learning (ICL) and create our own novel version of context distillation fine-tuning, originally proposed by [Anthropic](https://arxiv.org/pdf/2112.00861.pdf) in 2021.

We rely heavily on huggingface's [transformers](https://github.com/huggingface/transformers). For ease of compute and iteration, we experiment with smaller models: [OPT-125m](https://huggingface.co/facebook/opt-125m) and [OPT-350m](https://huggingface.co/facebook/opt-350m). We use the MNLI dataset from [GLUE](https://huggingface.co/datasets/glue) as in-domain and [HANS](https://huggingface.co/datasets/hans) as out-of-domain.

## Results
TODO

## Setup
Clone the repository.
```
git clone https://github.com/jmcc5/llm-finetuning.git
```
Create a conda environment from the .yml file.
```
conda env create -f environment.yml
conda activate fine-tuning
```
Update it if any packages are added.
```
conda env export --no-builds > environment.yml
```
To enable module importing, run:
```
pip install -e .
```
If you run into issues related to your torch-cuda version, please reinstall the recommended version for your system from https://pytorch.org/get-started/locally/.
