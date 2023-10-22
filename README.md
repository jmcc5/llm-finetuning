# Efficient_LLM_Few-Example_Fine-Tuning

## Tasks
- Baseline fine-tuning (Harrison)
- Context distillation fine-tuning (Joel)
- Evaluation (Ethan)

## Outline

- `data`: datasets and import functions for glue datasets
- `experiments`: notebooks for running experiments
- `fine_tuners`: fine tuning classes
- `models`: wrapper classes for opt models
- `utils.py`: utility functions for the entire project

## Environment
Create a conda environment from the .yml file.
```
conda env create -f environment.yml
conda activate fine-tuning
```
Update it if any packages are added:
```
conda env export --no-builds > environment.yml
```
