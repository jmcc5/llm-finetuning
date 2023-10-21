# Efficient_LLM_Few-Example_Fine-Tuning

## Outline

- `data`: datasets and import functions for glue datasets
- `experiments`: notebooks for running experiments
- `fine_tuners`: fine tuning classes
- `models`: wrapper classes for opt models
- `utils.py`: utility functions for the entire project

## Environment
Create a conda environment from the .yml file.
```
conda create --name fine-tuning --file requirements.txt
conda activate fine-tuning
```
Update it if any packageds are added:
```
conda list --export > requirements.txt
```