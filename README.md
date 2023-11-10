# Efficient_LLM_Few-Example_Fine-Tuning

## Tasks
- [ ] Few-shot fine-tuning (Harrison)
- [ ] In context learning (Ethan)
- [ ] Context distillation fine-tuning (Joel)


## Outline

- `data`: datasets from huggingface
- `experiments`: notebooks for running experiments
- `logs`: csv output from fine-tuning
- `models`: saved pre-trained and fine-tuned models
- `src/`: main project directory
  - `data`: import functions for datasets
  - `evaluation`: functions for evaluating fine-tuned models
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
To enable module importing with the repo, run:
```
pip install -e .
```
