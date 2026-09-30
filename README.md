## MLOps Assignment 3: Fashion-MNIST ANN Pipeline

A fully connected neural network that classifies Fashion-MNIST images into 10 clothing categories,
versioned end to end with Git and DVC.

### Pipeline

| Script | Output |
|---|---|
| `src/prepare.py` | Raw train/test arrays in `data/raw/` |
| `src/preprocess.py` | Normalized train/val/test arrays in `data/processed/` |
| `src/train.py` | `models/model.h5` and `models/history.csv` |
| `src/evaluate.py` | `metrics.json` and the confusion matrix plot |

Hyperparameters live in `params.yaml`.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
