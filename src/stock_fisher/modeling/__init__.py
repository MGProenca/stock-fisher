"""Modelling: train outcome predictors and track experiments.

Mirrors the ingestion/feature split: this layer consumes the *assembled dataset*
and produces trained-model evaluations. It owns no feature math and no network
logic.

  - `data.py`       — load the dataset and build the pooled / temporal splits.
  - `models.py`     — the estimators we compare (baselines + logistic + xgboost).
  - `evaluation.py` — 3-class metrics + confusion matrix.
  - `train.py`      — fit/evaluate across models × splits, tracked in MLflow.
"""

from .data import default_splits, feature_columns, load_dataset, pooled_split, temporal_split
from .evaluation import confusion_frame, evaluate
from .models import ModelSpec, default_model_specs
from .train import fit_evaluate, run_experiments, run_one

__all__ = [
    "load_dataset",
    "feature_columns",
    "default_splits",
    "pooled_split",
    "temporal_split",
    "evaluate",
    "confusion_frame",
    "ModelSpec",
    "default_model_specs",
    "fit_evaluate",
    "run_experiments",
    "run_one",
]
