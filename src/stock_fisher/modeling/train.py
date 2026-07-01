"""Training orchestration: fit every model on every split, evaluate, track.

This is the modelling counterpart to `dataset.build_dataset` — it owns the
control flow (which models × which splits) but delegates the actual estimators to
`models.py` and metrics to `evaluation.py`. Experiment tracking is handled by
**MLflow**: each (model, split) pair is one MLflow run that logs its params,
metrics, the fitted model, and the confusion-matrix / predictions artifacts.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from .data import LABELS, Split
from .evaluation import confusion_frame, evaluate
from .models import ModelSpec

logger = logging.getLogger(__name__)


def _aligned_proba(estimator, X: pd.DataFrame) -> pd.DataFrame | None:
    """predict_proba as a DataFrame with columns in the canonical LABELS order.

    Some classes may be unobserved in a tiny training set; those columns are
    filled with 0 so the matrix always has every label.
    """
    if not hasattr(estimator, "predict_proba"):
        return None
    proba = estimator.predict_proba(X)
    frame = pd.DataFrame(proba, columns=list(estimator.classes_), index=X.index)
    return frame.reindex(columns=LABELS, fill_value=0.0)


def _log_dataframe_artifact(frame: pd.DataFrame, filename: str, index: bool) -> None:
    """Persist a DataFrame as a CSV artifact on the active MLflow run."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        frame.to_csv(path, index=index)
        mlflow.log_artifact(str(path))


def fit_evaluate(spec: ModelSpec, split: Split) -> dict:
    """Train one model on one split and evaluate it — pure, no experiment tracking.

    Returns the fitted estimator, the metric bundle, the confusion matrix, the
    hard predictions, and the aligned class probabilities. `run_one` wraps this
    with MLflow logging; the analysis notebook calls it directly.
    """
    X_train = split.X_train[spec.features]
    X_test = split.X_test[spec.features]

    estimator = spec.build()
    estimator.fit(X_train, split.y_train)

    y_pred = estimator.predict(X_test)
    y_proba = _aligned_proba(estimator, X_test)

    return {
        "estimator": estimator,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": evaluate(split.y_test, y_pred, y_proba, labels=LABELS),
        "confusion": confusion_frame(split.y_test, y_pred, labels=LABELS),
    }


def run_one(spec: ModelSpec, split: Split, dataset_meta: dict) -> dict:
    """Train+evaluate one model on one split as an MLflow run. Returns a summary row."""
    with mlflow.start_run(run_name=f"{spec.name}__{split.name}") as run:
        fitted = fit_evaluate(spec, split)
        estimator = fitted["estimator"]
        y_pred = fitted["y_pred"]
        y_proba = fitted["y_proba"]
        metrics = fitted["metrics"]
        cm = fitted["confusion"]

        # --- log to MLflow -------------------------------------------------
        mlflow.set_tags({"model": spec.name, "split": split.name})
        mlflow.log_params(
            {
                "model": spec.name,
                "features": spec.features,
                "split": split.name,
                **{f"hp_{k}": v for k, v in spec.params.items()},
                **dataset_meta,
                **split.describe(),
            }
        )
        mlflow.log_metrics(metrics)

        _log_dataframe_artifact(cm, "confusion_matrix.csv", index=True)

        preds = pd.DataFrame(
            {"y_true": split.y_test.to_numpy(), "y_pred": y_pred},
            index=split.X_test.index,
        )
        if y_proba is not None:
            for label in LABELS:
                preds[f"proba_{label}"] = y_proba[label].to_numpy()
        _log_dataframe_artifact(preds, "predictions.csv", index=True)

        # cloudpickle (not the skops default) so non-sklearn estimators like the
        # XGBoost wrapper serialize without trusted-type restrictions.
        mlflow.sklearn.log_model(
            estimator,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        run_id = run.info.run_id

    logger.info(
        "%s on %s: acc=%.3f bal_acc=%.3f macro_f1=%.3f log_loss=%s",
        spec.name,
        split.name,
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["macro_f1"],
        f"{metrics['log_loss']:.3f}" if "log_loss" in metrics else "n/a",
    )

    return {
        "run_id": run_id,
        "model": spec.name,
        "split": split.name,
        **{
            k: metrics[k]
            for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
        },
        "recall_draw": metrics.get("recall_draw", np.nan),
        "log_loss": metrics.get("log_loss", np.nan),
    }


def run_experiments(
    specs: list[ModelSpec],
    splits: list[Split],
    dataset_meta: dict,
    experiment_name: str,
) -> pd.DataFrame:
    """Run every (model, split) pair under one MLflow experiment.

    Returns a comparison table of headline metrics. The MLflow tracking URI is
    expected to be configured by the caller (see `cli.py`).
    """
    mlflow.set_experiment(experiment_name)

    summaries = []
    for split in splits:
        for spec in specs:
            summaries.append(run_one(spec, split, dataset_meta))
    return pd.DataFrame(summaries)
