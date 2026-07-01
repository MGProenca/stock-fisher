"""Metrics for the 3-class outcome problem.

Reports both a hard-label view (accuracy, balanced accuracy, macro-F1, per-class
precision/recall) and a probabilistic view (log loss). Draws are the rare class
(~8% in blitz), so balanced accuracy and macro-F1 matter as much as raw accuracy
— a model that never predicts draws can still look fine on accuracy alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)


def evaluate(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: pd.DataFrame | None,
    labels: list[str],
) -> dict:
    """Compute the metric bundle for one set of predictions.

    `y_proba`, if given, must be a DataFrame whose columns are exactly `labels`
    (probability per class). Probabilities are clipped before log loss so a
    degenerate baseline (e.g. majority class with 0/1 probabilities) yields a
    large-but-finite loss instead of infinity.
    """
    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        # macro = unweighted mean over classes (rare draw counts 1/3);
        # weighted = mean weighted by class support (reflects prevalence).
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels)
        ),
    }

    if y_proba is not None:
        # sklearn's log_loss assumes the probability columns are in lexicographic
        # label order, so align both the columns and the `labels` arg to that
        # order (passing our display order misaligns them and inflates the loss).
        sorted_labels = sorted(labels)
        proba = y_proba[sorted_labels].to_numpy()
        proba = np.clip(proba, 1e-15, 1.0)
        proba = proba / proba.sum(axis=1, keepdims=True)
        metrics["log_loss"] = float(log_loss(y_true, proba, labels=sorted_labels))

    # Per-class precision/recall/f1 flattened into the metrics dict.
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    for label in labels:
        metrics[f"precision_{label}"] = float(report[label]["precision"])
        metrics[f"recall_{label}"] = float(report[label]["recall"])
        metrics[f"f1_{label}"] = float(report[label]["f1-score"])

    return metrics


def confusion_frame(
    y_true: pd.Series, y_pred: np.ndarray, labels: list[str]
) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame (rows=true, cols=pred)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
