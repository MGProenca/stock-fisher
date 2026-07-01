"""Model factory: the estimators we compare, each with its feature subset.

Kept small and well-reasoned (the exercise explicitly prefers this over a
leaderboard-topping black box):

  - `majority`  — always predict the most frequent class. The naive accuracy bar.
  - `prior`     — predict the class base rates as probabilities. The naive
                  *log-loss* bar (a probabilistic floor).
  - `elo`       — logistic regression on the single `elo_expected_white` feature.
                  Turns the Elo expectation into calibrated 3-class probabilities.
                  This is the "bar to beat".
  - `logistic`  — multinomial logistic regression on all features, standardized.
  - `xgboost`   — gradient-boosted trees. Captures non-linearities and feature
                  interactions a linear model can't, with no scaling needed.
  - `logistic_balanced` — same logistic model with class_weight="balanced", to
                  show the draw-recall vs accuracy trade-off on the rare class.

A `ModelSpec` couples an estimator with the columns it consumes, so adding a
model is a one-entry change and the training loop stays untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .data import ELO_FEATURE


class StringLabelClassifier(BaseEstimator, ClassifierMixin):
    """Adapt a classifier that requires integer labels (e.g. XGBoost) to the
    string-label interface the rest of the pipeline uses.

    Encodes `y` on fit and decodes on predict, exposing `classes_` as the original
    string labels. `LabelEncoder.classes_` is sorted, and XGBoost's `predict_proba`
    columns follow the same 0..K-1 class order, so the probability columns stay
    aligned with `classes_` (which `train._aligned_proba` relies on).
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, X, y):
        self._encoder = LabelEncoder().fit(y)
        self.classes_ = self._encoder.classes_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, self._encoder.transform(y))
        return self

    def predict(self, X):
        return self._encoder.inverse_transform(self.estimator_.predict(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


@dataclass
class ModelSpec:
    """An estimator plus the feature columns it is trained on."""

    name: str
    description: str
    build: Callable[[], BaseEstimator]
    features: list[str]
    params: dict = field(default_factory=dict)


def _logistic(class_weight=None) -> BaseEstimator:
    # Standardize so the L2 penalty treats features comparably; lbfgs handles
    # multinomial logistic regression natively. class_weight="balanced" up-weights
    # the rare draw class to trade accuracy for draw recall.
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight=class_weight)),
        ]
    )


def _elo_baseline() -> BaseEstimator:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


# Modest-depth boosting: enough capacity to model interactions, regularized
# (shallow trees, subsampling, low learning rate) to resist overfitting on the
# handful of features. Deliberately not tuned (see the README writeup).
_XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    objective="multi:softprob",
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=42,
)


def _xgboost() -> BaseEstimator:
    # Imported here so the wrapper/spec stay importable even if xgboost is absent.
    from xgboost import XGBClassifier

    return StringLabelClassifier(XGBClassifier(**_XGB_PARAMS))


def default_model_specs(feature_columns: list[str]) -> list[ModelSpec]:
    """The models compared by default, cheapest/most-naive first.

    `feature_columns` is the model-input set read from the dataset (see
    `data.feature_columns`), so the big models automatically use whatever feature
    tiers the dataset contains. The Elo baseline always uses just its one column.
    """
    return [
        ModelSpec(
            name="majority",
            description="Always predict the most frequent class.",
            build=lambda: DummyClassifier(strategy="most_frequent"),
            features=feature_columns,
        ),
        ModelSpec(
            name="prior",
            description="Predict class base rates (probabilistic floor).",
            build=lambda: DummyClassifier(strategy="prior"),
            features=feature_columns,
        ),
        ModelSpec(
            name="elo",
            description="Tier 0: logistic on elo_expected_white only.",
            build=_elo_baseline,
            features=[ELO_FEATURE],
            params={"C": 1.0},
        ),
        ModelSpec(
            name="logistic",
            description="Multinomial logistic on all dataset features.",
            build=_logistic,
            features=feature_columns,
            params={"C": 1.0, "max_iter": 1000},
        ),
        ModelSpec(
            name="xgboost",
            description="Gradient-boosted trees (XGBoost) on all dataset features.",
            build=_xgboost,
            features=feature_columns,
            params={
                k: _XGB_PARAMS[k]
                for k in ("n_estimators", "max_depth", "learning_rate", "subsample")
            },
        ),
        ModelSpec(
            name="logistic_balanced",
            description="Logistic with class_weight=balanced (draw recall vs accuracy).",
            build=lambda: _logistic(class_weight="balanced"),
            features=feature_columns,
            params={"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
        ),
    ]
