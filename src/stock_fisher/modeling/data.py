"""Loading the assembled dataset and turning it into model-ready X / y splits.

This sits on top of the dataset built by `stock_fisher.dataset`. It does **not**
create features (that's the feature layer's job) — it only:
  - loads the parquet/csv the miner produced,
  - exposes the two split strategies: a pooled stratified split and a temporal
    holdout (train on earlier events, test on the latest).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_COLUMN = "label"

# Identifier / bookkeeping columns that are NOT model inputs. Everything else in
# the dataset is a feature (note `round_number` is intentionally a feature). The
# model feature set is read from the dataset this way — not reconstructed from the
# feature registry — so training works for whatever feature tiers the committed
# dataset happens to contain (Tier 0-2 by default, Tier 0-3 if built with history).
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {"event", "group_number", "game_url", LABEL_COLUMN}
)

# Fixed class order so confusion matrices and probability columns line up across
# every model and run.
LABELS: list[str] = ["win", "draw", "loss"]

# The Tier 0 Elo-expectation column (produced by the feature layer, written by the
# miner). Referenced here so the Elo baseline model can select it; not created here.
ELO_FEATURE = "elo_expected_white"


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The model-input columns present in a loaded dataset (order preserved)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a dataset produced by `chess-dataset` (parquet or csv)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Build it first with `uv run chess-dataset`."
        )
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset extension: {path.suffix!r}")


@dataclass
class Split:
    """A named train/test partition."""

    name: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    def describe(self) -> dict[str, int]:
        return {"n_train": len(self.X_train), "n_test": len(self.X_test)}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing features or label so estimators don't choke."""
    return df.dropna(subset=feature_columns(df) + [LABEL_COLUMN])


def pooled_split(
    df: pd.DataFrame, test_size: float = 0.25, seed: int = 42
) -> Split:
    """Stratified random split across all events pooled together."""
    df = _clean(df)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[LABEL_COLUMN],
    )
    return Split(
        name=f"pooled_{int((1 - test_size) * 100)}-{int(test_size * 100)}",
        X_train=train_df,
        y_train=train_df[LABEL_COLUMN],
        X_test=test_df,
        y_test=test_df[LABEL_COLUMN],
    )


def temporal_split(df: pd.DataFrame, n_test_events: int = 1) -> Split:
    """Train on earlier events, test on the latest `n_test_events` (by date).

    This is the split that matches how the model would actually be used —
    predicting *future* games from *past* ones — so it is the honest measure of
    generalization. Event labels embed the ISO date (``...-2026-06-30``), so a
    plain lexicographic sort is chronological.
    """
    df = _clean(df)
    events = sorted(df["event"].dropna().unique())
    if len(events) < 2:
        raise ValueError(
            f"Temporal split needs >= 2 distinct events; got {len(events)}."
        )
    n_test_events = max(1, min(n_test_events, len(events) - 1))
    test_events = set(events[-n_test_events:])
    train_df = df[~df["event"].isin(test_events)]
    test_df = df[df["event"].isin(test_events)]
    held = ",".join(sorted(test_events))
    return Split(
        name=f"temporal_holdout__test_{held}",
        X_train=train_df,
        y_train=train_df[LABEL_COLUMN],
        X_test=test_df,
        y_test=test_df[LABEL_COLUMN],
    )


def default_splits(df: pd.DataFrame, test_size: float, seed: int) -> list[Split]:
    """The split set evaluated by default: a pooled stratified split plus, when
    there is more than one event, a temporal holdout (past → latest event)."""
    splits = [pooled_split(df, test_size=test_size, seed=seed)]

    events = df["event"].dropna().unique()
    if len(events) >= 2:
        splits.append(temporal_split(df, n_test_events=1))
    return splits
