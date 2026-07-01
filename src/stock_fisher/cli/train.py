"""Command-line entrypoint for training + experiment tracking (MLflow).

    uv run chess-train -v

Loads the assembled dataset, trains the baseline + Tier 1 models across the
pooled and cross-event splits, logs every run to MLflow, and prints a comparison
table sorted by the held-out log loss.

By default it logs to a local SQLite backend inside the repo (zero setup — no
server needed), so a fresh clone reproduces the results out of the box. Override
with --tracking-uri or MLFLOW_TRACKING_URI, e.g. point it at a running server
with --tracking-uri http://127.0.0.1:5000 to centralize runs.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd

from ..config import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_URI,
    MLARTIFACTS_DIR,
    OUTPUT_DIR,
)
from ..modeling.data import LABELS, default_splits, feature_columns, load_dataset
from ..modeling.models import default_model_specs
from ..modeling.train import run_experiments


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chess-train",
        description="Train outcome models and track experiments with MLflow.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=OUTPUT_DIR / "titled_tuesday.parquet",
        help="Dataset built by `chess-dataset` (parquet or csv).",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help=(
            "MLflow tracking URI. Resolution order: this flag, then the "
            "MLFLOW_TRACKING_URI env var, then the default local SQLite backend "
            f"({DEFAULT_TRACKING_URI}). Point at http://127.0.0.1:5000 to log to "
            "a running MLflow server instead."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name to log runs under.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.25, help="Pooled-split test fraction."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v INFO, -vv DEBUG."
    )
    return parser.parse_args(argv)


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def _resolve_tracking_uri(arg: str | None) -> str:
    # Explicit flag wins; then the standard MLflow env var; then the local server.
    if arg:
        return arg
    return os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI


def _ensure_experiment(name: str, tracking_uri: str) -> None:
    """Create the experiment if new.

    Against a tracking server the server owns artifact storage, so we let it pick
    the location. Only for the local SQLite fallback do we pin artifacts under
    data/mlartifacts/.
    """
    if mlflow.get_experiment_by_name(name) is not None:
        return
    artifact_location = None
    if tracking_uri.startswith("sqlite:"):
        MLARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        artifact_location = MLARTIFACTS_DIR.resolve().as_uri()
    mlflow.create_experiment(name, artifact_location=artifact_location)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    df = load_dataset(args.dataset)

    # Dataset-level metadata logged with every run for provenance.
    class_counts = df["label"].value_counts().to_dict()
    dataset_meta = {
        "dataset": str(args.dataset),
        "n_rows": int(len(df)),
        "seed": args.seed,
        "class_counts": class_counts,
    }

    tracking_uri = _resolve_tracking_uri(args.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    _ensure_experiment(args.experiment_name, tracking_uri)

    feat_cols = feature_columns(df)
    specs = default_model_specs(feat_cols)
    splits = default_splits(df, test_size=args.test_size, seed=args.seed)

    summary = run_experiments(
        specs, splits, dataset_meta, experiment_name=args.experiment_name
    )

    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", None)

    print(f"\nDataset: {args.dataset}  ({len(df)} rows, {len(feat_cols)} features)")
    print(f"Class balance: {class_counts}")
    print(f"Labels (fixed order): {LABELS}")
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment:   {args.experiment_name}")

    for split_name, block in summary.groupby("split"):
        print(f"\n=== Split: {split_name} ===")
        block = block.drop(columns=["split", "run_id"]).sort_values("log_loss")
        print(block.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if tracking_uri.startswith("http"):
        print(f"\nInspect runs in the MLflow UI at {tracking_uri}")
    else:
        print(f"\nInspect runs:  uv run mlflow ui --backend-store-uri {tracking_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
