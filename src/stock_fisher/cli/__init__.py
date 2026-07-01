"""Command-line entrypoints.

Two small composition scripts, one per pipeline phase, kept together so the
project's CLIs live in one obvious place:

  - `dataset.py` -> `chess-dataset`: mine the configured events and write the
    one-row-per-game dataset.
  - `train.py`   -> `chess-train`: train the models on that dataset and track the
    runs in MLflow.

They are glue only: argument parsing plus wiring the library layers together. All
real logic lives in `ingestion`, `features`, `dataset`, and `modeling`.
"""
