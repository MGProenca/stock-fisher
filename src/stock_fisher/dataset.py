"""Dataset assembly: GameRecords -> a tidy, one-row-per-game DataFrame.

This module is the orchestrator that wires ingestion and feature engineering
together. It:

  1. filters games to the modellable population (rated blitz standard chess,
     no fair-play-removed players),
  2. derives the White-perspective label and drops rows it can't label,
  3. runs the registered feature extractors,
  4. emits identifier columns + features + label as a DataFrame.

It owns no feature math itself — that all lives in `features/` — and no network
logic — that lives in `ingestion/`. Swapping the feature set (a different
registry) or the data source (a different record iterator) needs no change here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from .features import FeatureRegistry, default_registry
from .features.labels import derive_label
from .ingestion.models import GameRecord

logger = logging.getLogger(__name__)

# Identifier / bookkeeping columns kept alongside features for traceability and
# splitting (e.g. train-on-one-event / validate-on-the-other). These are NOT
# model inputs — the model should consume `feature_columns` + `label` only.
ID_COLUMNS = ("event", "round_number", "group_number", "game_url", "label")

LABEL_COLUMN = "label"


@dataclass
class FilterStats:
    """Counts of why rows were kept or dropped, for an auditable run summary."""

    total: int = 0
    dropped_wrong_rules: int = 0
    dropped_not_blitz: int = 0
    dropped_unrated: int = 0
    dropped_fair_play: int = 0
    dropped_unlabelable: int = 0
    kept: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "dropped_wrong_rules": self.dropped_wrong_rules,
            "dropped_not_blitz": self.dropped_not_blitz,
            "dropped_unrated": self.dropped_unrated,
            "dropped_fair_play": self.dropped_fair_play,
            "dropped_unlabelable": self.dropped_unlabelable,
            "kept": self.kept,
        }


@dataclass
class DatasetResult:
    frame: pd.DataFrame
    feature_columns: list[str]
    stats: FilterStats = field(default_factory=FilterStats)


def _passes_filters(record: GameRecord, stats: FilterStats) -> bool:
    """Apply the modellable-population inclusion rules.

    Note on fair play: the per-game object does not carry the group's
    `fair_play_removals` list, so here we drop a game only if a participating
    player's account result/status indicates removal is not derivable. The
    coarse, reliable signal available per game is the `rules`/`time_class`/
    `rated` fields; richer fair-play filtering can be layered in later when the
    group-level list is threaded through.
    """
    game = record.game

    if game.get("rules") != "chess":
        stats.dropped_wrong_rules += 1
        return False

    if game.get("time_class") != "blitz":
        stats.dropped_not_blitz += 1
        return False

    # `rated` may be absent on odd rows; treat missing as not-rated to be safe.
    if not game.get("rated", False):
        stats.dropped_unrated += 1
        return False

    return True


def build_dataset(
    records: Iterable[GameRecord],
    registry: FeatureRegistry | None = None,
    fair_play_usernames: set[str] | None = None,
) -> DatasetResult:
    """Build the feature matrix from an iterable of GameRecords.

    `fair_play_usernames` (optional, lowercased) lets a caller pass the union of
    every group's `fair_play_removals`; games touching those accounts are dropped.
    """
    registry = registry or default_registry()
    fair_play_usernames = fair_play_usernames or set()
    feature_columns = registry.feature_columns()
    stats = FilterStats()

    # Materialize so extractors that need cross-row context (e.g. Tier 2 form,
    # reconstructed from earlier rounds) can do a one-shot pass before we extract
    # per game. Reconstruction sees all games, including ones later filtered out.
    records = list(records)
    registry.prepare_all(records)

    rows: list[dict] = []
    for record in records:
        stats.total += 1

        if not _passes_filters(record, stats):
            continue

        white_user = (record.white.get("username") or "").lower()
        black_user = (record.black.get("username") or "").lower()
        if white_user in fair_play_usernames or black_user in fair_play_usernames:
            stats.dropped_fair_play += 1
            continue

        label = derive_label(record)
        if label is None:
            stats.dropped_unlabelable += 1
            continue

        row = {
            "event": record.event,
            "round_number": record.round_number,
            "group_number": record.group_number,
            "game_url": record.url,
            LABEL_COLUMN: label,
        }
        row.update(registry.extract_all(record))
        rows.append(row)
        stats.kept += 1

    # Build with an explicit, stable column order even when there are no rows.
    ordered_columns = list(ID_COLUMNS) + [
        c for c in feature_columns if c not in ID_COLUMNS
    ]
    frame = pd.DataFrame(rows, columns=ordered_columns)

    logger.info("dataset built: %s", stats.as_dict())
    return DatasetResult(frame=frame, feature_columns=feature_columns, stats=stats)
