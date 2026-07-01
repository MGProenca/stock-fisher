"""Feature-extractor framework.

A `FeatureExtractor` turns one `GameRecord` into a dict of named columns. The
dataset builder runs every *registered* extractor over every game and merges the
column dicts into a row. Adding a new feature block means writing a new extractor
and registering it — no change to the builder, the ingestion layer, or existing
extractors.

The leakage rule is enforced structurally: extractors are the only place features
are produced, and each declares its output columns, so the admissible (pre-game)
feature set is always an explicit, inspectable allow-list.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..ingestion.models import GameRecord


class FeatureExtractor(ABC):
    """Produces a set of feature columns from a single game.

    Subclasses must set `name` (for logging/debugging) and `columns` (the exact
    keys `extract` returns), and implement `extract`.
    """

    name: str = "unnamed"
    columns: tuple[str, ...] = ()

    @abstractmethod
    def extract(self, record: GameRecord) -> dict[str, Any]:
        """Return {column_name: value} for this game.

        Implementations should return the full `columns` key set every time,
        using None for values that cannot be computed, so the resulting table
        has a stable schema.
        """
        raise NotImplementedError

    def prepare(self, records: "list[GameRecord]") -> None:
        """Optional one-shot pass over *all* records before per-game extraction.

        Stateless extractors (e.g. ratings) ignore this. Extractors that need
        cross-row context — such as in-tournament form, which is reconstructed
        from a player's earlier-round results — precompute their lookup state
        here. `extract` is then a pure per-game lookup. Default: no-op.
        """
        return None


class FeatureRegistry:
    """Ordered collection of extractors applied to build the feature matrix."""

    def __init__(self) -> None:
        self._extractors: list[FeatureExtractor] = []

    def register(self, extractor: FeatureExtractor) -> FeatureExtractor:
        self._extractors.append(extractor)
        return extractor

    @property
    def extractors(self) -> list[FeatureExtractor]:
        return list(self._extractors)

    def feature_columns(self) -> list[str]:
        """All feature columns contributed by registered extractors, in order."""
        cols: list[str] = []
        for ex in self._extractors:
            cols.extend(ex.columns)
        return cols

    def prepare_all(self, records: list[GameRecord]) -> None:
        """Run every extractor's one-shot `prepare` pass over all records."""
        for ex in self._extractors:
            ex.prepare(records)

    def extract_all(self, record: GameRecord) -> dict[str, Any]:
        """Merge every extractor's output for one game into a single row dict."""
        row: dict[str, Any] = {}
        for ex in self._extractors:
            row.update(ex.extract(record))
        return row
