"""Tier 0 — the Elo expectation baseline feature.

    elo_expected_white = 1 / (1 + 10^(-rating_diff/400))

The logistic Elo expected score for White. It is a deterministic transform of the
pre-game rating difference, so it is strictly pre-game (no leakage), and on its own
it doubles as the "bar to beat" baseline model.

It lives in the feature layer (not the modeling layer) so it is written into the
dataset alongside every other feature — models only ever *consume* columns.
"""

from __future__ import annotations

from typing import Any

from ..ingestion.models import GameRecord
from .base import FeatureExtractor


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class EloExpectationFeatureExtractor(FeatureExtractor):
    """Logistic Elo expected score for White."""

    name = "tier0.elo"
    columns = ("elo_expected_white",)

    def extract(self, record: GameRecord) -> dict[str, Any]:
        white = _as_int(record.white.get("rating"))
        black = _as_int(record.black.get("rating"))

        elo = None
        if white is not None and black is not None:
            elo = 1.0 / (1.0 + 10.0 ** (-(white - black) / 400.0))

        return {"elo_expected_white": elo}
