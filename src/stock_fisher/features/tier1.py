"""Tier 1 — core pre-game features.

    white_rating, black_rating, rating_diff, abs_rating_diff, mean_rating,
    round_number

All are knowable before the first move: the ratings come from inside the game
object (the players' blitz rating *at the time of the game*, §1) and the round
index comes from the traversal path. Nothing here touches how the game unfolded.

Higher tiers (form, enrichment) will be added as additional extractor modules and
registered alongside these; they do not modify this file.
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


class RatingFeatureExtractor(FeatureExtractor):
    """Rating-based features — the primary pre-game signal."""

    name = "tier1.rating"
    columns = (
        "white_rating",
        "black_rating",
        "rating_diff",
        "abs_rating_diff",
        "mean_rating",
    )

    def extract(self, record: GameRecord) -> dict[str, Any]:
        white = _as_int(record.white.get("rating"))
        black = _as_int(record.black.get("rating"))

        rating_diff = abs_rating_diff = mean_rating = None
        if white is not None and black is not None:
            rating_diff = white - black                 # signed, White's view
            abs_rating_diff = abs(rating_diff)
            mean_rating = (white + black) / 2.0

        return {
            "white_rating": white,
            "black_rating": black,
            "rating_diff": rating_diff,
            "abs_rating_diff": abs_rating_diff,
            "mean_rating": mean_rating,
        }


class RoundFeatureExtractor(FeatureExtractor):
    """Round index within the event — a fatigue / late-event proxy."""

    name = "tier1.round"
    columns = ("round_number",)

    def extract(self, record: GameRecord) -> dict[str, Any]:
        return {"round_number": record.round_number}
