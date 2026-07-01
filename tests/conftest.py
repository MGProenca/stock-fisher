"""Shared test fixtures.

Everything here builds synthetic `GameRecord`s in memory, so the whole suite runs
offline and deterministically — no API calls, no cache, no committed data needed.
"""

from __future__ import annotations

import pytest

from stock_fisher.ingestion.models import GameRecord


@pytest.fixture
def make_record():
    """Factory for a single game record.

    Defaults describe a valid, labelable game (White wins a rated blitz game).
    Override any field to exercise a specific branch (filters, leakage, labels).
    """

    def _make(
        *,
        event: str = "E1",
        rnd: int = 1,
        group: int = 1,
        white: str = "alice",
        black: str = "bob",
        white_rating=2500,
        black_rating=2500,
        white_result: str | None = "win",
        black_result: str | None = "resigned",
        rules: str = "chess",
        time_class: str = "blitz",
        rated: bool = True,
        url: str | None = None,
    ) -> GameRecord:
        game = {
            "rules": rules,
            "time_class": time_class,
            "rated": rated,
            "white": {"username": white, "rating": white_rating, "result": white_result},
            "black": {"username": black, "rating": black_rating, "result": black_result},
            "url": url or f"https://example.test/{event}/{rnd}/{group}/{white}-{black}",
        }
        return GameRecord(event=event, round_number=rnd, group_number=group, game=game)

    return _make
