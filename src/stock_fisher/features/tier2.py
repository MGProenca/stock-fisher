"""Tier 2 — in-tournament form / momentum.

    in_tourn_white_points_pre, in_tourn_black_points_pre, in_tourn_points_diff_pre,
    in_tourn_white_streak,     in_tourn_black_streak,
    in_tourn_white_games_played, in_tourn_black_games_played,
    in_tourn_white_recent_winrate, in_tourn_black_recent_winrate

Each value is reconstructed from a player's results in **earlier rounds of the
same event only** (rounds 1..r-1), so it is strictly pre-game for round r.

Leakage safety: this is the one feature block that depends on other rows, so it
is the easy place to leak future information. The reconstruction guards against
it structurally — when iterating rounds in order, the pre-round state for every
player in round r is *snapshotted before any round-r result is applied*. Nothing
from round r or later can influence round r's features.

Missing values are filled with neutral defaults (0 points/games/streak, 0.5
recent score) rather than NaN, so round 1 (no prior games) stays in the dataset
instead of being dropped by the model's NaN filter.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..ingestion.models import GameRecord
from .base import FeatureExtractor
from .labels import outcome_from_result

# Points scored by the player holding each outcome (standard chess scoring).
_SCORE = {"win": 1.0, "draw": 0.5, "loss": 0.0}

# Window for the "recent form" feature.
_RECENT_K = 3

# Neutral pre-state for a player with no prior games (round 1, or not found).
_EMPTY = {"points": 0.0, "games": 0, "streak": 0, "recent": 0.5}


def _streak(outcomes: list[str]) -> int:
    """Signed current streak: +n consecutive wins, -n consecutive losses, 0 on a
    trailing draw or empty history."""
    if not outcomes:
        return 0
    last = outcomes[-1]
    if last == "draw":
        return 0
    count = 0
    for o in reversed(outcomes):
        if o == last:
            count += 1
        else:
            break
    return count if last == "win" else -count


def _recent(outcomes: list[str], k: int = _RECENT_K) -> float:
    """Mean score (win=1, draw=0.5, loss=0) over the last k games; 0.5 if none."""
    if not outcomes:
        return 0.5
    window = outcomes[-k:]
    return sum(_SCORE[o] for o in window) / len(window)


class FormFeatureExtractor(FeatureExtractor):
    """In-tournament running form, reconstructed leak-free per event."""

    name = "tier2.form"
    columns = (
        "in_tourn_white_points_pre",
        "in_tourn_black_points_pre",
        "in_tourn_points_diff_pre",
        "in_tourn_white_streak",
        "in_tourn_black_streak",
        "in_tourn_white_games_played",
        "in_tourn_black_games_played",
        "in_tourn_white_recent_winrate",
        "in_tourn_black_recent_winrate",
    )

    def __init__(self) -> None:
        # (event, round_number, username_lower) -> pre-round snapshot dict.
        self._pre: dict[tuple[str, int, str], dict[str, Any]] = {}

    # -- one-shot reconstruction -------------------------------------------

    def prepare(self, records: list[GameRecord]) -> None:
        self._pre = {}

        # Group games by event, then by round.
        by_event: dict[str, dict[int, list[GameRecord]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for rec in records:
            by_event[rec.event][rec.round_number].append(rec)

        for event, rounds in by_event.items():
            # Running per-player state, accumulated as rounds advance.
            state: dict[str, dict[str, Any]] = defaultdict(
                lambda: {"points": 0.0, "games": 0, "outcomes": []}
            )

            for r in sorted(rounds):
                games = rounds[r]

                # 1) Snapshot pre-round state for every player in round r BEFORE
                #    applying any round-r result (this is what makes it leak-free).
                for rec in games:
                    for user in (self._user(rec.white), self._user(rec.black)):
                        if user is None:
                            continue
                        st = state[user]
                        self._pre[(event, r, user)] = {
                            "points": st["points"],
                            "games": st["games"],
                            "streak": _streak(st["outcomes"]),
                            "recent": _recent(st["outcomes"]),
                        }

                # 2) Apply round-r results to the running state.
                for rec in games:
                    self._apply(state, self._user(rec.white), rec.white.get("result"))
                    self._apply(state, self._user(rec.black), rec.black.get("result"))

    @staticmethod
    def _user(player: dict[str, Any]) -> str | None:
        username = player.get("username")
        return username.lower() if username else None

    @staticmethod
    def _apply(state: dict, user: str | None, result_code: str | None) -> None:
        outcome = outcome_from_result(result_code)
        if user is None or outcome is None:
            return  # can't score an unknown/missing result; skip it
        st = state[user]
        st["points"] += _SCORE[outcome]
        st["games"] += 1
        st["outcomes"].append(outcome)

    # -- per-game lookup ----------------------------------------------------

    def extract(self, record: GameRecord) -> dict[str, Any]:
        w = self._user(record.white)
        b = self._user(record.black)
        pw = self._pre.get((record.event, record.round_number, w), _EMPTY)
        pb = self._pre.get((record.event, record.round_number, b), _EMPTY)

        return {
            "in_tourn_white_points_pre": pw["points"],
            "in_tourn_black_points_pre": pb["points"],
            "in_tourn_points_diff_pre": pw["points"] - pb["points"],
            "in_tourn_white_streak": pw["streak"],
            "in_tourn_black_streak": pb["streak"],
            "in_tourn_white_games_played": pw["games"],
            "in_tourn_black_games_played": pb["games"],
            "in_tourn_white_recent_winrate": pw["recent"],
            "in_tourn_black_recent_winrate": pb["recent"],
        }
