"""The hand-off object between ingestion and feature engineering.

A `GameRecord` is the *raw* game JSON exactly as the API returned it, plus the
traversal context the API does not put inside the game object itself (which event
it came from, which round, which group). Feature extractors consume these; they
never touch the network. Keeping the raw `game` dict intact means new features
can mine fields we don't use yet without re-running ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GameRecord:
    event: str          # Event.label
    round_number: int   # 1-based
    group_number: int   # 1-based
    game: dict[str, Any]  # raw game object (see API reference §9)

    # Convenience accessors used widely by feature extractors. They tolerate the
    # field being absent so a malformed game degrades to None rather than KeyError.

    @property
    def white(self) -> dict[str, Any]:
        return self.game.get("white", {}) or {}

    @property
    def black(self) -> dict[str, Any]:
        return self.game.get("black", {}) or {}

    @property
    def url(self) -> str | None:
        return self.game.get("url")

    @property
    def uuid(self) -> str | None:
        return self.game.get("uuid")

    def dedup_key(self) -> str:
        """Stable identity for a game, used to drop duplicates."""
        return self.uuid or self.url or f"{self.event}:{self.round_number}:{id(self.game)}"
