"""Traverse a tournament into a flat stream of GameRecords.

Path through the API (see reference §4):

    /pub/tournament/{slug}            -> { rounds: [round_url, ...] }
    {round_url}                       -> { groups: [group_url, ...] }
    {group_url}                       -> { games: [game_obj, ...] }

Round and group numbers are recovered from the trailing path segments of their
URLs, so the records carry the round index (a legitimate pre-game feature) even
though the game object itself does not include it.
"""

from __future__ import annotations

import logging
from typing import Iterator

from ..config import BASE_URL, Event
from .client import ChessApiClient, NotFoundError
from .models import GameRecord

logger = logging.getLogger(__name__)


def _trailing_int(url: str, default: int = 0) -> int:
    """Last path segment of a round/group URL as an int (e.g. .../1/2 -> 2)."""
    try:
        return int(url.rstrip("/").rsplit("/", 1)[-1])
    except (ValueError, IndexError):
        return default


def fetch_tournament_games(
    client: ChessApiClient,
    event: Event,
    fair_play_sink: set[str] | None = None,
) -> Iterator[GameRecord]:
    """Yield every game in `event` as a GameRecord, in round/group order.

    If `fair_play_sink` is provided, each group's `fair_play_removals` usernames
    are added to it (lowercased) so the caller can drop affected games later.
    """
    tournament_url = f"{BASE_URL}/tournament/{event.slug}"
    logger.info("fetching tournament: %s", event.label)
    info = client.get_json(tournament_url)

    round_urls: list[str] = info.get("rounds", []) or []
    if not round_urls:
        logger.warning("tournament %s has no rounds", event.label)
        return

    for round_url in round_urls:
        round_number = _trailing_int(round_url)
        try:
            round_info = client.get_json(round_url)
        except NotFoundError:
            logger.warning("round not found, skipping: %s", round_url)
            continue

        for group_url in round_info.get("groups", []) or []:
            group_number = _trailing_int(group_url)
            try:
                group_info = client.get_json(group_url)
            except NotFoundError:
                logger.warning("group not found, skipping: %s", group_url)
                continue

            if fair_play_sink is not None:
                for username in group_info.get("fair_play_removals", []) or []:
                    fair_play_sink.add(username.lower())

            games = group_info.get("games", []) or []
            logger.debug(
                "round %d group %d: %d games", round_number, group_number, len(games)
            )
            for game in games:
                yield GameRecord(
                    event=event.label,
                    round_number=round_number,
                    group_number=group_number,
                    game=game,
                )


def fetch_events_games(
    client: ChessApiClient,
    events: list[Event],
    fair_play_sink: set[str] | None = None,
) -> Iterator[GameRecord]:
    """Yield GameRecords across several events, de-duplicated on game identity."""
    seen: set[str] = set()
    for event in events:
        for record in fetch_tournament_games(client, event, fair_play_sink):
            key = record.dedup_key()
            if key in seen:
                continue
            seen.add(key)
            yield record
