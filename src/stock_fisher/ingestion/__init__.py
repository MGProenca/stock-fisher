"""Data ingestion: fetching raw data from the Chess.com API.

This layer is responsible *only* for getting raw game data off the network and
into `GameRecord` objects. It performs no feature engineering and makes no
modelling decisions. Feature logic lives in `stock_fisher.features`.
"""

from .client import ChessApiClient, ChessApiError, NotFoundError
from .models import GameRecord
from .tournament import fetch_events_games, fetch_tournament_games

__all__ = [
    "ChessApiClient",
    "ChessApiError",
    "NotFoundError",
    "GameRecord",
    "fetch_tournament_games",
    "fetch_events_games",
]
