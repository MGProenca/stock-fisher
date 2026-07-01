"""Central configuration: the events to mine and ingestion defaults.

Keeping this in one place means adding a new tournament (or pointing the cache
somewhere else) is a one-line change, not a code change scattered across modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# API / HTTP
# ---------------------------------------------------------------------------

BASE_URL = "https://api.chess.com/pub"

# Chess.com asks every client to identify itself. Override via the CLI / env if
# you fork this; include a contact so they can reach you if needed.
DEFAULT_USER_AGENT = (
    "chess-dataset/0.1 (Titled Tuesday outcome modelling; contact: "
    "martim.proenca@traivefinance.com)"
)

# Be polite and resilient. The cache makes re-runs offline, so a simple
# sequential client with retry/backoff is plenty for this data volume.
REQUEST_TIMEOUT = 30.0       # seconds per request
MAX_RETRIES = 5              # on 429 / 5xx / connection errors
BACKOFF_BASE = 1.0          # seconds; exponential: base * 2**attempt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "output"

# MLflow tracking. By default we log to a local SQLite backend inside the repo so
# a fresh clone runs with ZERO setup (no server to start). Artifacts go under
# data/mlartifacts/. Override with --tracking-uri or the MLFLOW_TRACKING_URI env
# var — e.g. point it at http://127.0.0.1:5000 to centralize runs in a server.
# The absolute path keeps the URI independent of the current working directory.
MLFLOW_DB_PATH = DATA_DIR / "mlflow.db"
MLARTIFACTS_DIR = DATA_DIR / "mlartifacts"
DEFAULT_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.resolve()}"
DEFAULT_EXPERIMENT_NAME = "titled-tuesday-outcomes"


# ---------------------------------------------------------------------------
# Events to mine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """A single tournament to ingest.

    `slug` is the tournament `url-ID` used to build `/pub/tournament/{slug}`.
    `label` is a short, stable name stamped onto every row (the `event` column).
    """

    label: str
    slug: str


# Titled Tuesday blitz events to mine — one per month, Feb–Jun 2026. The brief
# suggests two events; we use five (a bit more data for a more robust temporal
# evaluation) while keeping scope tight. Slugs are the API `url-ID` from each
# event URL, each confirmed fetchable via /pub/tournament/{slug}. Labels embed the
# ISO date so a lexicographic sort is chronological (the temporal split relies on
# this). Adding more events is a one-line change here.
EVENTS: list[Event] = [
    Event("titled-tuesday-blitz-2026-02-10", "titled-tuesday-blitz-february-10-2026-6221327"),
    Event("titled-tuesday-blitz-2026-03-10", "titled-tuesday-blitz-march-10-2026-6277141"),
    Event("titled-tuesday-blitz-2026-04-14", "titled-tuesday-blitz-april-14-2026-6362193"),
    Event("titled-tuesday-blitz-2026-05-12", "titled-tuesday-blitz-may-12-2026-6431785"),
    Event("titled-tuesday-blitz-2026-06-09", "titled-tuesday-blitz-june-09-2026-6521721"),
]


@dataclass
class IngestionConfig:
    """Runtime knobs for a mining run. Defaults pull from the constants above."""

    user_agent: str = DEFAULT_USER_AGENT
    cache_dir: Path = CACHE_DIR
    timeout: float = REQUEST_TIMEOUT
    max_retries: int = MAX_RETRIES
    backoff_base: float = BACKOFF_BASE
    use_cache: bool = True
    events: list[Event] = field(default_factory=lambda: list(EVENTS))
