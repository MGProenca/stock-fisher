"""Thin HTTP client for the Chess.com Published-Data API.

Responsibilities, deliberately narrow:
  - send GET requests with the required User-Agent header,
  - retry transient failures (429 / 5xx / connection drops) with exponential
    backoff, honoring `Retry-After` when present,
  - read/write through the on-disk cache.

It knows nothing about tournaments, games, or features — just "give me the JSON
at this URL". Higher layers compose it. Finished-tournament data is immutable, so
the cache makes re-runs instant and offline; throughput is never the bottleneck,
which is why a simple sequential client (no concurrency) is enough here.
"""

from __future__ import annotations

import logging
import time

import requests

from ..config import IngestionConfig
from .cache import JsonCache

logger = logging.getLogger(__name__)


class ChessApiError(RuntimeError):
    """A request ultimately failed (after retries) or returned a non-retryable
    4xx (other than 404/410, which callers may treat as 'missing')."""


class NotFoundError(ChessApiError):
    """The resource does not exist (HTTP 404 / 410)."""


class ChessApiClient:
    def __init__(self, config: IngestionConfig | None = None) -> None:
        self.config = config or IngestionConfig()
        self.cache = JsonCache(self.config.cache_dir) if self.config.use_cache else None
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.config.user_agent,
                "Accept": "application/json",
            }
        )

    # -- public API ---------------------------------------------------------

    def get_json(self, url: str):
        """Fetch and parse JSON at `url`, using the cache when enabled.

        Raises NotFoundError on 404/410, ChessApiError on persistent failure.
        """
        if self.cache is not None:
            cached = self.cache.get(url)
            if cached is not None:
                logger.debug("cache hit: %s", url)
                return cached

        payload = self._fetch_with_retries(url)
        if self.cache is not None:
            self.cache.set(url, payload)
        return payload

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "ChessApiClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals ----------------------------------------------------------

    def _fetch_with_retries(self, url: str):
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=self.config.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("request error for %s (attempt %d): %s", url, attempt, exc)
                self._sleep_backoff(attempt)
                continue

            if resp.status_code == 200:
                logger.debug("fetched: %s", url)
                return resp.json()

            if resp.status_code in (404, 410):
                raise NotFoundError(f"{resp.status_code} for {url}")

            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = self._retry_after_seconds(resp)
                logger.warning(
                    "transient %d for %s (attempt %d); backing off",
                    resp.status_code,
                    url,
                    attempt,
                )
                self._sleep_backoff(attempt, retry_after)
                last_exc = ChessApiError(f"{resp.status_code} for {url}")
                continue

            # Other 4xx: not retryable.
            raise ChessApiError(f"{resp.status_code} for {url}: {resp.text[:200]}")

        raise ChessApiError(f"giving up on {url} after retries") from last_exc

    def _sleep_backoff(self, attempt: int, retry_after: float | None = None) -> None:
        delay = retry_after if retry_after is not None else self.config.backoff_base * (2 ** attempt)
        time.sleep(delay)

    @staticmethod
    def _retry_after_seconds(resp: requests.Response) -> float | None:
        header = resp.headers.get("Retry-After")
        if not header:
            return None
        try:
            return float(header)
        except ValueError:
            return None
