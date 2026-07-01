"""On-disk JSON cache keyed by request URL.

The Chess.com data for a finished tournament is immutable, so caching is a pure
win: it makes re-runs instant and keeps us from re-hammering the API while we
iterate on feature engineering. Each URL maps to one JSON file; the key is a
hash of the URL so arbitrary URLs become safe filenames.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class JsonCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def get(self, url: str) -> Any | None:
        path = self._path_for(url)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            # Corrupt / partial cache entry: treat as a miss and let it be
            # overwritten on the next successful fetch.
            return None

    def set(self, url: str, payload: Any) -> None:
        path = self._path_for(url)
        # Write to a temp file then atomically rename, so an interrupted run
        # never leaves a half-written entry that would later read as valid.
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        tmp.replace(path)
