"""Command-line entrypoint: mine the configured events and write the dataset.

    uv run chess-dataset --out data/output/titled_tuesday.parquet

Run `uv run chess-dataset --help` for options.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..config import EVENTS, OUTPUT_DIR, IngestionConfig
from ..dataset import build_dataset
from ..ingestion import ChessApiClient, fetch_events_games


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chess-dataset",
        description="Mine Chess.com Titled Tuesday data into a game-outcome dataset.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR / "titled_tuesday.parquet",
        help="Output path. Extension picks the format: .parquet (default) or .csv.",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=None,
        help="Override the User-Agent header sent to the API.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the on-disk cache and always hit the network.",
    )
    parser.add_argument(
        "--no-fair-play-filter",
        action="store_true",
        help="Keep games involving fair-play-removed accounts (off by default).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v for INFO, -vv for DEBUG logging.",
    )
    return parser.parse_args(argv)


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def _write(frame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(out, index=False)
    elif suffix in (".parquet", ".pq"):
        frame.to_parquet(out, index=False)
    else:
        raise SystemExit(f"Unsupported output extension: {suffix!r} (use .parquet or .csv)")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    log = logging.getLogger("stock_fisher.cli")

    config = IngestionConfig(use_cache=not args.no_cache)
    if args.user_agent:
        config.user_agent = args.user_agent

    fair_play: set[str] = set()
    sink = None if args.no_fair_play_filter else fair_play

    log.info("mining %d event(s): %s", len(EVENTS), ", ".join(e.label for e in EVENTS))
    with ChessApiClient(config) as client:
        records = list(fetch_events_games(client, config.events, fair_play_sink=sink))
        log.info("fetched %d unique games", len(records))
        if sink:
            log.info("fair-play-removed accounts seen: %d", len(fair_play))

        result = build_dataset(
            records,
            fair_play_usernames=fair_play if not args.no_fair_play_filter else None,
        )

    _write(result.frame, args.out)

    # Human-readable run summary.
    stats = result.stats.as_dict()
    print(f"Wrote {result.stats.kept} rows -> {args.out}")
    print(f"Feature columns: {', '.join(result.feature_columns)}")
    print("Filter summary:")
    for key, value in stats.items():
        print(f"  {key:24s} {value}")
    if not result.frame.empty:
        print("Label distribution:")
        counts = result.frame["label"].value_counts()
        for label, count in counts.items():
            print(f"  {label:8s} {count}  ({count / len(result.frame):.1%})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
