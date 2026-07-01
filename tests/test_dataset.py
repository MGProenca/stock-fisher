"""Tests for dataset assembly: filtering to the modellable population + labelling.

These check that only rated, standard-chess, blitz, labelable games survive, that
the reasons are counted, and that the output schema is stable.
"""

from __future__ import annotations

from stock_fisher.dataset import build_dataset


def test_keeps_valid_games_and_labels_them(make_record):
    recs = [
        make_record(url="a", white_result="win", black_result="resigned"),
        make_record(url="b", white_result="agreed", black_result="agreed"),
    ]
    result = build_dataset(recs)
    assert result.stats.kept == 2
    assert sorted(result.frame["label"]) == ["draw", "win"]


def test_drops_non_blitz(make_record):
    result = build_dataset([make_record(time_class="rapid")])
    assert result.stats.dropped_not_blitz == 1
    assert result.stats.kept == 0


def test_drops_unrated(make_record):
    result = build_dataset([make_record(rated=False)])
    assert result.stats.dropped_unrated == 1
    assert result.stats.kept == 0


def test_drops_non_standard_variants(make_record):
    result = build_dataset([make_record(rules="chess960")])
    assert result.stats.dropped_wrong_rules == 1
    assert result.stats.kept == 0


def test_drops_unlabelable_contradiction(make_record):
    result = build_dataset([make_record(white_result="win", black_result="win")])
    assert result.stats.dropped_unlabelable == 1
    assert result.stats.kept == 0


def test_fair_play_filter_drops_flagged_accounts(make_record):
    recs = [
        make_record(url="clean", white="alice", black="bob"),
        make_record(url="flagged", white="cheater", black="bob"),
    ]
    result = build_dataset(recs, fair_play_usernames={"cheater"})
    assert result.stats.dropped_fair_play == 1
    assert result.stats.kept == 1
    assert list(result.frame["game_url"]) == ["clean"]


def test_empty_input_yields_stable_schema(make_record):
    result = build_dataset([])
    assert result.stats.kept == 0
    assert len(result.frame) == 0
    # Columns still present so downstream code sees a consistent schema.
    assert "label" in result.frame.columns
    for col in result.feature_columns:
        assert col in result.frame.columns
