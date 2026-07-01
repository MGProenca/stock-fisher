"""Data-leakage guards — the correctness property that matters most here.

A feature must be knowable *before the first move*. Two places can leak the future:
  1. Tier 2 in-tournament form, which reads other rows (earlier rounds).
  2. The train/test splits, which must not share events/rows across the boundary.
Both are pinned down below.
"""

from __future__ import annotations

from stock_fisher.dataset import build_dataset
from stock_fisher.features import default_registry
from stock_fisher.features.tier2 import FormFeatureExtractor
from stock_fisher.modeling.data import feature_columns, pooled_split, temporal_split


# --- Tier 2: in-tournament form must only see earlier rounds -----------------

def test_form_round1_uses_neutral_defaults(make_record):
    """Round 1 has no prior games, so form must be the neutral prior, not NaN."""
    ex = FormFeatureExtractor()
    recs = [make_record(event="E", rnd=1, white="A", black="B")]
    ex.prepare(recs)
    out = ex.extract(recs[0])
    assert out["in_tourn_white_points_pre"] == 0.0
    assert out["in_tourn_white_games_played"] == 0
    assert out["in_tourn_white_streak"] == 0
    assert out["in_tourn_white_recent_winrate"] == 0.5


def test_form_does_not_leak_the_current_round_result(make_record):
    """The heart of the leakage guard: a player who *wins* round 1 must still have
    empty pre-round form on that round-1 row; the win only shows up from round 2."""
    recs = [
        make_record(event="E", rnd=1, white="A", black="B", white_result="win", black_result="resigned"),
        make_record(event="E", rnd=2, white="A", black="C", white_result="win", black_result="resigned"),
    ]
    ex = FormFeatureExtractor()
    ex.prepare(recs)
    r1 = ex.extract(recs[0])
    r2 = ex.extract(recs[1])

    # Round 1: A already won this game, but its result must NOT be in the features.
    assert r1["in_tourn_white_points_pre"] == 0.0
    assert r1["in_tourn_white_streak"] == 0

    # Round 2: now the round-1 win is visible (1 point, 1 game, +1 streak).
    assert r2["in_tourn_white_points_pre"] == 1.0
    assert r2["in_tourn_white_games_played"] == 1
    assert r2["in_tourn_white_streak"] == 1


def test_form_tracks_the_losing_player_too(make_record):
    """B loses round 1 (as Black); by round 2 that loss is reflected in B's form."""
    recs = [
        make_record(event="E", rnd=1, white="A", black="B", white_result="win", black_result="resigned"),
        make_record(event="E", rnd=2, white="B", black="C", white_result="resigned", black_result="win"),
    ]
    ex = FormFeatureExtractor()
    ex.prepare(recs)
    r2 = ex.extract(recs[1])
    assert r2["in_tourn_white_points_pre"] == 0.0     # B scored 0 in round 1
    assert r2["in_tourn_white_games_played"] == 1
    assert r2["in_tourn_white_streak"] == -1          # one loss


# --- Feature allow-list: no post-game field surfaces as a model input --------

def test_dataset_exposes_only_declared_pregame_features(make_record):
    recs = [make_record(rnd=i, url=f"u{i}") for i in range(1, 4)]
    result = build_dataset(recs)
    cols = set(result.frame.columns)

    # The produced feature set is exactly what the extractors declare — an
    # explicit allow-list, so nothing sneaks in.
    assert set(result.feature_columns) == set(default_registry().feature_columns())

    # Raw result / outcome fields never appear as columns.
    for leaky in ("result", "white_result", "black_result", "outcome"):
        assert leaky not in cols

    # Model-input columns exclude identifiers and the label itself.
    feats = feature_columns(result.frame)
    for non_feature in ("label", "event", "game_url", "group_number"):
        assert non_feature not in feats
    assert "round_number" in feats  # this one *is* an intentional feature


# --- Split-level leakage: train and test must not overlap --------------------

def _multi_event_df(make_record):
    """A tiny dataset over three chronologically-labelled events."""
    recs = []
    for ev in ("tt-2026-01-06", "tt-2026-02-10", "tt-2026-03-10"):
        for r in range(1, 4):
            recs.append(
                make_record(event=ev, rnd=r, url=f"{ev}-{r}", white_rating=2500 + r)
            )
    return build_dataset(recs).frame


def test_temporal_split_holds_out_latest_event_only(make_record):
    df = _multi_event_df(make_record)
    split = temporal_split(df, n_test_events=1)

    train_events = set(split.X_train["event"])
    test_events = set(split.X_test["event"])

    assert test_events == {"tt-2026-03-10"}            # the latest event
    assert train_events.isdisjoint(test_events)         # no event on both sides
    assert set(split.X_train["game_url"]).isdisjoint(split.X_test["game_url"])


def test_pooled_split_has_no_row_overlap(make_record):
    # Mix of win / loss / draw so the stratified split has all three classes.
    recs = []
    for i in range(300):
        kind = i % 3
        wr, br = ("win", "resigned") if kind == 0 else (
            ("resigned", "win") if kind == 1 else ("agreed", "agreed")
        )
        recs.append(make_record(event="E", rnd=(i % 11) + 1, url=f"u{i}",
                                white_rating=2400 + (i % 50), white_result=wr, black_result=br))
    df = build_dataset(recs).frame
    split = pooled_split(df, test_size=0.25, seed=0)

    assert set(split.X_train.index).isdisjoint(split.X_test.index)
    assert len(split.X_train) + len(split.X_test) == len(df)  # partition, nothing dropped/dupd
