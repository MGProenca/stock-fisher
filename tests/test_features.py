"""Tests for the pre-game feature math (Tier 0 Elo + Tier 1 ratings/round)."""

from __future__ import annotations

from stock_fisher.features.tier0 import EloExpectationFeatureExtractor
from stock_fisher.features.tier1 import RatingFeatureExtractor, RoundFeatureExtractor


def _elo(rec):
    return EloExpectationFeatureExtractor().extract(rec)["elo_expected_white"]


def test_elo_equal_ratings_is_half(make_record):
    assert abs(_elo(make_record(white_rating=2500, black_rating=2500)) - 0.5) < 1e-12


def test_elo_is_symmetric_and_in_range(make_record):
    hi = _elo(make_record(white_rating=2800, black_rating=2400))
    lo = _elo(make_record(white_rating=2400, black_rating=2800))
    assert 0.5 < hi < 1.0
    assert 0.0 < lo < 0.5
    assert abs(hi + lo - 1.0) < 1e-12  # White's + Black's expectation sum to 1


def test_elo_known_value_for_400_point_gap(make_record):
    # A 400-point edge is the canonical Elo example: expected score = 10/11.
    assert abs(_elo(make_record(white_rating=2800, black_rating=2400)) - 10 / 11) < 1e-9


def test_elo_missing_rating_is_none(make_record):
    assert _elo(make_record(white_rating=None)) is None


def test_rating_features(make_record):
    out = RatingFeatureExtractor().extract(make_record(white_rating=2600, black_rating=2500))
    assert out["rating_diff"] == 100          # signed, White's view
    assert out["abs_rating_diff"] == 100
    assert out["mean_rating"] == 2550
    assert out["white_rating"] == 2600 and out["black_rating"] == 2500


def test_rating_diff_sign_flips_for_weaker_white(make_record):
    out = RatingFeatureExtractor().extract(make_record(white_rating=2400, black_rating=2500))
    assert out["rating_diff"] == -100
    assert out["abs_rating_diff"] == 100


def test_rating_missing_degrades_to_none(make_record):
    out = RatingFeatureExtractor().extract(make_record(white_rating=None, black_rating=2500))
    assert out["rating_diff"] is None and out["mean_rating"] is None


def test_round_number_passthrough(make_record):
    assert RoundFeatureExtractor().extract(make_record(rnd=7))["round_number"] == 7
