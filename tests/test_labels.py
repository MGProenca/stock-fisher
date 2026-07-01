"""Tests for target derivation — the thing we predict.

A wrong label silently corrupts every downstream metric, so the mapping and the
White/Black consistency check are worth pinning down.
"""

from __future__ import annotations

from stock_fisher.features.labels import derive_label, outcome_from_result


def test_outcome_mapping_covers_win_loss_draw():
    assert outcome_from_result("win") == "win"
    # losses come in several "manners"; all collapse to loss
    for code in ("checkmated", "timeout", "resigned", "abandoned"):
        assert outcome_from_result(code) == "loss"
    # draws likewise
    for code in ("agreed", "repetition", "stalemate", "insufficient", "50move"):
        assert outcome_from_result(code) == "draw"


def test_outcome_unknown_or_missing_is_none():
    assert outcome_from_result(None) is None
    assert outcome_from_result("not-a-real-code") is None


def test_derive_label_white_perspective(make_record):
    assert derive_label(make_record(white_result="win", black_result="checkmated")) == "win"
    assert derive_label(make_record(white_result="resigned", black_result="win")) == "loss"
    assert derive_label(make_record(white_result="agreed", black_result="agreed")) == "draw"


def test_derive_label_drops_contradictory_rows(make_record):
    # Both players cannot win the same game — refuse to label it.
    assert derive_label(make_record(white_result="win", black_result="win")) is None


def test_derive_label_drops_missing_result(make_record):
    assert derive_label(make_record(white_result=None, black_result="win")) is None
