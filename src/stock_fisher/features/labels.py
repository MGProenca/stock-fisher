"""Target derivation: map a game's result codes to a White-perspective label.

The label is the thing we predict, not a feature, so it lives apart from the
feature extractors. We derive it from `white.result` and cross-check against
`black.result` to catch inconsistent rows.

Chess.com encodes the *manner* of each player's result (checkmated, timeout,
resigned, agreed, …); `outcome_from_result` collapses those codes into a plain
win/loss/draw for the player holding the code. It is used both here (labelling)
and by the in-tournament form features (`tier2`).
"""

from __future__ import annotations

from ..ingestion.models import GameRecord

# Canonical three-class label space, White's perspective.
LABEL_WIN = "win"
LABEL_DRAW = "draw"
LABEL_LOSS = "loss"

# Raw Chess.com `result` code -> outcome for the player holding that code.
RESULT_TO_OUTCOME: dict[str, str] = {
    # win
    "win": "win",
    # losses (the manner varies; the outcome does not)
    "checkmated": "loss",
    "timeout": "loss",
    "resigned": "loss",
    "lose": "loss",
    "abandoned": "loss",
    "kingofthehill": "loss",
    "threecheck": "loss",
    "bughousepartnerlose": "loss",
    # draws
    "agreed": "draw",
    "repetition": "draw",
    "stalemate": "draw",
    "insufficient": "draw",
    "50move": "draw",
    "timevsinsufficient": "draw",
}

# Opposite outcome, for the consistency cross-check.
_OPPOSITE = {"win": "loss", "loss": "win", "draw": "draw"}


def outcome_from_result(result_code: str | None) -> str | None:
    """Outcome ('win'/'loss'/'draw') for the player holding `result_code`."""
    if result_code is None:
        return None
    return RESULT_TO_OUTCOME.get(result_code)


def derive_label(record: GameRecord) -> str | None:
    """White-perspective label for a game, or None if it can't be determined.

    Returns None when either result code is missing/unknown, or when White's and
    Black's results contradict each other (a sign of a malformed/odd game we'd
    rather drop than mislabel).
    """
    white_outcome = outcome_from_result(record.white.get("result"))
    black_outcome = outcome_from_result(record.black.get("result"))

    if white_outcome is None:
        return None

    # Cross-check: Black's outcome should be the opposite of White's.
    if black_outcome is not None and black_outcome != _OPPOSITE[white_outcome]:
        return None

    return white_outcome
