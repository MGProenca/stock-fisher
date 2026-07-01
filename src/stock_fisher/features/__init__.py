"""Feature engineering: turning raw GameRecords into model-ready rows.

This layer is responsible *only* for deriving labels and computing features from
already-fetched data. It never touches the network. Each group of features lives
in its own module and is wired together through `default_registry()`.

The features are grouped by what they depend on:
  - Tier 0 (`tier0`): the Elo expectation — the single-feature "bar to beat".
  - Tier 1 (`tier1`): pre-game ratings and the round index.
  - Tier 2 (`tier2`): in-tournament form, reconstructed leak-free from earlier
    rounds of the same event.
"""

from .base import FeatureExtractor, FeatureRegistry
from .labels import derive_label
from .tier0 import EloExpectationFeatureExtractor
from .tier1 import RatingFeatureExtractor, RoundFeatureExtractor
from .tier2 import FormFeatureExtractor


def default_registry() -> FeatureRegistry:
    """The set of feature extractors used to build the dataset (all pre-game)."""
    registry = FeatureRegistry()
    registry.register(RatingFeatureExtractor())
    registry.register(EloExpectationFeatureExtractor())
    registry.register(RoundFeatureExtractor())
    registry.register(FormFeatureExtractor())
    return registry


__all__ = [
    "FeatureExtractor",
    "FeatureRegistry",
    "derive_label",
    "EloExpectationFeatureExtractor",
    "RatingFeatureExtractor",
    "RoundFeatureExtractor",
    "FormFeatureExtractor",
    "default_registry",
]
