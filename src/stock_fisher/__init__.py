"""stock_fisher — mine Chess.com Titled Tuesday data and build a pre-game,
one-row-per-game dataset for predicting outcomes from White's perspective.

Two clearly separated responsibilities:
  - `stock_fisher.ingestion` — fetch raw data from the API (network, caching).
  - `stock_fisher.features`   — derive labels and compute features (no network).

`stock_fisher.dataset` orchestrates the two into a DataFrame. No modelling lives
in this package yet — it stops at the assembled dataset.
"""

__version__ = "0.1.0"
