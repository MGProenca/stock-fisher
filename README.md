# Predicting Titled Tuesday game outcomes

Predict the outcome of a Chess.com **Titled Tuesday** blitz game — **win / draw /
loss from White's perspective** — using only information available **before the
first move**.

Data comes from five Titled Tuesday events — one per month, Feb–Jun 2026 (the
brief suggests two; five gives more data and a more robust temporal split) —
traversed from the public [Published-Data API](https://www.chess.com/news/view/published-data-api).

- **Code** lives in `src/stock_fisher/` (a small, installable package).
- **The writeup** — split rationale, model quality, and next steps — is in
  [`notebook.ipynb`](notebook.ipynb) and summarized at the bottom of this file.

## Quickstart

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                    # install into .venv

# The processed dataset is committed, so you can train immediately (seconds, offline):
uv run chess-train         # trains all models, prints a comparison table, logs to MLflow

# Optional: rebuild the dataset from the API (uses the on-disk cache; ~seconds)
uv run chess-dataset -v

# Optional: read the writeup notebook end-to-end
uv run jupyter lab notebook.ipynb

# Run the tests (label derivation + data-leakage guards)
uv run pytest
```

Everything runs on a fresh clone with **no external services**: MLflow logs to a
local SQLite file (`data/mlflow.db`) by default. To browse runs:
`uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db`.

## How it's built

Three layers, separated so each can be understood and changed in isolation:

| Layer | Package | Responsibility | Network? |
|---|---|---|---|
| **Ingestion** | `stock_fisher.ingestion` | Fetch + cache game JSON; traverse tournament → round → group → `GameRecord`. | Yes |
| **Features** | `stock_fisher.features` | Derive the label and compute pre-game features. | No |
| **Modeling** | `stock_fisher.modeling` | Load the dataset, split, train, evaluate, track. | No |

`dataset.py` orchestrates ingestion + features into a one-row-per-game DataFrame.

```
src/stock_fisher/
├── config.py           # events to mine + ingestion knobs
├── dataset.py          # filter → label → run extractors → DataFrame
├── cli/                # entrypoints: dataset.py (chess-dataset), train.py (chess-train)
├── ingestion/          # client, cache, tournament traversal
├── features/           # labels + tier0 (elo) / tier1 (ratings) / tier2 (form)
└── modeling/           # data/splits, models, evaluation, train (MLflow)
```

### Features (all strictly pre-game)

The API returns many fields per game; most describe *how the game went* and are
**not** admissible. Features are produced only by registered extractors, each
declaring its columns — so the admissible set is an explicit, auditable allow-list.

- **Ratings** — `white_rating`, `black_rating`, `rating_diff`, `abs_rating_diff`,
  `mean_rating` (the players' blitz ratings *at game time*, the primary signal).
- **Elo expectation** — `elo_expected_white = 1 / (1 + 10^(-(W−B)/400))`, the
  single-feature "bar to beat".
- **Round index** — `round_number` (fatigue / late-event proxy).
- **In-tournament form** — per-player points, streak, games played and recent
  win-rate, reconstructed from **earlier rounds of the same event only**
  (rounds 1..r−1). This block is the one that *could* leak, so it is built by
  snapshotting each player's state **before** any round-`r` result is applied.

Deliberately **excluded** as leaky: end-of-tournament standings, final results,
PGN/accuracy, and anything derived from the game being predicted.

## Tests

`uv run pytest` — 26 fast, offline tests (synthetic games; no network or committed
data). They target what can be *silently* wrong rather than coverage for its own
sake, with **data leakage as the priority**:

- **Leakage guards** (`tests/test_leakage.py`):
  - in-tournament form never sees the current round's result — it is snapshotted
    *before* the round is applied, and only shows up from the next round;
  - the dataset's feature columns are exactly the declared pre-game allow-list, so
    no result/outcome/label field leaks in as a model input;
  - the temporal split holds out only the latest event with no event/row overlap,
    and the pooled split is a clean partition.
- **Label derivation** (`tests/test_labels.py`) — result-code → outcome mapping;
  dropping contradictory / missing-result rows.
- **Feature math** (`tests/test_features.py`) — Elo expectation + rating features.
- **Assembly filters** (`tests/test_dataset.py`) — only rated/blitz/standard/
  labelable games survive, with every drop reason counted.

## Writeup

### How I split the data, and why

Two complementary splits (both in `modeling/data.py`):

1. **Pooled stratified 75/25** — all five events pooled, split at random
   (stratified on the label). Measures in-distribution performance.
2. **Temporal holdout** — train on the four earlier events (Feb–May), test on the
   latest (Jun 9). This mirrors real use — predicting *upcoming* games from *past*
   ones — and is the **honest** generalization number. The pooled split leaks a
   little optimism because games from the same tournament share context (same
   field, same day).

### Is it a good model?

Headline numbers on ~9.8k games (accuracy / log loss; lower log loss is better):

| model | pooled acc | pooled log loss | temporal acc | temporal log loss |
|---|---|---|---|---|
| `prior` (floor) | 0.49 | 0.92 | 0.49 | 0.93 |
| `elo` | 0.68 | 0.77 | 0.67 | 0.78 |
| `logistic` | 0.68 | 0.76 | 0.67 | 0.77 |
| `xgboost` | **0.71** | **0.74** | **0.70** | **0.75** |
| `logistic_balanced` | 0.63 | 0.93 | 0.62 | 0.93 |

My read:

- **Rating is most of the story.** A logistic regression on the single Elo
  expectation already hits ~68% / 0.77 log loss; adding the other rating and form
  features barely moves it. That makes sense — a player's live blitz rating
  already integrates their form, and Elo is a purpose-built outcome model.
- **The non-linear model adds a small, *consistent* lift.** XGBoost is best on
  **both** splits (log loss ~0.74–0.75 vs the linear models' ~0.77), so there is
  some real non-linear structure (rating interactions, draw-prone regions). The
  margin is modest, though — a few points of log loss over a one-feature Elo model.
  *Aside:* with only two events the same XGBoost **overfit** and lost to logistic
  on the temporal split; with five it generalizes. A clean reminder that model
  complexity should scale with data — and why the honest temporal split matters.
- **Temporal ≈ pooled.** Every model degrades only slightly from pooled to temporal
  (~0.01–0.02 log loss), so the model generalizes across months — no meaningful
  distribution shift between events.
- **Draws are the hard part.** At ~8.5% prevalence, and with ratings barely
  separating them, the calibrated models essentially never predict `draw`
  (draw recall ≈ 0). `logistic_balanced` recovers ~20% draw recall but pays for
  it with lower accuracy and much worse calibration — a genuine trade-off, not a
  free win. Draws between titled players are close to irreducibly uncertain given
  pre-game info alone.

**Verdict:** a solid, well-calibrated model that clearly beats the naive floors and
improves modestly on a principled Elo baseline. XGBoost is marginally best; given
how small the gap is, shipping the interpretable `logistic`/`elo` model is equally
defensible, with XGBoost reserved for when that last bit of log loss matters. It is
good at what's knowable (rating mismatches) and appropriately humble about what
isn't (draws, near-equal pairings) — not a model I'd expect to push much further
without *new information*.

### What I'd do next

- **Player-history features** (biggest expected lift): each player's own game
  archives before the event — opening repertoire, decisive-vs-draw tendency,
  time-management style, recent form, head-to-head record — with a strict
  `end_time < event_start` cutoff to stay leak-free. (Prototyped and deliberately
  cut here to respect the time box; it's a clean extension of the extractor
  pattern.)
- **Model the draw explicitly** — a two-stage decisive-vs-draw then win/loss
  model, or a draw-aware loss — rather than relying on class weights.
- **Calibration + tuning** — isotonic/Platt calibration and light hyperparameter
  search, judged on the *temporal* split so we don't reward overfitting.
- **Scale the data further** — the pipeline takes an arbitrary event list (five
  here); adding more events would let the temporal split hold out several recent
  events instead of one, tightening the generalization estimate.
- **Grow the test suite** — the leakage/correctness guards in `tests/` (see
  [Tests](#tests)) would extend naturally to any new feature tiers.
