# NFL Prediction Engine

A runnable, research-first NFL pregame forecasting package. Milestone 1 imports public nflverse schedules, constructs strictly lagged team features, fits logistic regression, and evaluates it chronologically. It saves auditable forecasts, models, reliability plots, and measured metrics. No market adapter, order placement, real-money trading, or simulated profit is implemented.

## Install and test

Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/) are supported. Dependencies are pinned in `pyproject.toml`; `uv.lock` also locks transitive versions and artifact hashes.

```bash
uv sync --frozen --extra dev
uv run --frozen --extra dev pytest
uv run --frozen --extra dev ruff check .
uv run --frozen --extra dev ruff format --check .
```

Alternatively, create a virtual environment and run `python -m pip install -e '.[dev]'`, then `pytest`. The uv workflow is the fully locked installation. CI runs the locked checks on Python 3.11 and 3.12. Automated tests use local synthetic schedules and mocked HTTP, so they require no external data or credentials.

## Fetch and backtest

```bash
uv run --frozen nfl-predict fetch
uv run --frozen nfl-predict backtest --through-season 2025 --as-of 2026-10-26T00:00:00Z
```

The fetch command prints its immutable CSV path and metadata. By default, backtesting reuses the verified cache and selects the latest complete regular season as the final holdout. An incomplete requested season fails explicitly. `--as-of` limits which outcomes are available; a cached snapshot's fetch time further limits that cutoff, so waiting cannot turn a potentially live cached score into a final result. A timestamp without a timezone is interpreted as UTC.

To refresh the current source, use `fetch --refresh`. To rerun entirely offline, use the exact CSV path printed by `fetch`:

```bash
uv run --frozen nfl-predict backtest --input data/raw/5b8c66203d43c8faa93b0519b3ca078970aa0bdb39ccb51076dc99e5248f1486.csv --through-season 2025 --as-of 2026-10-26T00:00:00Z
```

A local CSV must have nflverse columns `game_id, season, game_type, week, gameday, gametime, home_team, away_team, home_score, away_score, location`. A matching provider metadata file preserves source provenance. A standalone CSV without metadata is explicitly recorded as user-supplied with unknown source/fetch time. Snapshots are checksum-verified. Missing columns, duplicate game IDs, unknown teams, partial scores, and corrupt caches fail with actionable errors. Network timeouts, HTTP 429, and server errors receive bounded retries; TLS verification remains enabled.

Options include `--window` (default 8), `--min-train-seasons` (5), `--bootstrap-samples` (1000), `--seed` (42), `--cache-dir`, and `--output-dir`. No tuning is performed automatically.

## Information cutoffs and evaluation

- Kickoff timestamps are converted from US Eastern time to UTC. Because this source lacks historical final-whistle/publication timestamps, scores become eligible **strictly after kickoff plus 24 hours**. Missing kickoff times use the earliest possible feature cutoff and a conservative end-of-day-plus-24-hours result availability.
- Each team's last eight eligible completed games produce win fraction, scoring margin, points scored/allowed, rest days (capped at 30), and history count. Features are home minus away, plus a neutral-venue-aware home indicator. Franchise aliases preserve relocation histories. Histories span seasons and can include prior postseason games. Empty history uses documented neutral defaults in `features.py`.
- Current outcomes, future results, closing odds, injuries, and postgame statistics are never model inputs. Every forecast records the kickoff feature cutoff and the latest historical-result availability actually used.
- Before each evaluation season, a scaler and fixed `C=1` logistic model are fitted on older regular-season games whose outcomes were available before that season's first kickoff. Both preprocessing and model fitting exclude the test season. The first five seasons are warm-up training; subsequent years use expanding training windows.
- The latest complete selected season is reported separately and never supplies fit, tuning, or recalibration targets. Earlier results in that season may enter later games' pregame form after their availability cutoff, as they would prospectively. Model coefficients remain frozen for the whole season.
- Ties contribute half a historical win but are excluded from binary evaluation targets. Probabilities and calibration therefore concern **decisive games**, not a three-way win/draw/loss market.

Accuracy, Brier score, log loss, ten-bin reliability tables, expected calibration error (ECE), and a reliability SVG are reported. A constant training-set home-win-rate baseline and always-home accuracy provide comparisons. Probabilities are raw logistic outputs: this milestone measures calibration and does not fit a probability calibrator. Any later calibrator must use separate held-out calibration data.

The 95% percentile confidence intervals resample whole season/week blocks with fixed fitted predictions. They reflect sampling variation under that scheme, not model-refit uncertainty, dependence across weeks, source revisions, or future trading performance.

## Measured results

The recorded run evaluated 2004–2024 out of sample and reserved 2025 as the holdout. The 2025 holdout contained 271 decisive regular-season games: **61.62% accuracy**, **0.22198 Brier score**, **0.63231 log loss**, and **0.05258 ECE**. The home-rate baseline achieved 53.87% accuracy. See [the measured report](docs/backtest-2025.md), [machine-readable metrics](docs/backtest-2025.json), and [reliability plot](docs/reliability-2025.svg) for provenance, confidence intervals, calibration bins, and limitations. These are retrospective measurements, not a claim of betting value.

## Audit artifacts

Each execution creates a fresh `reports/<run-id>/` with:

- `predictions.csv`: out-of-sample probabilities, labels, feature values, cutoff timestamps, model IDs, and source hashes.
- `model-<season>.joblib` and `.json`: fitted pipelines and training IDs hash, exact seasons/cutoff, coefficients, feature names, source metadata, code digest, and dependency versions. Only load model files you trust.
- `report.json`, `metrics.md`, and `reliability.svg`: aggregate and per-season metrics, calibration, uncertainty, configuration, and provenance.

`reports/predictions.sqlite` appends each run and its forecasts transactionally with unique `(run_id, game_id)` keys; reruns preserve earlier records. Raw snapshots, models, databases, and execution outputs remain ignored by Git. Only the small measured summary and plot are committed for review.

## Source terms and limitations

Data come from [nflverse/nfldata](https://github.com/nflverse/nfldata), whose README credits Lee Sharpe and the nflfastR/nflscrapR contributors. The adapter snapshots the [games CSV](https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv) with SHA-256, HTTPS source URL, and fetch timestamp. No explicit license file was found in this source repository during implementation; upstream and underlying-source rights require review before redistribution or commercial use. Public access alone does not establish a license. Raw data are not committed.

This is a present-day historical snapshot, not an archive of what the provider published at each historical cutoff. Conservative lagging prevents current/future outcomes entering features, but cannot undo later source corrections or prove historical schedule availability. This limitation is retained in every report. Injuries, market prices, roster changes, strength-of-schedule corrections, and prospective performance are outside this baseline.

Next is Milestone 2: independently validated challengers and held-out probability calibration, then scoring-distribution diagnostics before any simulations. A documented, read-only **Polymarket US** adapter remains disabled until endpoint, fees, settlement, and depth are verified. A Streamlit dashboard and append-only paper-trading ledger are later milestones. Real-money trading requires a separate explicit request and review. Read `AGENTS.md` for the project principles.
