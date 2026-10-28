# NFL Prediction Engine

A runnable, research-first NFL pregame forecasting package. Model 1.0 uses lagged game results; Model 2.0 adds causal play-by-play EPA, success/explosive rates, opponent-adjusted strength, and prior-observed quarterback performance. Both fit logistic regression and are compared on identical chronological splits. It saves auditable forecasts, models, reliability plots, and measured metrics. Live pregame inference, a verified read-only Polymarket US adapter, fee-aware EV, and an append-only paper ledger are available. No account access or real-money order placement is implemented.

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

## Model 2.0 comparison

```bash
uv sync --frozen --extra dev
uv run --frozen nfl-predict fetch-pbp --start-season 1999 --through-season 2025
uv run --frozen nfl-predict compare --input data/raw/5b8c66203d43c8faa93b0519b3ca078970aa0bdb39ccb51076dc99e5248f1486.csv --through-season 2025 --as-of 2026-10-26T00:00:00Z --offline
```

Use the schedule path returned by `fetch`. The comparison command can fetch missing snapshots unless `--offline` is supplied. The first PBP fetch downloads roughly 0.5 GB; EP estimators and verified caches are reused on later runs. `--pbp-dir`, `--ep-dir`, `--workers` (default 4), and `--output-dir` control local storage and bounded download concurrency. The fixed default comparison uses 2,000 paired week-block bootstrap draws.

Published nflfastR EPA can incorporate later seasons through its shared EP estimator. To avoid that historical estimator lookahead, Model 2.0 **reconstructs a distinct remaining-half EPA from raw nflverse PBP**, fitting each annual EP model only on earlier available seasons. The QB proxy comes from a prior observed game, never the target game's actual starter. [Methods and the Model 1.0 review](docs/model2-method.md) describe the definitions, cutoffs, data coverage, fixed design, and limitations.

On 5,426 walk-forward validation games, Model 2.0 achieved 65.09% accuracy versus 62.79% for Model 1.0, with favorable paired accuracy/Brier/log-loss intervals. On 271 reused 2025 holdout games, accuracy was 62.36% versus 61.62%, Brier 0.21993 versus 0.22198, log loss 0.62695 versus 0.63231, and ECE 0.05938 versus 0.05258. **All 2025 paired intervals include zero: holdout improvement is inconclusive.** Calibration is not established by the favorable loss estimates.

The 2025 results were already examined for Model 1.0; this preserves their exclusion from Model 2.0 fitting/tuning but cannot make the season unseen again. See the [actual comparison](docs/model2-comparison.md), [machine-readable results](docs/model2-comparison.json), and [paired reliability plots](docs/model2-reliability.svg). New run directories retain full per-model forecasts, trained win/EP model provenance, comparison reports, coverage, and paired predictions. Raw snapshots and binaries remain excluded from Git.

## Live pregame forecasts and paper trading

```bash
uv run --frozen nfl-predict forecast --game-id 2026_08_CAR_GB > /tmp/nfl-forecasts.json
uv run --frozen nfl-predict us-markets --limit 20
uv run --frozen nfl-predict quote --market aec-nfl-car-gb-2026-10-29 --outcome no --output reports/quotes/car-gb-no.json
uv run --frozen nfl-predict ev --forecast /tmp/nfl-forecasts.json --quote reports/quotes/car-gb-no.json --team GB --quantity 10 --output reports/ev/car-gb.json
uv run --frozen nfl-predict paper-audit
```

These dates and IDs identify the live smoke-check game; select current games and market slugs when running later. `forecast` refreshes public schedules and current-season PBP by default, caps information at its actual forecast time, and trains an immutable annual Model 2.0 bundle from older seasons only. `--model-version v1` selects the historical baseline family; `--bundle` reuses a trusted live bundle; `--offline --input PATH` explicitly uses cached inputs. Historical `backtest`/`compare` behavior and archived results remain unchanged.

Quotes use the documented public **Polymarket US** gateway, real ask depth, exact team/kickoff matching, timestamps, and immutable raw snapshots. No credentials are needed. NFL ties pay $0.50 per contract and cancellations may use last fair market price. Therefore the EV command above intentionally reports a pass until explicit tie and game-completion probability assumptions are supplied; the model does not estimate either. Any cancellation probability also needs a cancellation-price assumption. Add `--tie-probability`, `--completion-probability`, and, when needed, `--cancellation-settlement-price` using documented assumptions rather than implied defaults.

US taker fees use the verified coefficient 0.0695 and half-even cent rounding. EV consumes displayed price levels and deducts a conservative cumulative fee upper bound because aggregate depth hides individual maker-fill rounding. Stale/mismatched quotes, started games, unknown fees/terms, missing assumptions, insufficient depth, and EV below the chosen threshold cannot become paper candidates. Raw model probabilities remain uncalibrated estimates; a positive estimated EV does not establish a trading edge.

`forecast`, `quote`, and `ev` append their inputs/results to `data/paper/ledger.sqlite`. In this standalone workflow, an explicit `paper-fill --evaluation-id EV_ID --idempotency-key UNIQUE_KEY` creates a local simulated position. Optional `--max-exposure` caps outstanding paper cost. Paper fills never call an exchange and cannot exceed the retained book's available depth, including repeated polls of the same book. `market-settlement` reads final exchange settlement; `paper-settle` requires a price and evidence file. SQL append-only triggers, foreign keys, idempotency, retained evidence, and a hash chain support audit. Keep the audit head hash separately to detect complete-file rewriting or truncation.

See the [live/paper workflow](docs/live-paper-workflow.md) for exact commands, settlement evidence, freshness policies, and limitations, and the [verified US API references](docs/polymarket-us-reference.md). Production smoke checks and automated fixture outcomes are documented separately; no fabricated real trade or settled profit is reported.

## Scheduled research and private dashboard

```bash
uv run --frozen nfl-predict research-init --config research.local.json
docker compose -f deploy/compose.yml build
docker compose -f deploy/compose.yml up -d --no-build
docker compose -f deploy/compose.yml exec -T worker nfl-predict research-status --config /app/research.local.json
```

The self-hosted Linux worker refreshes pregame forecasts and Polymarket US quotes every 15 minutes, retains auditable run history, and automatically tracks the first issued forecast per game against subsequently observed outcomes. A private, authenticated Streamlit dashboard displays uncertainty, pass reasons, paper positions, calibration, and worker health. Docker publishes it only on the host's loopback interface; access it through SSH or a private network. The initial historical-data and model warmup can take 15–30 minutes or longer. Persistent volumes retain evidence across restarts.

Training-week bootstrap intervals describe model-estimation uncertainty conditional on features, not calibrated outcome coverage. Scheduled paper entries require fee/depth/freshness gates and worst-case EV above the threshold across the model interval and explicit risk ranges. Tie, completion, and cancellation assumptions default to missing, so the initial policy records passes and creates no positions. Prospective metrics remain unavailable until actual scored games exist; repeated polls cannot inflate the sample. Existing historical results and standalone point-forecast defaults are preserved.

See [deployment, private access, policy, and tracking instructions](docs/research-system.md) and [measured system validation](docs/research-system-validation.json). The container key lives in its private data volume, separately from a host development key. The dashboard reads reports through read-only mounts and cannot place trades or modify the ledger. These commands prepare a persistent-host deployment; a temporary cloud workspace does not supply always-on hosting.

## Source terms and limitations

Data come from [nflverse/nfldata](https://github.com/nflverse/nfldata), whose README credits Lee Sharpe and the nflfastR/nflscrapR contributors. The adapter snapshots the [games CSV](https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv) with SHA-256, HTTPS source URL, and fetch timestamp. No explicit license file was found in this source repository during implementation; upstream and underlying-source rights require review before redistribution or commercial use. Public access alone does not establish a license. Raw data are not committed.

The play-by-play source is [nflverse/nflverse-data](https://github.com/nflverse/nflverse-data); its [CC BY 4.0 license](https://github.com/nflverse/nflverse-data/blob/main/LICENSE.md) and attribution are recorded in each snapshot metadata file. This is distinct from the schedule source terms above.

This is a present-day historical snapshot, not an archive of what the provider published at each historical cutoff. Conservative lagging prevents current/future outcomes entering features, but cannot undo later source corrections or prove historical schedule availability. This limitation is retained in every report. Injuries, market prices, and confirmed upcoming starters remain outside the prediction features. Model 2.0 adds online opponent adjustment without using season-end ratings. Prospective tracking starts with actual recorded pregame forecasts; it does not turn retrospective results into prospective evidence.

Next is accumulating independent prospective evidence and studying held-out probability calibration, followed by scoring-distribution diagnostics before any scoring simulations. The public Polymarket US adapter remains read-only and paper positions remain local. No real-money trading is implemented. Read `AGENTS.md` for the project principles.
