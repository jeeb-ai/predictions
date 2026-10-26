# Sports Prediction Engine

Private, research-first sports forecasting project. NFL first; NBA and MLB after independent validation.

## Objective
Estimate calibrated pregame probabilities, compare them with executable Polymarket US prices after fees, and maintain a timestamped paper-trading ledger. **No automatic real-money trading**.

## MVP requirements
1. Download publicly available historical NFL data (nflverse) using a reproducible pipeline with caching, clear data licenses, and recoverable errors.
2. Build pregame team features with temporal cutoffs; no outcome or future-data leakage.
3. Fit logistic regression baseline and a tree-based challenger. Time-based train/validation/test splits; compare to naive baselines.
4. Report Brier score, log loss, calibration, reliability plots, accuracy and uncertainty. Do not advertise positive EV without prospective evidence.
5. Create simulated score/margin distributions **only after** fitting and checking a scoring model. Record assumptions and sensitivity.
6. Read-only Polymarket US market-data adapter, disabled by default until endpoint documentation, contract settlement rules, and fee schedules are verified. Never assume international markets are fungible with US contracts.
7. Streamlit dashboard: slate, probability intervals, best executable ask, fee-aware expected value, model version, timestamp and clear bet/lean/pass rules.
8. Append-only forecasts and paper trades in SQLite or DuckDB, with settlement and retrospective audit.
9. Automated tests, CI, setup docs and a scheduled data-refresh workflow with documented permissions and rate limits.

## Development stages
- Milestone 1: reproducible NFL feature pipeline + backtest with tests.
- Milestone 2: calibrated probabilities + simulation diagnostics.
- Milestone 3: verified market-data adapter + paper trades.
- Milestone 4: dashboard, monitoring and optional expansion.

## Start with Codex
Open this repo in Codex and give it the instructions in `AGENTS.md`. First implement milestone 1 end-to-end with a CLI and tests. Commit changes on a feature branch and propose a pull request.

## Guardrails
Do not commit secrets, API tokens or personal account details. No live trading, order placement, automated execution, or invented model performance. Keep raw data out of Git history. This software is experimental and cannot guarantee a trading edge.
