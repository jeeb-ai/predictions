# Codex project instructions

You are developing an auditable sports forecasting system, initially for NFL pregame moneyline probabilities.

## Principles
- Write working code and tests, not mock performance numbers.
- Record provenance, timestamps, model versions, and pregame information cutoffs.
- Prevent future-data leakage explicitly: for game G, features must be derived only from games and injury/market observations known strictly before G.
- Use chronological walk-forward backtests; reserve the latest season as an untouched holdout whenever data permit.
- Start with simple transparent baselines. Calibrate probabilities only on held-out calibration data.
- Report actual metrics and uncertainty; do not optimize or market for guaranteed profit.
- Before simulations, fit a scoring distribution to historical observed outcomes and check calibration. Vary assumptions in sensitivity testing.
- Isolate providers behind adapters, implement retry/rate-limit handling and snapshot raw responses.
- Market data must be for Polymarket **US**, on a documented read-only endpoint. Account for ask price, depth, fees, settlement terms and timestamps.
- Default to paper-trading only. Never implement or trigger real-money trades unless separately and explicitly requested for development, reviewed for security and legal constraints.
- Store secrets in environment variables; add .env.example and .gitignore.
- Use Python 3.11+, uv or pip, pandas, numpy, scikit-learn, pytest, DuckDB/SQLite, and Streamlit.
- Add reproducible commands in README; pin dependency versions.
- Check quality with pytest and linting. Add CI when possible.

## First task
Implement Milestone 1: a small, runnable Python package that imports nflverse historical game/schedule data, builds strictly lagged pregame features, trains baseline win probability, runs walk-forward evaluation, saves versioned predictions, and outputs a concise metrics report. Include synthetic-data tests that catch leakage and team-side inversion. If a live data source cannot be accessed in the development environment, provide a clean fetch command and test against local fixtures; never fabricate production results.

## Output at end of each task
Summarize files changed, commands executed, test results, known limitations and the next milestone. Prefer a PR rather than pushing unreviewed changes to main.
