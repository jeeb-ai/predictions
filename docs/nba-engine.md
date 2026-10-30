# NBA v1 — independent basketball predictions

PR #7 is merged on main. PR #8 was merged into the former Phase1 branch, not main; integration PR #9 restores that existing scoring merge intact. The NBA PR targets main and depends on #9, so the merged NFL winner and scoring engines are both preserved. It does not modify their estimators or archived comparisons.

## Data and information cutoffs

Historical regular-season team box scores come from NBA's public `stats.nba.com/stats/leaguegamefinder`, with `PlayerOrTeam=T`, a specified season and Regular Season. Official upcoming fixtures come from `cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json`. Every response is retained with canonical URL, retrieval time and SHA256; checksummed caches are reused and current-season data refreshed. TLS verification stays enabled, with bounded timeout/retry. Public access is not a redistribution/commercial license: NBA terms and underlying rights need review before public/commercial deployment. Raw data are not committed.

A season is labeled by its start year: 2025 means 2025–26. Full-game scores include overtime; final NBA ties are impossible and removed from predictive support. Historical box scores have dates rather than reliable historical tipoff timestamps. Features therefore use the earliest UTC time on the recorded day, with results eligible **strictly after that day plus48hours**. Source fetch time also limits eligibility: a cached response fetched before availability stays masked, even if used later. This is conservative and cannot prove original historical publication vintages or undo later corrections.

Possessions are estimated as the average of both teams' `FGA−OREB+TOV+0.44×FTA`. Offensive and defensive ratings are points scored/allowed per100estimated possessions; pace scales possessions to48minutes using team minutes rounded to regulation/overtime periods. These are estimates, not play-by-play possession counts. The last10eligible regular-season games supply efficiency, pace and form. Rest/back-to-backs use **known schedule dates**, including a previous scheduled game whose result is not yet eligible; NBA dates are aligned on the league's Eastern calendar to avoid Pacific-time errors. Offseasons cap rest at14days. Online Elo (1500initial,K20,nominal-home60points) updates only on eligible earlier outcomes. Home/away is the provider's nominal assignment; neutral/international venue effects are not separately fitted. Some neutral games have the same matchup string for both teams, so team codes are cross-checked rather than assuming one `vs.` row always exists.

No target score, target box score, market price, injury rumor or actual future lineup is a model input. Verified historical player availability is **not supplied by this feed**; it is unknown and not modeled. Do not interpret that as everyone being healthy. Future availability integration needs timestamped historical archives before it can enter training honestly.

## Models and actual evaluation

Each annual fold fits a standardized C1 logistic winner model on seasons before the immediately preceding season. The preceding season fits a one-dimensional Platt calibration, with no test labels/scaling leakage. The same pregame basketball inputs feed separate per-team Ridge (`alpha10`) score regressions. A joint nonnegative integer Gaussian score distribution uses residual bias/covariance from the older calibration season and excludes tied final scores. It yields coherent expected scores, margin/total, broad predictive intervals and exact integer/half-point cover/over/push probabilities. Scores are not made from the binary win probability. The two models can disagree. No scoring simulations are claimed.

After five training seasons and one calibration season, validation covers 2016–2024 (10,749 games); the separately reserved 2025–26 regular season has1,230games. [Actual results](nba-validation.json) include chronological folds, source hashes, proper scores, ten-bin reliability and500fixed-prediction week-block intervals (seed42). No hyperparameter was tuned on that held season. The winner baseline is the older training home-win rate; the score baseline uses the calibration season's constant means/covariance, not NFL constants.

| NBA model metric | Validation | 2025–26 holdout |
|---|---:|---:|
| Winner accuracy | 64.77% | 68.21% |
| Brier | 0.21768 | 0.20856 |
| Log loss | 0.62479 | 0.60529 |
| Calibration error (10bins) | 0.00824 | 0.03765 |
| Per-team score MAE | 9.2167 | 9.5653 |
| Per-team score RMSE | 11.6016 | 12.0394 |
| 90% score-interval coverage | 90.72% | 89.96% |

Measured winner calibration is imperfect; these numbers do not establish an edge. Week-block intervals omit parameter refitting and dependence across weeks. Spread/total reliability checks use fixed −5.5/220.5 research scenarios, **not historical market lines**. Other lines are model-derived scenarios. Three-outcome Brier includes win/push/loss and differs from binary Brier. There is no verified executable NBA price/fee/settlement adapter; **market value remains PASS**. Historical roster changes, rest approximations, venue ambiguity and source revisions limit reliability.

## Mac Python3.12 / uv commands

Until merge, fetch and switch to `feat/nba-engine`; after #9 and the NBA PR merge, use main. From your checkout:

```bash
git fetch origin
git switch feat/nba-engine
uv sync --frozen --extra dev --python 3.12
uv run --frozen nfl-predict research-init --config research.local.json
uv run --frozen nfl-predict nba-backtest --start-season 2010 --through-season 2025 > /tmp/nba-report.json
uv run --frozen nfl-predict nba-run --start-season 2010 --through-season 2026 --validation /tmp/nba-report.json --config research.local.json
uv run --frozen nfl-predict dashboard --config research.local.json
```

For October2026 through spring2027 the current season's start year is2026. Use your existing private key to unlock; initialization preserves it. Do not share/commit the key. Run the NFL research worker separately as before. NBA predictions are discovered independently of markets and written to a separate immutable run directory plus `nba-predictions.json` in the configured research state. A failed sport refresh leaves its last predictions retained; stale predictions are labeled. Corrupt NBA snapshots do not hide valid NFL forecasts. All snapshots are read only after authentication, and the dashboard makes no provider calls.

To refresh NBA automatically in a separate terminal, install your completed report under the configured reports directory (default below), then run:

```bash
mkdir -p reports/research
cp /tmp/nba-report.json reports/research/nba-approved.json
uv run --frozen nfl-predict leagues-worker --sport NBA --config research.local.json
```

The worker uses process locking, per-sport error/status files and bounded interruptible waits. It archives timestamped forecasts and checksummed model/score artifacts. New league archives are separate from the existing NFL paper ledger; no real-money execution or new paper fill mechanism is added. Prospective NBA grading is not yet automatic; retain the immutable predictions for later verification. [Authenticated UI evidence](nba-ui-evidence.json) verifies actual official-source future matchup cards, score projections and sport filtering in the shared mobile-responsive Streamlit layout. Detailed coefficients stay under Advanced Analysis; supporting/opposing templates reconcile to the calibrated winner log-odds and describe associations, not causality.
