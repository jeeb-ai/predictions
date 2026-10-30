# V5 Phase 2 — independent NFL scoring research

This is the second, separate PR for [Issue #6](https://github.com/jeeb-ai/predictions/issues/6), stacked on Phase 1. It preserves the existing binary winner engine. It adds an independent, historically tested scoring engine, immutable pregame score distributions, exact-line spread/total scenarios and an optional dashboard projection. It does **not** establish a validated trading edge or complete the NBA/NHL phase.

## Models and chronology

All inputs are strictly lagged schedule results: rolling scoring/allowance, margins, win fractions, history counts, rest and home field. No binary win probability, market price, closing line, target-game result, injury or actual target starter is a model input. Source is the existing SHA-verified nflverse schedule provider, with its conservative kickoff-plus-24-hour result availability and retrospective-vintage caveat. Full-game scores include overtime. Only regular-season scoring is evaluated and enabled; postseason scoring is unavailable.

Three predeclared approaches are compared:

1. Constant historical mean, independent Poisson score baseline.
2. Training-standardized per-team Ridge regression (`alpha=10`), with bias and joint residual covariance estimated on a later, separate calibration season. A nonnegative, integer-grid truncated joint Gaussian distribution is normalized explicitly.
3. Training-standardized per-team Poisson regression (`alpha=0.1`), with a joint four-regime negative-binomial mixture fitted only on the calibration season. Mixture regimes describe signs of paired calibration residuals; their empirical weights, score multipliers and component dispersion capture dependence and overdispersion. This is a count approximation, not a possession/drive simulator.

For test season Y, coefficients/scaling use seasons before Y−1. Season Y−1 calibrates the distribution; its outcomes must be available before the first test kickoff. Earlier regular/postseason results can enter a later game's lagged form once available, as in the winner engine. Fits never consume that evaluation season's targets. The first six seasons supply five training seasons and a separate calibration season. Evaluation spans 2005–2024 (5,183 games) and reports 2025 separately (272 games, including one tie). Lowest validation joint log loss selects the family before the 2025 scoring results are evaluated. The 2025 season was already examined by earlier winner-model work; it cannot be called a truly unseen dataset. Future timestamped forecasts supply a prospective cohort.

The score grid is 0–150 per team. Joint probability mass is normalized; count distributions fail on >1% truncated mass. The Gaussian is deliberately conditioned on nonnegative scores. This does not model football's exact scoring lattice. Means, medians, intervals, expected margin and expected total are all derived from the same joint distribution. Expected scores are averages, not a promised exact legal final-score pair. Predictive intervals omit parameter/model-selection uncertainty.

## Actual out-of-sample results

See [full machine-readable validation](v5-scoring-validation.json), with all models, annual metrics, folds, ten-bin reliability, integer/half-point push diagnostics and paired uncertainty. Selected Ridge joint Gaussian:

| Metric | 2005–2024 validation | 2025 evaluation |
|---|---:|---:|
| Joint score log loss (lower is better) | 7.3629 | 7.3121 |
| Per-team score MAE | 7.7415 | 7.4909 |
| Per-team score RMSE | 9.7031 | 9.4026 |
| Margin MAE | 10.6874 | 10.3365 |
| Total MAE | 10.8505 | 10.6611 |
| Per-team CRPS | 5.4716 | 5.3270 |
| 90% marginal score interval coverage | 89.77% | 89.15% |

2025 MAE's week-block 95% interval is [7.0116, 7.9394]; coverage's interval is [87.03%, 91.15%]. These are 500 fixed-prediction season/week block draws, seed42, not model-refit or selection-aware intervals. Paired 2025 joint-log-loss difference versus the Poisson baseline is −2.2778 [−2.6970, −1.8430]. Against the joint NB mixture it is −0.1834 [−0.2925, −0.0790]. The mixture has slightly lower point-score MAE (7.4820 versus 7.4909), but that paired difference is inconclusive. Selection is based on validation distribution loss, not a claim that Ridge wins every metric. Neither score model is promoted over the existing winner classifier.

For the scoring model, 2025 decisive-game winner accuracy is 61.62%, Brier 0.22204 and log loss0.63254. These are **separate from the preserved V4 winner probabilities**. The scoring model's 2025 margin/total CRPS are 7.4072/7.5939. Spread/total Brier scores in the report are **three-outcome win/push/loss sums**; at half-point lines they equal twice the usual binary Brier. Calibration checks use predeclared home −3/−3.5 and totals44/44.5 scenarios, not historical sportsbook lines. They establish no profitability.

### A material failed calibration check

At home −3, the selected model predicted a 2.79% push rate versus 8.46% observed in 2025. At total44, predicted pushes were 3.00% versus 4.41% observed. Generic score distributions miss NFL key numbers. The UI warns on integer lines; all spread/total market-value decisions remain **PASS**. Arbitrary user-entered lines are unvalidated scenarios. Broad score-interval coverage near90% does not repair this discrete-outcome miss or establish calibrated betting probabilities. Further NFL key-number calibration and prospective validation remain necessary before calling these edge estimates reliable. No such edge is claimed.

## Run locally with Python 3.12 and uv

After merging Phase 1 then this PR, update/sync as in the Phase 1 guide. Review this branch directly with `git switch feat/v5-nfl-scoring`. From the checkout:

```bash
uv sync --frozen --extra dev --python 3.12
uv run --frozen nfl-predict score-backtest \
  --input data/raw/YOUR_VERIFIED_SCHEDULE.csv \
  --as-of 2026-10-26T00:00:00Z --through-season 2025 \
  --output-dir reports/scoring > /tmp/scoring-report.json
uv run --frozen nfl-predict score-forecast \
  --input data/raw/YOUR_VERIFIED_SCHEDULE.csv \
  --validation /tmp/scoring-report.json --game-id CURRENT_GAME_ID \
  --home-spread -3.5 --total-line 44.5
```

Use `nfl-predict fetch --refresh` for a current schedule; retain the returned path and provenance. Do not use a past snapshot while describing it as fresh. Lines above are user-specified scenarios, not invented market observations. Each scoring forecast retains a model checksum, joint PMF checksum, source/cutoff timestamps, feature values and older training/calibration periods. Historical runs retain all per-fold models, forecasts and reports. Only load trusted local model artifacts.

For a full winner+score forecast and dashboard:

```bash
uv run --frozen nfl-predict forecast --game-id CURRENT_GAME_ID \
  --include-analysis --bootstrap-samples 100 \
  --score-validation /tmp/scoring-report.json
```

To opt the existing scheduler into scoring, copy the **completed evaluation report** to `scoring-approved.json` inside your configured `paths.reports_dir` (default `reports/research/scoring-approved.json`). Then restart the worker and run a fresh cycle. For Docker, install that report into `/app/reports/research/scoring-approved.json` in the persistent reports volume, preserving nonroot-readable permissions. A status-only stub is rejected; model selection must agree with validation loss and the report must contain coherent folds and nonempty holdout metrics. This is trusted operator configuration, not an authenticity signature. An invalid/missing scoring artifact leaves the independent winner prediction available. No archived forecast or ledger entry is rewritten.

Run `uv run --frozen nfl-predict dashboard --config research.local.json`; keep its existing private key and localhost binding. New cards show average projected points, expected margin/total and wide score intervals. Matchup Analysis offers exact-line scenarios from the retained, checksum-verified PMF. The existing Polymarket US moneyline adapter/EV remain separate. No verified executable spread/total quote adapter is supplied, so no spread/total price, overround, fee or settlement assumption is invented. Market Value explicitly separates a raw probability-minus-price difference from fee-aware conditional EV.

## Validation evidence and next work

[Authenticated UI evidence](v5-scoring-ui-evidence.json) comes from a newly generated pregame public-data winner+score forecast. Tests cover fit/calibration chronology, current-score leakage, normalized joint mass, coherent expectations, integer/half-point pushes, exact lines, corruption checks, postgame rejection, retained winner behavior and rendering. The original winner backtest comparison was also rerun unchanged in Phase 1.

NBA and NHL are not implemented by this PR. Read-only source probes found the NBA official gamefinder, NHL official schedule and MoneyPuck historical team/shot data accessible. Access does not establish redistribution rights or historical model vintages. NBA needs its own possession/pace, efficiency, rest/back-to-back and availability model; NHL needs causal xG/shot-quality provenance, goalie proxies, special teams and distinct regulation/OT/shootout settlement modeling. In particular, retrospectively published xG is not automatically leakage-free. Those expansions must not reuse NFL assumptions or be represented by placeholder forecasts. The immediate statistical priority remains the NFL discrete calibration miss above; no MLB or capper tracking is added.
