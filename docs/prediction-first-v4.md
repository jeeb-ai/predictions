# Version 4.0: prediction-first NFL analysis

Open the private dashboard as documented in [the Linux deployment guide](research-system.md). Predictions is the default area. The current slate comes from nflverse, using the worker's configured lookahead (seven days by default), regardless of Polymarket coverage. Each card names the model's predicted winner, both decisive-result win probabilities, probability advantage in percentage points, bootstrap uncertainty, kickoff, model version and update/cutoff times. Equal probabilities mean an even matchup. This product release preserves the existing logistic-regression-v1/v2 model families; it does not imply new predictive accuracy.

Select a game in Matchup Analysis to inspect offense against opposing defense, causal remaining-half EPA, success/explosive rates, rolling form, online opponent-adjusted Elo, observed QB proxy form, rest and home field. These comparisons are descriptive matchup factors. EPA is reconstructed using older-season EP estimators; **EPA is not itself opponent-adjusted**. Elo supplies the opponent adjustment. Defense EPA and rates measure allowed production, so lower is generally better. Recent form uses the last eight eligible games by default, including earlier seasons.

The explanation uses the actual fitted scaler and logistic regression. For every input, it retains the raw home-minus-away value, training mean/scale, standardized value, coefficient and signed log-odds contribution. The intercept plus all contributions exactly recovers the model's home log-odds, whose logistic transform recovers the displayed probability. Positive/negative contributions favor home/away relative to the intercept. Correlated inputs, regularization and incomplete information preclude causal claims or treating these as independent effects.

Team-side source values, historical game IDs, availability timestamps, sample counts and provider provenance are retained inside each new timestamped prediction. Missing histories retain documented neutral model defaults and are flagged through zero sample counts. QB IDs describe the most-used passer in the last observed QB game, **not a confirmed upcoming starter**. No injury claims, projected scores or scoring simulations are supplied. Bootstrap bounds remain conditional model-estimation uncertainty, not calibrated outcome coverage.

## Independence and recovery

The worker publishes `data/research/predictions.json` after schedule discovery and again after inference, **before settlement or quote requests**. This independent snapshot includes the discovered slate and verified forecasts. A slow market provider therefore cannot block publication of predictions. The homepage can display this snapshot while the research report is running, partial or failed. Missing forecasts get explicit unavailable cards; no market probability substitutes for them. A failed refresh retains earlier pregame forecasts only when game, teams and kickoff match the current schedule; stale forecasts are visibly labeled. Already-started games leave the current prediction cards. Historical records remain in the existing append-only ledger and immutable report artifacts.

The dashboard remains authenticated and read-only. Market Value is a secondary optional Polymarket **US** comparison of retained executable asks/depth, conservative fees, conditional EV and PASS reasons. It cannot change model inputs or probabilities. Current freshness must be checked before relying on an old evaluation. Performance, Paper Ledger and Audit are inside a collapsed Research / Administration section. Market access remains GET-only and there is no real-money execution.

Standalone forecasts preserve their existing schema unless explanation output is requested:

```bash
uv run --frozen nfl-predict forecast --game-id CURRENT_GAME_ID --include-analysis --bootstrap-samples 100
```

Scheduled forecasts request explanations automatically. Rebuild/recreate the services after upgrading; a fresh worker cycle supplies explanations. Older immutable forecasts cannot acquire trustworthy source values by reverse-engineering feature differences; the UI explicitly labels their analysis unavailable.

## Evidence and limitations

[UI evidence](v4-ui-evidence.json) records an actual authenticated Streamlit AppTest render of a newly generated public-data forecast: the default title/tabs, visible card metrics, source values and complete fitted contributions. This is rendering evidence, not a screenshot, historical performance measurement or claim of future accuracy. Automated tests exercise the complete trained-model → persisted snapshot → authenticated UI path and independently compare the decomposition to the fitted pipeline. Worker tests cover missing markets, quote outages, changed quotes, failed refreshes and schedule revisions.

Historical Model 1.0/2.0 fitting, comparison splits and 2025 results are preserved; no model improvement is claimed. All predictions remain conditional on decisive games and raw probabilities remain uncalibrated. Data are present-day snapshots rather than true historical publication archives. Existing conservative availability cutoffs and source hashes remain in effect.

## Separate scoring-model investigation

[Measured historical diagnostics](scoring-model-investigation.json) use 6,423 completed 1999–2023 regular-season games from the archived verified schedule. Home/away means are 23.14/20.88 points, while variances are 105.87/101.83. That overdispersion undermines a simple Poisson assumption. The measured score correlation is −0.032; that alone does not establish independence or a suitable scoring distribution. These are descriptive observations, not a fitted scoring predictor.

A future release should compare training-only negative-binomial and drive/possession scoring approaches against simple historical baselines. Evaluate expanding chronological folds on held-out log likelihood, score/total/margin errors, discrete probability calibration and interval coverage, including overtime and ties. Learn any calibration and choose model settings only on earlier validation folds. The already examined 2025 season cannot become an unseen holdout again; reserve new prospective outcomes before model selection. Validate joint distributions and sensitivity before implementing simulations or publishing projected scores. This release implements neither.

After this redesign, the sport roadmap is NBA, then NHL, then exploratory ATP/WTA tennis. MLB and social-media capper tracking are outside scope.
