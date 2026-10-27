# Model 2.0: causal play-by-play features and paired evaluation

Model 2.0 retains the Model 1.0 logistic regression and adds play-by-play form, an observed-QB proxy, and opponent-adjusted strength. It is a research candidate, not a validated production upgrade. The prespecified [design](model2-design.json) was saved before computing Model 2.0 validation or holdout metrics. No parameters or features were tuned against the measured results.

## Review of merged Model 1.0

The starting point is merged commit `8b68a705814530fd57cb146cafa65f40d53975fe`. Its team-form construction strictly excludes results unavailable before the target kickoff, fits preprocessing on training games alone, freezes model coefficients before each evaluation season, and reserves 2025 from fitting. The original seven predictors, training IDs, season-opening cutoffs, and fixed `C=1` estimator are preserved. Replaying the original schedule snapshot exactly reproduces its published point metrics. Tests retain those behaviors and reject differences between comparison folds or target sets.

Its limitations remain: decisive-game targets exclude ties, source data are retrospective, publication timestamps are imputed, and historical validation does not establish prospective trading value. Its 2025 results were already observed; this comparison reuses that holdout while excluding it from Model 2.0 fitting, selection, tuning, and calibration. It cannot restore a globally unseen holdout.

## Why published nflfastR EPA is not the default

The [official nflfastR model description](https://github.com/nflverse/open-source-football/blob/master/_posts/2020-09-28-nflfastr-ep-wp-and-cp-models/nflfastr-ep-wp-and-cp-models.Rmd) describes leave-one-season-out calibration across 2000–2019; early years' models can therefore train on later years. The [model training source](https://github.com/nflverse/fastrmodels/blob/master/data-raw/MODELS.R) also fits a shared production EP estimator. Lagging those published EPA rows would leave estimator-training lookahead in early evaluation years.

This implementation reads raw nflverse states and scores, not the published `epa`, `ep`, probability, or win-probability outputs, when reconstructing EPA. A regression test sets published EPA to an absurd constant and verifies that the default comparison replaces it with causal values.

## Chronological EP reconstruction

Our EP definition is **expected possession-team net points remaining until the end of the half**, with overtime treated as one terminal group. It differs from nflfastR's next-score target and must not be presented as identical official nflfastR EPA.

For season Y, a fixed histogram gradient-boosting regressor trains on the previous three supplied seasons. Training games must have completed outcomes available strictly before Y's first scheduled kickoff. Inputs are preplay yard line, down, yards to go, remaining half time, score difference, and both teams' remaining timeouts. The target uses those older games' actual remaining-half score changes. The model is frozen for all plays in Y. The 1999 warm-up season uses a declared zero-EP prior because earlier training data are unavailable.

EPA is the actual signed score change between consecutive valid preplay states, plus the possession-adjusted next state's estimated EP, minus current estimated EP. The terminal potential is zero at half end. This transition can include intervening conversions or special teams; it is not a replica of official single-play next-score EPA. Missing or invalid states receive missing EPA, are counted in coverage, and do not cause win-prediction targets to disappear.

The EP estimator uses 60 iterations, 15 leaves, 64 bins, learning rate 0.1, L2 regularization 10, seed 42, and no early stopping. Fit threads are bounded. Each annual artifact records prior training seasons, exact availability cutoff, latest training-result timestamp, training-ID hash, raw source hashes, source-code digest, parameters, model checksum, and reconstructed EPA checksum.

## Pregame features

All additions are home minus away and retain the original seven predictors:

| Addition | Construction from strictly available history |
| --- | --- |
| Offensive and defensive EPA/play | Total causal EPA divided by eligible plays in the latest eight observed games; defense uses opponents' offensive plays. |
| Offensive and defensive success | Fraction with causal EPA greater than zero, weighted by play counts. |
| Offensive and defensive explosive rate | Rushes of at least 10 yards or dropbacks of at least 20 yards, divided by eligible plays. |
| QB EPA/dropback and success | Last eight available appearances of the previously observed QB, following the player across teams. |
| Opponent-adjusted strength | Prequential Elo, initially 1500, K=20, 400-point scale, 50 home points unless neutral. Both teams update from pre-update ratings only after a result becomes available. |
| History coverage | Offensive/defensive play counts, QB dropbacks, and missing QB-proxy indicator, all derived from prior history. |

The QB proxy is the player with the most attributed dropbacks in the latest available prior team game. It never reads the target game's actual starter. Injuries, benchings, and depth-chart changes can therefore make it stale. Sacks and scrambles are included when attributable; scrambles can use the recorded rusher ID. Kneels, spikes, special teams, no-plays, and penalized plays are excluded from aggregate rates. Defaults are fixed: EPA 0, success 0.45, explosive rate 0.10, Elo 1500, and zero observed counts.

A source game's result availability must be strictly earlier than the target kickoff; equality is excluded. Cached potentially live PBP cannot become final simply by waiting: schedule completion is additionally masked by that season snapshot's fetch time. Every forecast carries advanced-history timestamps, QB proxy identity/source game, coverage, and source/code/model provenance. No target-game coverage indicator or season-end opponent statistic is a predictor.

## Comparison and interpretation

Both models use the exact same 5,426 validation targets from 2004–2024 and 271 decisive 2025 targets, identical expanding training IDs beginning in 1999, and the same annual cutoffs. Missing PBP uses declared priors and coverage features; there is no complete-case intersection or target dropping. Both use StandardScaler and fixed logistic regression with raw output probabilities. Calibration is measured, not fitted.

The comparison reports accuracy, Brier score, log loss, fixed-ten-bin ECE and reliability bins, per-season metrics, and individual and paired confidence intervals. Two thousand identical season/week block-bootstrap draws produce Model 2 minus Model 1 deltas, recomputing ECE each time. Favorable point estimates alone are not evidence of improvement. These are unadjusted per-metric intervals, conditional on fitted forecasts; they omit model-refit uncertainty, cross-week dependence, source revisions, and selection across future experiments.

Historical raw sources can be revised, and the conservative kickoff-plus-24-hours publication proxy is not an archive of historical delivery times. Local temporal filtering and chronological EP training remove direct outcome and estimator-training lookahead in the supplied data; they cannot prove exact historical source availability. A future prospective test or genuinely unexamined season is needed for stronger confirmation. No real-money trading is implemented.
