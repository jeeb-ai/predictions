# NFL Model 1.0 versus Model 2.0

Run: `20261027T053315125622Z-d8868df1`

| Split | Model | Games | Accuracy | Brier | Log loss | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation | model_1 | 5426 | 62.79% | 0.22385 | 0.63863 | 0.02157 |
| validation | model_2 | 5426 | 65.09% | 0.22006 | 0.63015 | 0.01919 |
| holdout | model_1 | 271 | 61.62% | 0.22198 | 0.63231 | 0.05258 |
| holdout | model_2 | 271 | 62.36% | 0.21993 | 0.62695 | 0.05938 |

Differences below are Model 2.0 minus Model 1.0. Accuracy is better when positive; losses and ECE are better when negative.

| Split | Metric | Difference | Paired 95% interval | Evidence |
| --- | --- | ---: | --- | --- |
| validation | accuracy | 0.02304 | [0.01422, 0.03203] | supports_improvement |
| validation | brier_score | -0.00379 | [-0.00548, -0.00195] | supports_improvement |
| validation | log_loss | -0.00848 | [-0.01226, -0.00434] | supports_improvement |
| validation | expected_calibration_error | -0.00238 | [-0.01373, 0.01087] | inconclusive |
| holdout | accuracy | 0.00738 | [-0.02602, 0.03732] | inconclusive |
| holdout | brier_score | -0.00205 | [-0.00981, 0.00510] | inconclusive |
| holdout | log_loss | -0.00536 | [-0.02227, 0.01041] | inconclusive |
| holdout | expected_calibration_error | 0.00681 | [-0.03232, 0.05140] | inconclusive |

## Method and limitations

- Both versions use identical scheduled games, decisive regular-season targets, expanding prior-season training IDs, season-opening cutoffs, StandardScaler, and fixed C=1 logistic regression. No missing-data intersection or split changes are allowed.
- The first five seasons are training warm-up by default. Annual coefficients and scaling remain frozen; earlier same-season outcomes may enter later pregame form only after their strict availability cutoff.
- The 2025 outcomes were previously reported for Model 1.0. It is a reused holdout excluded from Model 2.0 fitting, tuning, selection, and calibration; it cannot be described as a newly unseen test.
- Probabilities are raw logistic outputs. Calibration is measured using ten fixed-width bins, not fitted on evaluation targets.
- Individual and paired 95% percentile intervals resample identical season/week blocks with fixed forecasts, recomputing ECE in every sample. They omit model-refit uncertainty, cross-week dependence, and source revisions; per-metric intervals are not adjusted for multiple comparisons.
- Paired deltas are Model 2.0 minus Model 1.0. Improvement evidence requires the corresponding interval to exclude zero in the favorable direction; no overall improvement or trading-value claim follows automatically.
- Model 2.0 recomputes EPA and success locally using EP models trained only on prior seasons; published retrospective nflfastR EPA is excluded from its predictors. EP training provenance must be retained with the source metadata.
- Historical raw source snapshots can be revised retrospectively. Pregame lagging and prior-season EP fitting enforce temporal cutoffs in the supplied data, but do not establish archived historical publication availability.

## Actual execution and interpretation

This report uses actual nflverse snapshots and the fixed [prespecified design](model2-design.json). No tuning followed the holdout results. A clean locked installation reproduced every aggregate and per-season metric exactly; Model 1.0 point metrics match the original report. Its uncertainty is recomputed jointly here using 2,000 paired draws rather than the original separate 1,000-draw intervals.

- Generated at: `2026-10-27T05:33:15.125622+00:00`
- Source-code digest: `d1ab4c067d2b695d61768da7f4194371e4f07e7e09392ddae0aeba8bbc21a7ec`
- Schedule snapshot: `5b8c66203d43c8faa93b0519b3ca078970aa0bdb39ccb51076dc99e5248f1486`
- Schedule outcome cutoff: `2026-10-26T00:00:00+00:00`
- Full execution directory: `reports/model2-clean/20261027T053315125622Z-d8868df1` (generated artifacts excluded from Git).
- Same targets: 5,426 validation and 271 decisive holdout games; every annual training-ID hash and cutoff matches across versions.
- All recorded base, advanced-history, and QB-proxy availability timestamps precede their target kickoff. Both sides have observed offensive PBP history for 100.0% of evaluation games.
- The 2025 EP estimator trains on 2022–2024 only; the 2025 win estimator trains on 1999–2024 only. Neither learns coefficients from 2025 targets.
- Local validation: 146 tests passed, Ruff lint and formatting passed, and source/wheel builds passed under a clean locked Python 3.12 environment. CI also checks Python 3.11 and 3.12; live status is available on the pull request.

Model 2.0's validation gains in accuracy, Brier score, and log loss have favorable paired intervals. Holdout accuracy improves by only 0.74 percentage points in point estimates; its paired 95% interval is **−2.60 to +3.73 percentage points**. All four holdout metric-difference intervals include zero. ECE is nominally worse. The result supports historical validation gains, **not a claim of reliable 2025 holdout superiority or better calibration**.

The new EPA is a locally learned remaining-half measure reconstructed from nflverse PBP, not published nflfastR EPA. Historical source revisions and imputed publication delays remain limitations, and 2025 is a reused holdout whose Model 1.0 outcomes were already observed. See [methods](model2-method.md), [machine-readable metrics and audit](model2-comparison.json), and [reliability plots](model2-reliability.svg). No trading or profit evaluation was performed.

```bash
uv sync --frozen --extra dev
uv run --frozen nfl-predict fetch-pbp --start-season 1999 --through-season 2025
uv run --frozen nfl-predict compare --input data/raw/5b8c66203d43c8faa93b0519b3ca078970aa0bdb39ccb51076dc99e5248f1486.csv --through-season 2025 --as-of 2026-10-26T00:00:00Z --offline
```

Exact replay requires retaining the schedule and PBP snapshots with the checksums recorded in the JSON; later upstream revisions may change fetched contents. Per-model forecasts, training metadata, and EP artifacts remain cached locally rather than committed.
