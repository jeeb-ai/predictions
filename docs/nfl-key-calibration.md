# NFL key-number calibration investigation — research only

The production winner and scoring engines are unchanged. This isolated command tests a coherent probability adjustment to the existing `ridge_joint_gaussian` scoring model. It does not promote a model, change live inference, create paper positions or use market prices.

The Gaussian model smooths over football's discrete scoring structure. In the original 2025 diagnostic, a home −3 scenario had only 2.79% predicted pushes versus 8.46% observed. This suggests missing score-lattice structure, not a known tradable opportunity. A simple categorical correction is a diagnostic before considering drive/possession and overtime models.

## Method and leakage boundaries

For each **identical original chronological fold**, refit the existing selected Ridge joint score model on the same training/calibration games. Use only its older calibration season to fit multiplicative joint-score weights for absolute margins `{0,3,7,10,14}` and totals `{37,41,44,47,51}`, with all other values as references. These categories, L2 penalty `0.02` and log-weight bounds `[-2,2]` were fixed before running this experiment. The corrected mass is proportional to base mass times the margin and total weights, then normalized across all home/away scores. This preserves coherent score/line/push probabilities and home/away symmetry of the correction; it is not a collection of unrelated probability tweaks.

The calibration season already estimates the base residual bias/covariance; the extra weights also fit that same older calibration season, never test outcomes. This is extra fitting complexity and may overfit. The command rejects invalid chronology, different input SHA256, changed fold counts or a failure to reproduce original base joint log loss and score MAE. Model and training/calibration-ID hashes are retained. It does not load untrusted external serialized models; it refits the existing implementation from the verified schedule.

The repeated 2025 season **is not newly untouched**: its outcomes and key-number problems had been inspected before this experiment was designed. Results are retrospective evidence, requiring new prospective confirmation. No hyperparameters were tuned after the first experiment results.

## Actual results

The [machine-readable report](nfl-key-calibration.json) contains all folds, weights, source/code hashes, metrics, push diagnostics, fixed-prediction week-block intervals and paired differences. Original baseline results reproduced to floating-point tolerance. Validation comprises 5,183 games (2005–2024); the reused 2025 season contains 272 games, including one tie.

| Metric | Validation base → candidate | Reused 2025 base → candidate |
|---|---:|---:|
| Joint score log loss | 7.36285 → 7.28594 | 7.31210 → 7.23983 |
| Per-team score MAE | 7.74154 → 7.74522 | 7.49095 → 7.49298 |
| Home −3 three-outcome Brier | 0.52845 → 0.52594 | 0.51901 → 0.51549 |
| Home −7 three-outcome Brier | 0.45475 → 0.45326 | 0.44107 → 0.43902 |
| Total 44 three-outcome Brier | 0.52765 → 0.52772 | 0.51844 → 0.51882 |
| Total 47 three-outcome Brier | 0.50571 → 0.50550 | 0.51553 → 0.51520 |

| 2025 scenario | Base push estimate | Candidate push estimate | Actual pushes |
|---|---:|---:|---:|
| Home −3 | 2.79% | 5.47% | 8.46% |
| Home −7 | 2.61% | 3.29% | 4.78% |
| Total 44 | 3.00% | 3.09% | 4.41% |
| Total 47 | 3.07% | 3.96% | 4.04% |

The candidate's paired 2025 joint-log-loss difference is −0.07227 with a 95% week-block interval **[−0.10696, −0.04087]**. The spread −3 three-outcome Brier difference is −0.00352 **[−0.00680, −0.00038]**. Those diagnostics support reduced retrospective loss under these frozen predictions. They do **not** establish full calibration or a betting edge. Margin −3 mass remains materially too small; the −7 proper-loss intervals include zero. Total 44 Brier worsens slightly and neither total scenario has convincing paired evidence of improvement. Per-team MAE changes are practically small and do not show a 2025 gain.

All intervals use 500 paired season/week block resamples with seed42 and fixed fitted forecasts. They omit refitting, source vintages, experiment selection and cross-block dependence; they are not multiplicity-adjusted. Fixed −3/−7/44/47 scenarios are not historical market lines. No prices, fees, available liquidity or settlements were evaluated. A reliable betting-edge claim would be unsupported.

**Decision: no production promotion.** Keep the original NFL models and market PASS gates. Next useful work is prospective key-number monitoring plus a historically validated drive/possession scoring model with explicit scoring and overtime structure, using fresh data for confirmation.

## Exact Mac commands

With Homebrew, install dependencies using `brew install python@3.12 uv`. From your existing repository (or `git clone https://github.com/jeeb-ai/predictions.git` then `cd predictions`):

```bash
git fetch origin
git switch research/nfl-key-calibration
uv sync --frozen --extra dev --python 3.12
uv run --frozen --extra dev pytest
uv run --frozen nfl-predict fetch
```

Use the **original CSV whose SHA matches your completed scoring report**, not an arbitrarily refreshed source. The measured report used the path below. If you do not retain that snapshot, run the standard `score-backtest` on your downloaded verified CSV first, and use its new completed report. These are real computations and may take several minutes:

```bash
uv run --frozen nfl-predict score-backtest --input data/raw/5b8c66203d43c8faa93b0519b3ca078970aa0bdb39ccb51076dc99e5248f1486.csv --through-season 2025 --as-of 2026-10-26T00:00:00Z > /tmp/nfl-score-base.json
uv run --frozen python -m predictions.scoring_key_calibration --input data/raw/5b8c66203d43c8faa93b0519b3ca078970aa0bdb39ccb51076dc99e5248f1486.csv --validation /tmp/nfl-score-base.json > /tmp/nfl-key-calibration.json
```

To reproduce the committed diagnostic with that exact original snapshot, `--validation docs/v5-scoring-validation.json` is also supported. Different sources/cutoffs produce different results; the command refuses to silently compare mismatched samples. New models and per-game results are written to a fresh ignored `reports/scoring-calibration/key-calibration-*` directory. This research command does not alter the dashboard, live models, historical comparison artifacts or paper ledger.
