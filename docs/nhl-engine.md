# NHL v1 — independent hockey forecasts

This is a separate hockey engine alongside the preserved NFL winner/scoring models and NBA engine. The NHL PR targets main and depends on integration PR #9 and NBA PR #10. Merge those first. PR #8's scoring code had been merged into its old base branch; #9 integrates that merge into main intact.

## Sources and provenance

Official outcomes, team identities, actual UTC starts and `gameOutcome.lastPeriodType` come from NHL's club-season API, `https://api-web.nhle.com/v1/club-schedule-season/{club}/{start}{end}`. The adapter discovers opponents from multiple seed clubs, including the disconnected 2020–21 divisions, and cross-checks duplicate official records. Current fixtures come from `https://api-web.nhle.com/v1/schedule/{date}` independently of markets. Postponed fixtures are excluded. Historical outcomes become usable strictly after actual start plus 24 hours; official and event snapshot observation times also cap eligibility.

Regulation shot geometry and observed goalies come from NHL event data distributed by [sportsdataverse / fastRhockey](https://github.com/sportsdataverse/fastRhockey-nhl-data). The [release assets](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/nhl_pbp_lite) use **end-year** names: `play_by_play_lite_2026.parquet` contains the 2025–26 season. The adapter verifies each asset against GitHub's published SHA256, retains the canonical URL and retrieval timestamp, and rejects mismatches. TLS verification stays enabled. Bounded concurrent GETs, retries, immutable raw snapshots and cache locks prevent silent partial/cache collisions.

An early bulk schedule comparison disagreed with official game outcomes, so it is **not** an outcome source. No MoneyPuck data or published vendor xG is consumed by this implementation. Publicly available files do not themselves confer permission to scrape or redistribute. fastRhockey software is MIT licensed; underlying NHL data rights still require review before public/commercial redistribution. Raw data are not committed. Historical snapshots include later corrections and cannot prove original historical publication vintages.

## Basketball and hockey are different models

Hockey xG is estimated here from shot distance/angle and their squares, empty-net status and skater advantage. Each annual standardized logistic shot model, including median imputation, uses only the **previous three shot seasons**, never the current season's goals. 2009–10 supplies initial warmup; team histories begin in 2010–11. These simple xG estimates omit rebound/rush sequences and shooter skill. They are not a vendor's richer shot-quality metric.

Each team's last ten eligible regulation games supply xG for/against, quality per unblocked attempt, attempts for/against, special-teams xG and recent form. Man-advantage features require an opponent below five skaters and exclude empty-net situations; an extra attacker against five skaters is not counted as a power play. Schedule dates supply rest/back-to-backs even when yesterday's result is not yet eligible. Online Elo starts at 1500, uses K16 and a nominal 40-point home offset, and updates only on eligible earlier results. Venue-specific effects and travel distance are not separately modeled.

The goalie feature is the previous eligible team's most-observed goalie, identified by tracked shots faced. Its save fraction uses earlier eligible appearances, including prior teams. It is **not a confirmed upcoming starter**, not isolated goalie skill and not a complete official save percentage. Missing proxy history uses a documented neutral 0.90 with zero tracked shots. Player availability and actual upcoming starting goalies remain unknown; the UI says so explicitly. A reliable point-in-time availability archive is required before those can honestly enter historical training.

Winner probabilities use a standardized C1 logistic classifier fitted on older full-game winners, followed by Platt calibration on the immediately preceding season. Independent Poisson (`alpha=0.1`) models forecast each team's **regulation** goals; older calibration data fit goal-scale and regularized draw-mass corrections. Separate old-data logistic models estimate home wins conditional on OT or shootout. The prior calibration season estimates shootout share. A regulation tie is redistributed to full-game outcomes by adding one deciding official goal. This handles official final scores, including shootouts; actual market rules may use different goal/settlement conventions. The main winner classifier and score-derived winner probability are separate and can disagree.

## Actual chronological results

There are 19,151 usable regular-season games from 2010–11 through 2025–26. Coverage and shot-model training seasons are recorded in [the actual report](nhl-validation.json). Five training seasons plus a separate calibration season precede each test season. Validation spans 2016–17 through 2024–25 (10,970 games); the separately reserved 2025–26 season contains 1,312 games. Hyperparameters were not tuned using that held season. A source-definition correction excluded 6-on-5 extra-attacker situations from power-play inputs before the final rerun; do not treat repeated retrospective evaluation as prospective evidence.

| Metric | Validation | 2025–26 holdout |
|---|---:|---:|
| Winner accuracy | 59.93% | 54.19% |
| Winner Brier | 0.23662 | 0.24838 |
| Winner log loss | 0.66571 | 0.68997 |
| Winner ECE (10 bins) | 0.01484 | 0.04840 |
| Per-team final-score MAE | 1.3666 | 1.3792 |
| Per-team final-score RMSE | 1.6923 | 1.7164 |
| Nominal 90% score interval coverage | 95.21% | 94.82% |
| Regulation three-outcome Brier | 0.62574 | 0.64993 |
| Predicted / actual extra-time frequency | 21.61% / 22.19% | 20.22% / 24.85% |
| Predicted / actual shootout frequency | 7.35% / 7.26% | 5.78% / 9.07% |

The holdout winner accuracy's fixed-prediction week-block 95% interval is **52.14–56.23%**; Brier/log-loss and score intervals are in the report. The home-rate baseline achieved 52.21% accuracy, 0.24986 Brier and 0.69287 log loss on the holdout; its score baseline MAE was 1.3814. These small numerical differences do **not** establish a reliable improvement or betting edge. Score intervals over-cover; extra-time and shootout frequencies are materially underestimated in the latest season. Spread/total reliability is imperfect. The dashboard labels NHL research limitations prominently.

The report includes calibration bins, CRPS, score/line proper losses and 500 fixed-seed week-block intervals. These intervals omit model refitting, feature/shot-model uncertainty and dependence across blocks. Fixed −1.5/5.5 line diagnostics are research scenarios, not historical executable market lines. Other entered lines are derived estimates, not separately validated edges. **Market comparison remains PASS**, with no NHL executable quote/fee/settlement adapter and no real-money execution.

## Exact Mac installation and run commands

Install Python 3.12 and uv with Homebrew (`brew install python@3.12 uv`). For a fresh checkout, run `git clone https://github.com/jeeb-ai/predictions.git` and `cd predictions`. From that directory while the PR is under review:

```bash
git fetch origin
git switch feat/nhl-engine
uv sync --frozen --extra dev --python 3.12
uv run --frozen --extra dev pytest
uv run --frozen --extra dev ruff check .
uv run --frozen nfl-predict research-init --config research.local.json
uv run --frozen nfl-predict nhl-backtest --start-season 2009 --through-season 2025 > /tmp/nhl-report.json
uv run --frozen nfl-predict nhl-run --start-season 2009 --through-season 2026 --validation /tmp/nhl-report.json --config research.local.json
uv run --frozen nfl-predict dashboard --config research.local.json
```

First historical download/fitting can take several minutes and use substantial disk space; raw event files are cached. Current-season updates reuse verified older xG/team caches and rebuild the current season. Use start year 2026 for October 2026 through spring 2027. Existing private-key initialization is preserved. The [NBA commands](nba-engine.md) populate that sport independently; the existing NFL worker continues to supply NFL forecasts. No market contract is needed.

For continuous refresh with the default reports directory, in another terminal:

```bash
mkdir -p reports/research
cp /tmp/nhl-report.json reports/research/nhl-approved.json
cp /tmp/nba-report.json reports/research/nba-approved.json
uv run --frozen nfl-predict leagues-worker --sport NBA --sport NHL --config research.local.json
```

On the persistent private Linux host, the optional Compose profile runs that worker alongside the existing NFL worker/dashboard:

```bash
docker compose -f deploy/compose.yml --profile leagues build
docker compose -f deploy/compose.yml --profile leagues up -d --no-build
docker compose -f deploy/compose.yml --profile leagues logs --tail 30 leagues
```

This profile mounts the committed measured approval reports read-only at the default `/app/reports/research/` paths. If your config overrides `reports_dir`, adjust those two mounts to match. The worker trains from verified downloaded histories; approval reports do not load machine-specific model paths. Dashboard remains bound to `127.0.0.1:8501`, authenticated, with read-only data mounts. No temporary workspace is represented as permanent hosting.

Per-sport errors preserve other valid snapshots; retained stale predictions are labeled. Immutable issued batches, source manifests, fitted models and separate full/regulation distributions are checksummed. Existing NFL comparisons, timestamped forecasts and paper ledger are preserved. NBA/NHL prospective automatic grading is not yet implemented; retain the archives for later grading. [Authenticated three-sport UI evidence](nhl-ui-evidence.json) records the actual engine outputs and scope selector. The responsive cards show model probability, projected score, supporting/opposing inputs and missing availability; coefficients remain in Advanced Analysis.
