# NFL pregame logistic-regression backtest

Run: `20261026T194204061367Z-e243c51e`

| Split | Games | Accuracy | 95% accuracy CI | Brier | Log loss | ECE | Home-rate baseline accuracy |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| validation | 5426 | 62.8% | 61.5%–64.1% | 0.2239 | 0.6386 | 0.0216 | 55.9% |
| holdout | 271 | 61.6% | 55.8%–67.4% | 0.2220 | 0.6323 | 0.0526 | 53.9% |

The JSON report contains full per-season metrics, reliability bins, and baseline losses.

## Method and limits

- Regular-season decisive games only; tied games are excluded from binary targets (their past form may count half a win).
- A StandardScaler and fixed C=1 logistic regression are fitted separately before each evaluation season, using older-season outcomes available strictly before its first kickoff.
- The latest selected season is a separate holdout: no fitting, model selection, tuning, or probability recalibration uses its targets. Earlier holdout games can enter later pregame rolling form once available.
- Features use completed historical outcomes only, under a conservative kickoff-plus-24-hours availability assumption, not archived publication timestamps; corrected historical source values are not point-in-time snapshots.
- Probabilities are raw logistic outputs; calibration is measured, not fitted. Reliability uses ten fixed-width bins, including probability 1 in the final bin.
- 95% percentile intervals resample season/week blocks with fixed fitted forecasts; they do not include model refitting, source revision uncertainty, or dependence across weeks.
- The baseline predicts the training-set home-win rate; always-home accuracy is also reported. These retrospective results do not establish a trading edge.

## Per-season model results

| Season | Split | Games | Accuracy | Brier | Log loss | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2004 | validation | 256 | 62.1% | 0.2263 | 0.6433 | 0.0243 |
| 2005 | validation | 256 | 63.7% | 0.2185 | 0.6357 | 0.0673 |
| 2006 | validation | 256 | 52.3% | 0.2433 | 0.6800 | 0.1204 |
| 2007 | validation | 256 | 65.2% | 0.2193 | 0.6282 | 0.0623 |
| 2008 | validation | 255 | 63.9% | 0.2213 | 0.6341 | 0.0401 |
| 2009 | validation | 256 | 64.1% | 0.2086 | 0.6034 | 0.0606 |
| 2010 | validation | 256 | 58.6% | 0.2369 | 0.6660 | 0.0422 |
| 2011 | validation | 256 | 68.8% | 0.2143 | 0.6170 | 0.0786 |
| 2012 | validation | 255 | 64.3% | 0.2213 | 0.6335 | 0.0343 |
| 2013 | validation | 255 | 64.7% | 0.2175 | 0.6235 | 0.0896 |
| 2014 | validation | 255 | 63.9% | 0.2136 | 0.6152 | 0.0568 |
| 2015 | validation | 256 | 63.7% | 0.2306 | 0.6529 | 0.0557 |
| 2016 | validation | 254 | 62.6% | 0.2224 | 0.6353 | 0.0598 |
| 2017 | validation | 256 | 65.2% | 0.2200 | 0.6303 | 0.0445 |
| 2018 | validation | 254 | 65.4% | 0.2179 | 0.6258 | 0.0461 |
| 2019 | validation | 255 | 63.1% | 0.2247 | 0.6388 | 0.0726 |
| 2020 | validation | 255 | 61.6% | 0.2276 | 0.6462 | 0.0760 |
| 2021 | validation | 271 | 60.5% | 0.2328 | 0.6592 | 0.0805 |
| 2022 | validation | 269 | 62.5% | 0.2234 | 0.6375 | 0.0624 |
| 2023 | validation | 272 | 60.3% | 0.2326 | 0.6567 | 0.0401 |
| 2024 | validation | 272 | 62.5% | 0.2266 | 0.6454 | 0.0583 |
| 2025 | holdout | 271 | 61.6% | 0.2220 | 0.6323 | 0.0526 |

## Reproduction and provenance

This is an actual run against the fetched nflverse CSV, not synthetic test performance. Hyperparameters and feature defaults were fixed before examining the holdout; no tuning or recalibration was performed. A final repeat after provenance-field and opening-season-cutoff fixes produced the same measured aggregate values.

- Source: https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv
- Snapshot SHA-256: `5b8c66203d43c8faa93b0519b3ca078970aa0bdb39ccb51076dc99e5248f1486`
- Source fetched at: `2026-10-26T19:39:22.532045+00:00`
- Outcome cutoff: `2026-10-26T00:00:00+00:00`
- Run generated at: `2026-10-26T19:42:04.061367+00:00`
- Source-code digest: `91ef8dbbc899976be8b9206e67e1f7a176a6555c9400366aaa24dfa02e48148a`
- Warm-up training starts in 1999; annual validation covers 2004–2024; the separate final holdout is 2025.
- Ties are excluded from binary targets; 2025 has 271 decisive games and one excluded tie.
- Checks: 59 automated tests passed, Ruff lint and formatting passed, and wheel/source builds passed on Python 3.12.14 with the locked project dependencies. CI for Python 3.11 and 3.12 is configured; remote CI is not yet verified.

```bash
uv sync --frozen --extra dev
uv run --frozen nfl-predict fetch
uv run --frozen nfl-predict backtest --input data/raw/5b8c66203d43c8faa93b0519b3ca078970aa0bdb39ccb51076dc99e5248f1486.csv --through-season 2025 --as-of 2026-10-26T00:00:00Z
```

The fetch command may return a different checksum after upstream revisions. Exact replay requires retaining this raw snapshot and its metadata; it is cached locally and excluded from Git. A future fetch alone cannot guarantee identical data. The repository contains aggregate results and a reliability plot, not the raw source or model binaries. See `backtest-2025.json` for full aggregate calibration bins and confidence intervals.

The holdout's mean predicted home-win probability was 56.08%, versus an observed home-win fraction of 53.87%. Holdout ECE was 0.0526, compared with 0.0239 for the constant training-rate baseline. Lower Brier/log loss does not imply that probabilities are fully calibrated or that a trading edge exists.

Upstream and underlying-source redistribution/commercial rights have not been verified; no explicit license file was found at this source. The data are a present-day corrected snapshot, not a point-in-time source archive. No market data, moneyline profitability, prospective accuracy, simulations, or real-money trades were evaluated.
