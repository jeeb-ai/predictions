# Reliable NBA acquisition (5.3.1)

The normal historical and current-season acquisition path no longer depends on a successful connection to `stats.nba.com`. Existing verified historical NBA Stats snapshots remain reusable; new historical downloads use published NBA Stats archives on GitHub. If the NBA archive lacks a season, validated ESPN archives supply the same box-score inputs. This fixes transport availability without changing winner/score estimators, feature definitions, chronological folds or the existing NFL/NHL models.

## Diagnosis

The failing request is **not the homepage**. For the default first season it is:

```
https://stats.nba.com/stats/leaguegamefinder?LeagueID=00&Season=2010-11&SeasonType=Regular+Season&PlayerOrTeam=T
```

The same four parameters are used for each following season. A root-page 301 only proves the root redirect can be reached; it says nothing about this API route. Remote disconnects and read timeouts occur before a usable API response, not inside model training.

All 17 direct season requests, 2010–11 through 2026–27, returned HTTP200 in this cloud environment. The exact URLs, response row counts and timings are in [the acquisition verification](nba-download-verification.json). **The Mac failure was not reproducible here, and the supplied exceptions do not identify its season.** The old implementation did not print that context. This fix logs each requested season, provider, exact failed GET, attempt count, failure type and validated game count. It does not pretend to identify a Mac-specific failing season without a corresponding log. Region/network-specific API filtering is plausible; no server-side cause was proven. Changing browser headers or repeatedly testing the homepage is not a permanent solution.

A separate CDN schedule probe returned HTTP403, while the final application request with its normal headers succeeded. Its fallback is also checked against real archived NBA identities/dates and actual ESPN tipoff timestamps, with controlled failure tests. It never invents a midnight tipoff.

## Reliable providers and validation

- Historical boxes: [sportsdataverse NBA Stats team-box releases](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/nba_stats_team_boxscores), documented by [hoopR-nba-stats-data](https://github.com/sportsdataverse/hoopR-nba-stats-data/blob/main/docs/datasets/team_boxscores.md). These originate in official NBA Stats v3 boxes and preserve official NBA team/game IDs.
- Independent game coverage/identities/scores: [NBA schedule releases](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/nba_stats_schedules). Every completed-season box set must match final schedule IDs, home/away identities, dates and scores. Known regular-season totals catch a truncated denominator too: 1,230 ordinarily, 990 in 2011–12, 1,229 in 2012–13, 1,059 in 2019–20 and 1,080 in 2020–21. Supported historical start years are 2010 onward. Canceled-game placeholders are excluded only when the schedule says the game was not final; no result is imputed.
- Current fallback: [ESPN team boxes](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_nba_team_boxscores) and [ESPN schedules](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_nba_schedules), distributed by [hoopR-nba-data](https://github.com/sportsdataverse/hoopR-nba-data). The NBA Stats 2026–27 box asset had not been published; these supplied real current results. NBA game IDs come from matching the official NBA schedule, not from assuming ESPN IDs are NBA IDs. Scores and team sides must agree. Dates use actual ESPN UTC timestamps converted to Eastern. Observed period counts distinguish overtime and preserve the existing pace definition. ESPN `turnovers` matches the NBA box input; `total_turnovers` includes extra team turnovers and is deliberately excluded.

Archive filenames use the **ending** year: `team_boxscores_2011.parquet` is 2010–11. Each file must match the publisher's GitHub release SHA256 and pass schema/season/numeric/pairing validation. A cached/downloaded HTML page, truncated game set, wrong season, incomplete pair, unknown overtime duration or contradictory score is rejected. Closed-season coverage is checked for **every** requested season. Current coverage requires every conservatively eligible final result; unresolved old schedule entries fail explicitly, rather than quietly assuming cancellations or missing games.

An early provider check found placeholders for the canceled 2013 Boston–Indiana game and historical ESPN dates that cross UTC midnight. Those cases have explicit handling and tests. The final 19,118-game archive dataset matches the original NBA Stats snapshots exactly in all 16 pregame feature values, scores, efficiencies and pace. The full chronological backtest reproduces the original validation/holdout metrics exactly. No model tuning or baseline replacement was performed.

Public releases are a legitimate documented distribution route; this does not bypass NBA access controls. hoopR software is MIT licensed, but that does **not** grant blanket commercial redistribution rights to the underlying NBA/ESPN data. Review underlying provider terms before redistribution/commercial publication. Raw games are not committed. Present-day source revisions remain possible; original historical publication vintages are not established.

## Retry, resume and freshness behavior

`--nba-provider auto` (default) checks existing valid NBA Stats caches, then uses NBA/ESPN archives. It makes **no live stats.nba.com request**. `archive` bypasses old direct-API caches and uses release data; `nba-stats` explicitly opts into the live API for diagnosis. Scheduled league workers inherit `auto`; `--nba-provider archive` can make the choice explicit.

NBA network calls have separate bounded connect/read timeouts, three attempts, exponential delays and bounded Retry-After handling. HTTP429, transient server errors, timeouts, nested `RemoteDisconnected`, incomplete transfers and invalid successful responses receive useful diagnostics. HTTP403 is not retried. TLS verification stays enabled; no alternate proxy, credentials, infinite retry or browser-fingerprint workaround is added. The shared snapshot adapter retains its existing four-attempt/40-second defaults for NHL callers, with safer writes and diagnostics. NFL providers are unchanged.

Raw bodies, retrieval timestamps, canonical URLs and hashes are retained per file. Metadata is published only after a complete validated body is atomically written. Per-request locks and the NBA acquisition lock prevent concurrent cache corruption. Each fully validated season writes a manifest under `data/leagues/nba/seasons/`. After a later season fails, earlier files/manifests remain reusable and are rechecked. Resume occurs **at file/season boundaries**, not at a byte offset inside a partially transferred parquet file. No delete-and-retry ritual is required. A real second historical acquisition completed with **zero HTTP requests**.

`nba-run` refreshes only the requested last season; old validated seasons reuse cache. `nba-fetch --refresh` does the same explicitly. Data eligibility remains strictly after recorded schedule day plus48hours, capped by the earliest actual retrieval time of required source snapshots. Waiting alone cannot make a potentially incomplete cached score eligible. Rest/back-to-backs still use known schedule dates, even when the previous result is not usable. Confirmed player availability remains unknown. Acquisition failures do not publish a partial prediction batch or overwrite valid NFL/NHL predictions; previously retained forecasts keep their stale labeling.

## Actual validation

The [complete archive-backed backtest](nba-download-validation.json) has 19,118 historical games, 10,749 chronological validation games and 1,230 reserved 2025–26 games. Holdout accuracy is **68.21%**, Brier **0.20856**, log loss **0.60529**, ECE **0.03765**, and per-team score MAE **9.56527**. These reproduce the existing report; no accuracy improvement is claimed and 2025–26 is no longer newly unseen.

A real current-season run produced **52 future forecasts**, using 68 completed current-season archive games and no live NBA Stats API request. The [verification](nba-download-verification.json) records source/transport counts and comparisons. [Authenticated UI evidence](nba-download-ui-evidence.json) checks actual independent probabilities, scores and contribution reconciliation. Counts describe this run, not a guaranteed future slate. NBA markets remain PASS without verified executable quotes/fees/settlements; no betting edge or real-money trading is implemented.

Automated checks cover transport failures/retries, Retry-After, response validation, atomic cache retention, corruption/checksum rejection, file/season resume, completeness, canceled games, cross-provider identity/score checks, UTC/Eastern dates, overtime/turnovers, source cutoffs and UI. CI retains full Python3.11/3.12 and container checks, plus a macOS/Python3.12 NBA acquisition/UI test job. CI uses fixtures, not an unstable live NBA service; real-data validation above was run separately from Linux.

## Exact Mac update commands

From your existing repository checkout, while the fix PR is under review:

```bash
set -e
git fetch origin
git switch fix/nba-reliable-history
uv sync --frozen --extra dev --python 3.12
uv run --frozen --extra dev pytest
uv run --frozen nfl-predict nba-fetch --start-season 2010 --through-season 2025 --nba-provider archive > /tmp/nba-downloads.json
uv run --frozen nfl-predict nba-backtest --start-season 2010 --through-season 2025 --nba-provider archive > /tmp/nba-report.json
uv run --frozen nfl-predict research-init --config research.local.json
uv run --frozen nfl-predict nba-run --start-season 2010 --through-season 2026 --nba-provider archive --validation /tmp/nba-report.json --config research.local.json
mkdir -p reports/research
cp /tmp/nba-report.json reports/research/nba-approved.json
uv run --frozen nfl-predict dashboard --config research.local.json
```

If uv/Python are missing and you use Homebrew, run `brew install uv python@3.12` first. After the PR is merged, use `git switch main` and `git pull --ff-only` in place of switching to the fix branch. Keep your existing config and private dashboard key; initialization preserves them. These commands do not clear any old caches, forecasts, models or ledger. For October2026 through spring2027 the season start year is2026. Use your completed report at the configured `reports_dir` if you changed that path from the default above.

For continuous refresh, in another terminal:

```bash
uv run --frozen nfl-predict leagues-worker --sport NBA --sport NHL --nba-provider archive --config research.local.json
```

The optional `--nba-provider` only affects NBA; NHL and the separate NFL worker retain their original providers. Persistent Linux Compose workers use the improved default automatically after rebuilding the updated code.

For an **optional one-time direct API diagnostic**, a separate cache reveals exactly which season fails without touching your working archive cache:

```bash
uv run --frozen nfl-predict nba-fetch --start-season 2010 --through-season 2025 --nba-provider nba-stats --cache-dir data/leagues/nba-api-diagnostic > /tmp/nba-api-diagnostic.json
```

That command stops at the first exhausted request and names its season and exact URL on stderr. It is not required to run predictions. A complete upstream archive and an accessible GitHub endpoint remain prerequisites; unavailable, stale or contradictory sources fail clearly rather than generating fake games.
