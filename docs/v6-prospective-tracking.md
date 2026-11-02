# Version 6, Phase 2: prospective results and accuracy

This phase builds on [Phase 1 automation](v6-automation.md), in a separate reviewable PR. NFL, NBA and NHL models, historical validations, authentication and paper records are preserved. The homepage stays prediction-first. An expandable **Prospective accuracy — future games** panel shows actual future-test performance independently of retrospective backtests. No real-money execution or new market provider is added.

## Automatic operation

Each independent `platform-worker` cycle refreshes forecasts and then reconciles results, even if that sport's forecast refresh fails. Completed forecast slots are reused when retrying result reconciliation. Provider failures preserve previously verified observations and report missing results; other sports continue independently. Current report files are atomically published and the dashboard only reads them. It makes no result-provider requests and writes no tracking data.

The default forecast interval remains 900 seconds; official-result polling defaults to 3,600 seconds. Set `automation.results_interval_seconds` (60–86,400) in your existing config. Failed result checks retry on the existing `retry_seconds` schedule, subject to the worker's process watchdog and provider retry limits. At most 100 eligible fixtures per cycle are checked, prioritizing unverified games and rotating by last check. Finals are rechecked for seven days; unresolved games continue to be checked. `platform-reconcile --refresh-all` explicitly rechecks older final games, including corrections outside that window. An internet outage cannot convert an unverified game into a loss.

**Conservative availability delay:** results are considered only 24 hours after the independently verified scheduled start. This preserves the existing availability convention and avoids declaring a live/unfinished game final. It means the accuracy panel deliberately does not update immediately after the final whistle. An official final state is required in addition to this delay.

## Primary cohort and audit

The first valid **locally received pregame** forecast per sport/game is the primary record. It is selected by immutable receipt sequence, not by subsequent accuracy, model confidence or version. Both issuance and actual receipt must precede kickoff. Later pregame forecasts remain separate immutable receipts and cannot replace the primary. Old forecast files first imported after play are retained as late imports, excluded from prospective metrics. No historical accuracy is manufactured during migration.

Original probabilities, projected score means and verified joint-distribution research-line probabilities are frozen at receipt. The evaluator never refits a model or reads a future distribution file to grade old predictions. Original source references, feature cutoffs, model versions, payload hashes and all later forecast receipts remain available in `state_dir/prospective.sqlite`. Confirmed results, revisions and metric summaries append to that same transactional SQLite journal. The database is an audit aid, not external tamper-proof timestamping: synchronize the host clock and back up the complete durable state.

A conflicting verified result temporarily removes the fixture from grading and retains a disputed observation. A second independently fetched matching observation at least five minutes later is required to confirm the correction. Stale observations cannot roll back newer evidence. A lower-trust archive cannot retract a league-hosted final or resolve an official-source dispute on its own. Earlier metric snapshots and result revisions remain intact. The latest JSON report is a replaceable view of this history.

Cancellation is void; postponed/incomplete games remain pending. A final with changed opponents or a changed scheduled start is held for review rather than matched by team name or silently reassigned. This conservative rule can leave rescheduled fixtures ungraded; there is no unsafe automatic override. NFL ties are excluded from decisive-winner accuracy/Brier/log loss and included in score errors. A precisely 50% forecast has no invented winner pick but receives proper probability scoring. NBA/NHL final ties are invalid. NHL **REG, OT and SO** are explicit: full-game final scores include the deciding goal, while regulation scores for OT/SO are tied with that deciding goal removed. Full-game and regulation line evaluations stay separate.

## Official result sources and limitations

- **NFL:** public league-hosted `https://www.nfl.com/scores/{season}/week-{week}` (postseason route when applicable). The adapter reads the page's embedded scoreboard data without executing scripts or accessing a private API. It resolves explicit game identities, opponents, scheduled time, final phase and quarter/OT totals. A redirect/homepage, contradictory totals or unknown markup is rejected. Current score-strip and requested-week game objects are separately supported. React page changes remain a provider risk; failures are visible and ungraded.
- **NBA:** official NBA CDN league schedule, `https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json`, requiring explicit final game status and team scores. If unavailable or missing an older season, the existing documented NBA Stats **official-origin archive** is used with an independently observed ESPN schedule timestamp and completed-state/score cross-check. This fallback is labeled `official_stats_archive`, not represented as a new direct league-hosted response. Conflicting scores are disputed. See [NBA acquisition documentation](nba-download-fix.md) for archive origins/access constraints. No dependency on the unreliable live stats API is introduced.
- **NHL:** public official `https://api-web.nhle.com/v1/gamecenter/{game_id}/landing`, requiring explicit final state, scores and REG/OT/SO period outcome. Missing or inconsistent scope is rejected; no outcome type is guessed from the margin. A four-request thread bound isolates per-game failures.

Raw responses, checksums, fetch times, URLs and provider classifications are stored under `state_dir/outcome-sources`. Acquisition uses bounded retries/timeouts and checksum-verified snapshots. These new outcome snapshots are **not model inputs**. Public access is not an unrestricted redistribution license; retain provider attribution and review their terms before broader redistribution. This is not market-settlement evidence: league scores alone do not establish a particular contract's rules or payout.

## Metrics and dashboard

Per sport, the panel reports the actual winner record, eligible/graded/pending counts, accuracy, Brier score, log loss, reliability bins and team-score MAE. Reports additionally contain margin/total MAE, model-version/confidence groups, 7/30/90-day kickoff windows, weekly trends, original forecast history and observation IDs. Empty samples have null metrics and display **—**, with no illustrative performance substituted. Metrics use only information observed by their report timestamp. Weekly 95% block-bootstrap intervals (500 draws) begin at 20 binary results across at least two UTC calendar weeks; they measure sampling variation with fixed forecasts, not uncertainty from refitting or source revisions. Small samples cannot demonstrate model calibration or profitability.

Spread and total probabilities captured at fixed research scenarios are graded as win/push/loss, with three-outcome Brier, log loss, push frequency and win-event reliability. These are **not recorded executable market prices**. They preserve prospective evidence for later key-number investigation, but do not constitute market-line history, verified EV or a betting edge. Market value remains **PASS**. New compliant market feeds, NFL key-number model changes and market-specific settlement validation belong to Phase 4; this phase does not claim to solve them.

The performance panel and each sport's calibration/history expanders provide recent-period selection, actual reliability/trend charts once there are results, original probabilities, projected/actual scores and Chicago display timestamps. The existing historical Research / Administration view remains separate. Complete machine-readable groups and recorded-line evidence are available via `platform-performance`.

## Exact Mac update and verification commands

Review both Phase 1 and Phase 2 PRs; do not merge automatically. To test this Phase 2 branch, which includes the Phase 1 dependency, from your existing checkout:

```bash
set -e
git fetch origin feat/v6-prospective-tracking:refs/remotes/origin/feat/v6-prospective-tracking
git switch feat/v6-prospective-tracking
uv sync --frozen --extra dev --python 3.12
uv run --frozen nfl-predict research-init --config research.local.json
uv run --frozen nfl-predict platform-capture --config research.local.json
uv run --frozen --extra dev pytest
uv run --frozen nfl-predict platform-run --sport NFL --config research.local.json
uv run --frozen nfl-predict platform-run --sport NBA --config research.local.json
uv run --frozen nfl-predict platform-run --sport NHL --config research.local.json
uv run --frozen nfl-predict platform-performance --config research.local.json
uv run --frozen nfl-predict dashboard --config research.local.json
```

Install prerequisites if needed with `brew install uv python@3.12`. Existing config/key/records are retained; no reset or destructive migration is required. A single sport's result-only check is `uv run --frozen nfl-predict platform-reconcile --sport NBA --config research.local.json`. Add `--refresh-all` to force older verified finals to be revisited. Let a daemon run using `platform-worker --sport NBA --config research.local.json`; separate processes handle NFL and NHL. A nonzero `platform-run` result indicates a failed/partial cycle; retained forecasts/results remain readable and the daemon retries.

For unattended updates when your Mac is off, deploy these reviewed branches to an actual always-on Linux host using the [persistent Compose commands and volume/backup instructions](v6-automation.md#persistent-private-linux-host). Rebuild then `docker compose -f deploy/compose.yml up -d --no-build --remove-orphans`; the same three workers now grade results automatically. Keep the authenticated dashboard private on loopback. Secure iPhone HTTPS deployment and automated backups/monitoring are Phase 5, not claimed here. This temporary development workspace is not continuous hosting.

## Actual validation

[Recorded validation and authenticated UI evidence](v6-tracking-validation.json) includes real official-result checks and actual worker/report counts. Three completed games per sport were acquired and every final score matched existing independently verified historical data; NHL samples cover regulation, OT and shootout. Those nine historical checks had **no eligible prospective receipt**, so they contribute zero prospective wins or losses. At validation the real captured future-test cohort was still pending: no accuracy, calibration improvement or betting profit is claimed.

Automated tests cover receipt selection, late imports, probability-score reconciliation, frozen score distributions, NFL ties, NHL scopes, cancellations/postponements/disputes, correction confirmation, source hierarchy, future-source rejection, immutable records, duplicate prevention, poll intervals, provider failures, process restarts/staleness and authenticated capture→result→dashboard behavior. Tests use explicitly synthetic local fixtures; real-source evidence is recorded separately. Models and existing historical validation remain unchanged. CI results and platform-specific limitations are recorded in the PR, rather than describing a queued macOS runner as a successful Mac execution.
