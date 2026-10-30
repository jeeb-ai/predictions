# Version 6, Phase 1: automatic predictions and immediate receipt tracking

The prediction models, retrospective validations, private dashboard key and existing paper ledger are preserved. A prediction-only scheduler now runs NFL, NBA and NHL independently. It does not call market discovery, request prices, fill paper positions or execute real-money trades. The legacy research/paper commands remain available separately.

## What runs automatically

The production Compose stack starts three workers plus the existing authenticated dashboard. Each worker refreshes its sport's schedule/history and calls its **existing** winner/score forecasting implementation. Default intervals are15minutes per sport; existing config files acquire compatible defaults. Configure these optional fields in `research.local.json`:

```json
{
  "automation": {
    "intervals": {"NFL": 900, "NBA": 900, "NHL": 900},
    "retry_seconds": 60,
    "max_cycle_seconds": 7200,
    "nba_provider": "auto",
    "approvals_dir": "docs"
  }
}
```

Intervals are60–86,400seconds. `lookahead_hours` controls the forecast horizon (default168, maximum336). Results of every model run retain the existing immutable model/prediction artifacts. Completed validation reports already in the configured reports directory take precedence; otherwise the committed measured reports are used. The container includes those reports; it does not load machine-specific serialized models from them. Optional NFL score failure leaves its winner forecast intact. A missing NBA/NHL approval is an actionable setup error, never a reason to invent projections.

Each process has a per-sport lock, fixed UTC slots and durable SQLite job state. Completed slots are not generated again. A new model batch's forecast receipts and restart checkpoint commit together before publication. If publication is interrupted, the checkpoint republishes the same issued forecasts on restart. Failures retain old forecasts and retry on a bounded delay; the existing provider adapters retain their bounded network retries and verified caches. Separate sport processes, watchdog timeouts and independent persistent snapshots prevent one hung provider from blocking other sports. Missed slots do not generate backdated forecasts: restart uses the current clock and only future fixtures.

SIGTERM/SIGINT stop the worker and its child process group; a bounded grace period precedes forced cleanup. Docker restarts crashed workers after host/Docker startup. Provider errors do not stop a daemon permanently. Per-cycle private logs live under `state_dir/automation-logs/{sport}`. `platform-status` and the dashboard's expandable Research / Administration section show sport health, last successful refresh/cycle, current forecast count, staleness and recovery instructions without showing a raw Python traceback.

Cold history/EP/xG warmup can take tens of minutes and needs substantial resources. Plan for roughly16GB RAM and ample persistent disk for simultaneous initial warmup, then monitor actual growth. The default process limits are NFL2GB/NBA2GB/NHL4GB; warming is visible and bounded by the configurable timeout. Existing model algorithms and feature cutoffs are unchanged; NBA uses the validated archive acquisition path by default.

## Prospective tracking begins at receipt

`state_dir/prospective.sqlite` appends the original forecast, hashes, model version, source references, **actual local receipt time**, original generation/cutoff/kickoff times and verified score evidence. The database uses transactions, WAL, full synchronous writes, unique sport/forecast IDs and append-only triggers. Repeated receipt of the same ID/content is a no-op; changed content under an existing ID is rejected. Later forecasts remain separate records. Score evidence freezes means and exact win/push/loss probabilities at explicitly labeled fixed research lines; these are **not** market prices or measured betting edges.

Only forecasts received and generated strictly before the recorded start, with a valid information cutoff, are eligible. An old JSON imported after the game starts is retained as a **late import**, never relabeled as prospective because its claimed generation timestamp was early. No historical file modification time is accepted as issuance proof. Model/source/time checks and original leakage protections remain in force. The first eligible receipt per matchup will define the primary cohort in Phase2; later model versions/refreshes cannot displace it based on accuracy.

`platform-capture` immediately captures existing dashboard snapshots. Worker startup does this before fetching new data. Manual NBA/NHL publication also records receipts. The new NFL worker records receipts directly, without depending on the paper ledger. Existing legacy NFL journal/metrics remain intact; this phase does not silently backfill old NBA/NHL archives into a prospective cohort. Your old records are preserved, but postgame imports cannot manufacture a future-test history.

During development, real existing pregame snapshots were captured immediately:1NFL/52NBA/48NHL. Real independent cycles subsequently issued14NFL/52NBA/48NHL forecasts. [Measured evidence](v6-automation-validation.json) records actual counts, receipt ranges and test results. These are forecast counts, **not** a claimed accuracy record. Automatic official-result reconciliation and prospective metric publication are the separate Phase2 PR.

## Exact Mac update and initial capture

From your repository checkout, with Python3.12/uv installed (`brew install uv python@3.12` if needed):

```bash
set -e
git fetch origin feat/v6-automated-forecasts:refs/remotes/origin/feat/v6-automated-forecasts
git switch feat/v6-automated-forecasts
uv sync --frozen --extra dev --python 3.12
uv run --frozen nfl-predict research-init --config research.local.json
uv run --frozen nfl-predict platform-capture --config research.local.json
uv run --frozen --extra dev pytest
uv run --frozen nfl-predict platform-run --sport NBA --config research.local.json
uv run --frozen nfl-predict platform-status --config research.local.json
uv run --frozen nfl-predict dashboard --config research.local.json
```

Your existing configuration/key/data are preserved. `platform-capture` can safely be repeated; any existing forecasts already past kickoff are retained but excluded. Test another sport with `--sport NFL` or `--sport NHL`. A single-sport daemon is `uv run --frozen nfl-predict platform-worker --sport NBA --config research.local.json`. It remains active until stopped; `--max-cycles 1` is available for smoke checks. No new sports are added.

## Persistent private Linux host

Use an actual always-on host, not GitHub Actions, this temporary workspace or a Mac that will be switched off. Check out the reviewed code and retain the existing config and named data/model/report volumes. On a fresh host, initialize the config once; do not copy your secret key into Git. On an existing host, use its existing config/key/volumes.

```bash
uv sync --frozen --extra dev --python 3.12
uv run --frozen nfl-predict research-init --config research.local.json
docker compose -f deploy/compose.yml build
docker compose -f deploy/compose.yml up -d --no-build --remove-orphans
docker compose -f deploy/compose.yml exec -T worker nfl-predict platform-status --config /app/research.local.json
```

The old optional `leagues` service is replaced by the independent default `nba`/`nhl` services; `--remove-orphans` stops that obsolete container, without deleting its volumes. Do **not** run the old league daemon alongside these workers. The existing Compose project name/volumes and loopback-only dashboard binding remain unchanged. The new production command replaces the old scheduled paper-research loop; explicitly run that legacy command separately only if you still want its paper workflow.

Enable Docker startup on host boot. Health checks run per sport; `docker compose -f deploy/compose.yml logs --tail 30 nba` shows its process log. A health check reports both a live bounded warmup and fresh successful cycles; a failed latest refresh is visible even when a prior forecast remains usable. No temporary-cloud deployment is represented as permanent hosting. No public HTTP port or HTTPS/auth configuration is weakened: dashboard remains authenticated at127.0.0.1:8501. Secure iPhone HTTPS access is Phase5, after the requested tracking review.

Before migration, back up existing config, secret file and persistent volumes. For a consistent simple backup, stop workers, copy the data/model/report volumes (including SQLite WAL files if present), then restart them. Do not copy only an actively written `.sqlite` file and discard its WAL. Automated encrypted backups/HTTPS/mobile monitoring are later phases; this PR establishes durable storage, process recovery and per-sport health, not a claim that those later phases are already deployed.

## Limits and next phase

The host clock is part of the receipt trust boundary; keep it synchronized. Local ownership can alter/drop SQLite triggers, so append-only checks/hashes are audit aids, not a tamper-proof external timestamping service. Keep independent backups. Model/source artifacts are retained and must be backed up with the journal. Raw and model archives grow; no historical forecasts are deleted automatically.

Long provider outages or unavailable approvals may leave stale forecasts; those are labeled, and no replacement market probability is inserted. Source vintages, conservative feature availability and unknown confirmed starters remain documented model limitations. This phase does not fabricate a performance record. Phase2 will grade the first eligible receipts, explicitly handle final/void/tie/disputed states and publish separate prospective metrics while preserving retrospective reports.
