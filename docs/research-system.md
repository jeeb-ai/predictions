# Scheduled NFL research and paper trading

The research worker refreshes strictly pregame forecasts, reads public Polymarket **US** markets, records uncertainty-aware evaluations, and tracks subsequent observed outcomes. Eligible entries and settlements affect a local paper ledger only. The dashboard reads retained reports; it has no provider calls, trade controls, or ledger writes. No account credentials or exchange order endpoints are used.

## Run on a private Linux host

Use an always-on Linux host with Docker Engine, Docker Compose v2, Python 3.11/3.12, and `uv`. Run these commands from the repository checkout:

```bash
uv sync --frozen --extra dev
uv run --frozen nfl-predict research-init --config research.local.json
docker compose -f deploy/compose.yml build
docker compose -f deploy/compose.yml up -d --no-build
docker compose -f deploy/compose.yml ps
docker compose -f deploy/compose.yml exec -T worker nfl-predict research-status --config /app/research.local.json
```

`research-init` creates an ignored, secret-free policy file and a random dashboard access key. It preserves an existing configuration and key. The configuration is readable by the nonroot container; the separate key must be a regular private file with mode `0600`. Defaults use relative paths, which resolve against the configuration directory. Keep those paths for the supplied Compose deployment: it mounts persistent named volumes at `/app/data`, `/app/models`, and `/app/reports`.

The host initialization and Docker worker use **different storage**. The worker initializes its own key inside the `research-data` volume; the host's `data/private/dashboard.token` does not unlock the containerized dashboard. After the worker has initialized, retrieve the container key directly to a private file on your own computer through SSH. For example, from that computer:

```bash
umask 077
ssh USER@HOST 'cd /srv/predictions && docker compose -f deploy/compose.yml exec -T worker cat /app/data/private/dashboard.token' > "$HOME/.nfl-research-dashboard.token"
ssh -N -L 8501:127.0.0.1:8501 USER@HOST
```

Substitute the actual user, host, and checkout directory. Open the forwarded dashboard port in your local browser and use the key from that private file. Keep it in a password manager or other private local storage; do not paste it into chat, commits, shared logs, or command arguments. The Compose file publishes port 8501 only on the host loopback interface. Use SSH access or a private network; do not change that binding to a public interface. The dashboard authenticates before reading research data, rechecks the current key during refresh, and invalidates sessions after key rotation. This shared-key login is intended for private access, not public multi-user hosting.

The first Model 2.0 cycle downloads roughly 0.5 GB of historical play-by-play and builds causal EP models and a bootstrap ensemble. A cold start can take 15–30 minutes or longer depending on network and CPU; verified caches and model bundles are reused afterward. The worker health check allows a 30-minute startup grace, but `running` or `not_started` is not a successful readiness result. Inspect the retained run and health output rather than assuming a running container has completed research. The worker has no published ports; the dashboard mounts research volumes read-only.

If the host requires an HTTPS proxy with a private CA, retain certificate verification and provide its public CA certificate to the build:

```bash
export NFL_PROXY_CA_FILE=/path/to/trusted-proxy-ca.crt
docker compose -f deploy/compose.yml build --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY
```

The ordinary build needs no additional CA. Compose forwards the host's proxy variables to the worker when present. Never disable TLS verification to make a download succeed.

## Configure an explicit paper policy

`research.example.json` documents the checked-in defaults. The worker polls 15-minute UTC buckets, forecasts games up to seven days ahead, and uses Model 2.0 with 100 training-week bootstrap fits and 95% intervals. Default limits are ten contracts per simulated entry, $100 outstanding paper cost, one open position per game, and a lower EV threshold of $0.02 per contract. Quotes must be no older than 60 seconds and forecasts no older than one hour when evaluated. These settings are policy, not evidence that a strategy is suitable.

The three assumption ranges default to `null`:

| Field under `paper.assumptions` | Meaning |
| --- | --- |
| `tie_probability_range` | Probability of a tie conditional on game completion. |
| `completion_probability_range` | Probability the contract settles through a completed game. |
| `cancellation_settlement_price_range` | **YES** contract's last-fair-market-price payout if the game does not complete. |

Supply two-element decimal ranges satisfying `0 <= lower <= upper <= 1` only when you have a documented basis. The model estimates decisive-game win probabilities; it does not estimate these event risks. Missing required ranges produce a recorded **PASS**. No automatic entry is possible under the default missing assumptions. Setting `paper.enabled` to `false` keeps evaluations and tracking while disabling new automatic paper entries; existing paper positions can still receive evidenced settlements.

For scheduled evaluations, the NO cancellation payout range is `[1 - YES_upper, 1 - YES_lower]`. This convention is specific to research configuration; the standalone `ev --cancellation-settlement-price` argument continues to describe the selected contract's payout. Range midpoints supply the scheduled representative point calculation and are recorded as policy assumptions, not empirical estimates.

After editing the policy file, recreate the worker so it loads the change:

```bash
docker compose -f deploy/compose.yml up -d --no-build --force-recreate worker
```

Each report retains the normalized policy and its hash. A change creates a distinct scheduled job identity. Keep JSON free of secret values. The default research ledger, `data/paper/research.sqlite`, is separate from the standalone Model 3.0 ledger.

## Interpret uncertainty and evaluations

The live ensemble resamples whole season/week blocks from exactly the eligible older-season training rows, then refits the scaler and fixed logistic regression. Target-season labels never enter fitting. Forecast intervals are bound to the model, training cutoff, training IDs, feature hash, game, and forecast cutoff; corrupt or mismatched provenance fails validation. Standalone forecasts remain point-only unless `--bootstrap-samples` is requested. Historical backtests and comparison defaults remain unchanged.

These percentile intervals describe **model-estimation uncertainty conditional on the constructed features**. They do not refit EPA/feature construction, model upcoming injuries or starters, quantify source revisions, or guarantee calibrated coverage of future probabilities or outcomes. The point model remains uncalibrated.

Each market evaluation uses actual retained US ask depth, the documented fee formula with conservative fill-rounding costs, and observed timestamps. Worst and best EV are computed over all corners of the selected team's model interval and the explicit tie/completion/cancellation ranges. An automatic paper candidate requires the ordinary validation gates and lower EV strictly above the configured threshold. A positive lower EV under these inputs does not establish a profitable strategy or protect against model error outside the interval.

The worker evaluates YES and NO, retains their pass reasons, and selects the eligible candidate with the highest lower EV for a game. Transactional game/exposure/depth limits and deterministic position keys prevent repeated polls from repeatedly filling the same scheduled position. Settlement requires the exchange's finally resolved evidence and matching payout; unresolved positions remain pending. No outcome, fill, refund, or profit is fabricated when an upstream observation is unavailable.

## Scheduling, evidence, and recovery

The worker holds a Linux process lock to prevent overlapping cycles. A durable SQLite job store identifies each job by UTC interval bucket and configuration hash. Completed jobs are skipped only when their retained report matches; incomplete attempts can retry. Still-fresh forecast checkpoints may be reused, while quotes are fetched again. Restarted workers recover abandoned attempts and ledger idempotency prevents duplicate fills or settlements. Scheduling waits are interruptible; Compose restarts the worker after a process or host restart.

Each cycle retains an immutable `data/research/runs/<run-id>/run.json`, updates `latest.json`, and preserves `last_success.json`. Raw source snapshots, model checksums, forecast provenance, evaluations, paper entries, settlement evidence, and ledger audits remain available for review. Failed provider operations appear as stage errors and `partial` or `failed` status; a later successful stage does not conceal them. The ledger is audited before writes and again at the end. Retain its audit head separately if you need to detect replacement or truncation of the entire database.

The dashboard refreshes authenticated views every 30 seconds and shows run health, forecasts and intervals, market evaluations, paper positions, settlement PnL, calibration, and audit status. Display times use America/Chicago; persisted evidence uses UTC. Stale data and failed runs remain visibly labeled. The HTTP dashboard health check establishes that the UI is serving, while `research-status --healthcheck` checks the latest worker result and age. Inspect `status` and `errors` as well: a fresh partial run is considered process-healthy but still has unfinished work.

Back up all three named volumes together, the policy file, and an external audit head. Stop the worker for a consistent filesystem backup or use SQLite's supported backup mechanism; copying a live SQLite file alone can omit WAL contents. Treat backups as private because they contain the dashboard key and research history. Do not use `docker compose down -v` unless intentionally deleting all stored evidence. `docker compose down` preserves named volumes; `up -d` resumes the service. Pin and review dependency/image updates before rebuilding.

For development outside Docker:

```bash
uv run --frozen nfl-predict research-run --config research.local.json
uv run --frozen nfl-predict research-worker --config research.local.json --max-cycles 2
uv run --frozen nfl-predict dashboard --config research.local.json --port 8501
```

The development dashboard also binds to loopback and uses the host token. A development process or cloud workspace does not establish perpetual hosting. Deploy these services on the chosen persistent Linux host for unattended operation; filesystem snapshots do not preserve running processes.

## Automatic prospective tracking

The performance tracker selects the first actually generated and ledger-recorded **pregame** forecast per game and model version. Repeated polling cannot increase the game count. Outcomes must be known by the report cutoff and satisfy the source's conservative result-availability rules. Team mismatches, unresolved games, and ties are excluded from binary scoring. Source corrections append observations and new reports; earlier reports remain immutable, and a stale snapshot cannot overwrite newer observed evidence.

Once scored games exist, reports include accuracy, Brier score, log loss, ten-bin reliability, and expected calibration error. No scored games means `metrics: null` and `no_scored_games`, never a made-up zero score. Sampling intervals resample season/week blocks of fixed recorded forecasts (1,000 draws, seed 42). Fewer than two observed blocks produce no sampling interval; these intervals omit model refitting and cross-week dependence. A short prospective sample cannot establish calibration or improvement over a baseline.

Paper tracking reports outstanding cost, actual evidenced simulated settlement profit, its recorded timeline, and realized drawdown. It does not invent current marks for open positions. Fees remain conservative estimates rather than exchange executions. Paper PnL is not real-money performance. Historical Model 1.0/2.0 reports remain the historical baselines; this system begins a new prospective record and makes no new improvement claim.
