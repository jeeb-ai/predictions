# Live NFL inference and an auditable paper workflow

This release adds prospective forecasts and local paper positions while preserving both historical model families, evaluation splits, feature builders, comparison reports, and archived results. Public data reads are the only network operations added. There is no exchange account client, order preview, signing key, order submission, position closing, or real-money execution method.

## Information available at prediction time

`forecast` reads a public nflverse schedule and, for Model 2.0, current-season play-by-play plus retained historical snapshots. Online mode refreshes the schedule and current season; `--offline` explicitly uses cached data. Every result used in form, Elo, QB history, and causal EPA aggregates must be available **strictly before the forecast's as-of time**, not merely before a future kickoff. A cached potentially live partial game cannot become a final result simply by waiting: snapshot observation times also mask completion. Rest days use the known scheduled kickoff.

The annual live model uses the existing scaler and fixed logistic regression, trained on older-season decisive regular-season games available before the earlier of the season opener and forecast time. A 2026 model can train through 2025; this does not change the archived 2025 holdout experiment. Current-season outcomes never fit the live model's coefficients. Later available outcomes can update pregame form. The QB proxy remains the last previously observed QB, without injury/depth-chart or target-starter knowledge.

Models and forecast batches have unique immutable directories, hashes, feature versions, exact training IDs/cutoffs, raw source provenance, and actual generation timestamps. The forecast rejects future-observed inputs, unknown target kickoff times, started/finished games, mismatched model/feature versions, and corrupt model files. A backdated as-of argument cannot turn a postgame computation into a live forecast. Only trusted local joblib bundles should be loaded; checksums do not make an untrusted pickle safe.

Probabilities remain raw logistic outputs **conditional on a decisive result**. They do not include a learned tie/cancellation probability or a calibrated uncertainty interval. The historical Model 2.0 comparison did not establish 2025 holdout superiority or better calibration. Do not interpret a point probability or estimated EV as prospective profit evidence.

## Commands

Install with `uv sync --frozen --extra dev`. For the smoke-check game, before kickoff:

```bash
uv run --frozen nfl-predict forecast --game-id 2026_08_CAR_GB > /tmp/nfl-forecasts.json
uv run --frozen nfl-predict us-markets --limit 20
uv run --frozen nfl-predict quote --market aec-nfl-car-gb-2026-10-29 --outcome no --output reports/quotes/car-gb-no.json
uv run --frozen nfl-predict ev --forecast /tmp/nfl-forecasts.json --quote reports/quotes/car-gb-no.json --team GB --quantity 10 --output reports/ev/car-gb.json
uv run --frozen nfl-predict paper-audit
```

Game IDs and dated market slugs must refer to an upcoming game when used. Quote/EV output files use exclusive creation; choose a new filename for each retained observation. Standard output is also JSON. Forecast batches expose their immutable `artifact.path`; a batch with multiple forecasts requires `ev --game-id` to select exactly one.

Optional forecast flags include `--model-version v1`, `--bundle PATH`, `--input PATH`, `--offline`, `--window`, `--workers`, `--model-dir`, `--output-dir`, `--pbp-dir`, `--ep-dir`, and a timezone-aware `--as-of`. Every paper command accepts `--ledger PATH`; the default separate database is `data/paper/ledger.sqlite`. Backtest databases are not reused. Missing offline inputs fail instead of making a network request.

## Market identity, depth, settlement, and fees

The [verified API reference](polymarket-us-reference.md) documents the exact public US gateway and official sources. No API key is needed. All adapter calls are unauthenticated GETs to `gateway.polymarket.us`; redirects are rejected. Discovery, details, book depth, and final settlement responses are retained byte-for-byte with source URLs and SHA-256 checksums.

Only documented full-game NFL winner contracts with overtime, two explicit NFL teams, USD prices, known fees, and unambiguous rules are normalized. YES may be the away team. YES asks come from offers; NO asks are complementary YES bids, retaining their available quantities. Matching requires identical canonical team identities and exact kickoff timestamps; titles and midpoints are not a mapping or executable price.

NFL ties pay **$0.50 per contract**, while cancellations or postponements can pay the last fair market price specified in that contract's terms. These are not automatic refunds. The actual description is retained and hashed, including its postponement window. Finally resolved exchange metadata is required before `market-settlement` returns a settlement price; closed or provisional status is insufficient.

Standard taker fees use the verified US formula `0.0695 × quantity × price × (1 − price)`, effective October 19, 2026 at 1 PM Eastern. Cent rounding is half-even. The exchange caps commission across fills at the rounded cumulative exact fee. Level-two depth hides individual maker-order partitions, so EV and simulated paper cost use that conservative upper bound and separately label a per-displayed-level estimate. No rebates are assumed. Unknown fee coefficients, unsupported terms, malformed prices, or invalid timestamps fail or produce a pass.

## EV and explicit assumptions

The example EV command intentionally produces a pass unless explicit outcome assumptions are supplied:

- `--tie-probability`: tie probability conditional on an officially completed game.
- `--completion-probability`: probability the game completes under the contract's normal rules.
- `--cancellation-settlement-price`: assumed contract payout when completion probability is below one.

These are user-supplied assumptions, not estimates learned by this package. No zero-tie or certain-completion default is invented. The completed-game payout per contract is `(1 − tie_probability) × decisive_win_probability + 0.5 × tie_probability`. Completion-weighted EV additionally includes the explicit cancellation payout, then deducts every consumed ask level and the conservative fee bound. Cash calculations use Decimal; the selected team's probability is matched to the quoted YES/NO side.

Default policies require quotes no older than 60 seconds and forecasts no older than one hour; future observations and already-started games are rejected. The policies can be supplied explicitly with `--max-quote-age-seconds` and `--max-forecast-age-seconds`. Entry must have enough depth for the entire requested integer quantity, meet the market minimum and tick size, and exceed `--min-ev-per-contract` after fees. A pass remains auditable and never becomes a simulated fill. Results preserve inputs, assumptions, reasons, fills, timestamps, and evidence; a positive result is labeled `paper_candidate` only.

## Paper positions, settlement, and audit

`forecast`, `quote`, and `ev` record immutable observations/evaluations but create no positions. Explicitly run `paper-fill --evaluation-id EV_ID --idempotency-key UNIQUE_KEY` only for a candidate selected for paper research. `--max-exposure` optionally sets a maximum unsettled paper cost; no bankroll is invented. Fill time is rechecked against kickoff and input freshness. Evaluations are recalculated from linked records so editing a candidate flag, fill list, cost, or fees cannot bypass validation.

SQLite transactions reserve displayed quantities by the underlying book hash, market, outcome, and price. Repeating a quote request does not replenish the same snapshot's paper liquidity. This is a simulation of retained displayed depth, not a promise that real liquidity would still be executable. Reusing an idempotency key repeats the original action; conflicting contents are rejected. Transactions prevent concurrent overfill. No paper fill sends a network request.

After the exchange finally resolves the market, read its settlement:

```bash
uv run --frozen nfl-predict market-settlement --market MARKET_SLUG --output reports/settlements/final.json
uv run --frozen nfl-predict paper-settle --trade-id PAPER_TRADE_ID --price ACTUAL_PAYOUT --evidence reports/settlements/final.json --idempotency-key SETTLEMENT_KEY
uv run --frozen nfl-predict paper-audit
```

Substitute IDs and the actual chosen side's payout from the returned settlement object. The ledger verifies market identity, resolved status, raw response hashes, and the selected YES/NO payout. Operator-supplied evidence must include a reference, observed-at timestamp, and explicit price; it is labeled separately from a retained public US settlement snapshot. A declared tie must pay 0.50, a win 1, and a loss 0; cancellation has an explicit LFMP payout. No result is inferred or fabricated for an unresolved game. Settlements append once, calculate `quantity × actual_payout − simulated_entry_cost`, and retain the conservative estimated fee label.

All entity records and audit events have SQL UPDATE/DELETE guards, foreign keys, canonical payload hashes, and a monotonic hash chain. Retained provider quotes and settlement evidence are reconciled against their raw responses. Audit checks evidence, linkage, regenerated calculations, and the current head; exit status is nonzero on failure. Save the latest trusted head separately and use `paper-audit --expected-head-hash` to check that checkpoint. A local database owner who can replace the whole file can rewrite an unanchored chain; append-only triggers are not an external notarization service. Data, models, and SQLite files remain ignored by Git.

## Validation boundary

Automated tests use synthetic schedules/books and mocked public GETs, including the full forecast → quote → EV → simulated fill → settlement → audit flow. They test leakage, orientation, fee examples and rounding, fractional depth, stale/future observations, missing assumptions, idempotency, concurrency, direct SQL write rejection, tampering, and fractional/tie settlement. Network write methods are forbidden in the integration fixture.

The actual read-only smoke check validates an upcoming NFL forecast, an official US book with exact teams/kickoff, a missing-assumptions pass, and a valid ledger containing **zero paper positions and zero settlements**. Synthetic test profits are never presented as real or prospective results. The [execution report](live-paper-validation.json) records measured checks and current observations separately from fixture outcomes.
