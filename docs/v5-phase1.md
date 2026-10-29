# Version 5.0, Phase 1 — understandable predictions

Implements Phase A of [Issue #6](https://github.com/jeeb-ai/predictions/issues/6). Cards explain the pick in everyday language, followed by three strongest supporting contributions (when available), up to two opposing contributions, and an uncertainty explanation. The wording comes from the recorded logistic contribution, its actual sign and the recorded team-side statistic; it is deterministic, not generated injury/news commentary. A statistic can favor a team in the fitted model without proving a causal advantage. Neutral-default inputs and QB proxies remain labeled.

The summary first reconciles all contributions and the intercept with the recorded probability. Ranking flips direction for away picks. A counterweight to the winner can differ from the statistic that looks best in isolation because regression coefficients reflect the other included inputs. EPA is defined as estimated scoring efficiency on each play, rather than assuming readers understand the acronym. Exact contributions and source values remain under a collapsed Advanced Analysis section. Date, team search and sport filtering keep the slate navigable; NFL is the only available sport until other models are validated. Streamlit's native responsive layout avoids a wide statistics table on the homepage.

Confidence wording reports the opponent's chance, model-refit interval and whether it crosses 50%. It does not map a narrow interval or a large win probability to proven accuracy. Missing explanations or fewer than three positive contributions are labeled rather than filled with invented facts. Scores/spreads/totals stay explicitly unavailable pending Phase 2.

## Update and run on Mac (Python 3.12)

After this branch merges, from your existing checkout:

```bash
git switch main
git pull --ff-only
uv sync --frozen --extra dev --python 3.12
uv run --frozen nfl-predict research-init --config research.local.json
uv run --frozen nfl-predict research-run --config research.local.json
uv run --frozen nfl-predict dashboard --config research.local.json
```

To review before merge, fetch and switch to `feat/v5-plain-english` instead. Initialization preserves your existing private key and policy. Use that local key to unlock the dashboard; never paste it in chat or commit it. Keep the dashboard private. Run `research-worker` in another terminal for repeated refreshes. No migration rewrites archived forecasts or the paper ledger. Full historical comparison remains available using the README's `compare` command. New explanations do not change forecast probabilities or historical evaluation.

Actual UI rendering evidence and validation results accompany the PR. No scoring performance improvement is claimed by this presentation-only phase. Phase 2 must independently validate score distributions before any score or spread/total edge is displayed; NBA and NHL follow validated NFL scoring.
