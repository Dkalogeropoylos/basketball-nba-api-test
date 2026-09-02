# v2.17.2 — Disjoint player H2H + calibrated team shooting efficiency

This patch keeps the v2.17 possession / team structural model and the v2.17.1 adaptive player-role state, while fixing two remaining failure modes.

## 1. Player H2H is now genuinely disjoint

Before v2.17.2, the same current-opponent games could influence the player's Old/G6-10/L5 / adaptive role-state and then appear again in the explicit H2H modifier. The second effect was tiny, but it was still conceptual overlap.

v2.17.2 now:

1. splits the real Old / G6-10 / L5 timeline first;
2. removes current-opponent H2H rows from **opportunity** buckets only;
3. fits adaptive role-state on the non-H2H opportunity history;
4. keeps the full history for 3P% / 2P% / FT% shooting skill;
5. sends the excluded H2H games only to the explicit H2H layer.

The audit adds `raw_bucket_games` and `H2H_excluded`.

### No more universal 5% H2H cap

For 2PA/min, 3PA/min, REB/min and AST/min, v2.17.2 learns a stat-specific empirical-Bayes pooling mass from historical repeat player/opponent matchups.

For each stat, the model asks whether previous H2H performance improves prediction of the **next** H2H game relative to the player's non-H2H baseline. The pooling strength is expressed as prior-equivalent player minutes `K`:

```
posterior_rate = (relevant_H2H_count + K * non_H2H_rate)
                 / (relevant_H2H_minutes + K)
```

`K` is selected from a data-scaled grid by next-H2H Poisson predictive likelihood. `K = infinity` is always a candidate, meaning the data can decide that H2H should contribute zero for a stat.

Current rotation similarity and minute similarity reduce the **effective H2H minutes**, rather than being multiplied into another arbitrary blend cap.

This is a standard empirical-Bayes / Gamma-Poisson partial-pooling idea: sparse pair-specific evidence is shrunk strongly, while repeat-matchup evidence can matter more when it has historically shown predictive persistence.

## 2. Team 3P% / 2P% are now attempt-weighted and held-out calibrated

v2.17 already fixed the structural chain for 3P share / FTA / TOV / AST. It intentionally left shooting efficiency very conservative. That could create a good total but an overly compressed team margin if one side's matchup shooting efficiency was not allowed to move enough.

v2.17.2 does **not** touch:

- pace / possessions;
- FGA;
- 3P share;
- 3PA / 2PA;
- FTA / TOV;
- rebounds;
- AST structure.

It changes only the conditional make probabilities.

For 3P% and 2P%, historical team-games are modeled as binomial makes given attempts:

```
makes_t ~ Binomial(attempts_t, p_t)

logit(p_t) = logit(own_offense_skill)
             + beta * [logit(opponent_allowed_skill) - logit(league_skill)]
```

Important details:

- attempts are the statistical exposure, so 2/4 is not treated like 12/30;
- offense and defense shrinkage masses are selected from chronological data, not hard-coded as the final values;
- the opponent-response coefficient `beta` is selected from chronological data;
- the learned layer activates only if it beats an own-offense-only model on later held-out games using binomial log loss;
- the live own shooting profile remains the existing large-sample regularized team skill, so this patch does not replace stable offensive skill with a second noisy estimate;
- there is still no direct H2H shooting-percentage boost. H2H shooting results remain part of general shooting history only once.

This follows the broader basketball/statistical literature preference for hierarchical/regularized shooting models and attempt-aware likelihoods rather than treating raw game FG% as equally reliable observations.

## Why the two fixes belong together

The player change prevents the same matchup games from being counted as both current role and H2H evidence.

The team change keeps the already-improved shot allocation intact while allowing opponent defense to move expected conversion only when historical held-out data says that it improves prediction.

So v2.17.2 is designed to change **where the evidence enters**, not to make the model generically more aggressive.

## New audits

### Player H2H calibration

The deep-dive audit shows:

- calibration events;
- learned prior minutes `K` by 2PA / 3PA / REB / AST;
- no-H2H predictive NLL;
- best H2H predictive NLL;
- effective H2H minutes for the selected player;
- posterior H2H weight;
- final posterior rate and modifier.

### Team shooting-efficiency calibration

The team audit shows:

- whether 3P% / 2P% calibration is active;
- held-out binomial NLL versus own-only baseline;
- learned defense beta;
- learned attempt shrinkage masses;
- opponent allowed posterior percentage;
- target percentage and applied modifier for each team.

## Files changed

Required for GitHub upload:

- `core/player_model.py` — REPLACE
- `core/matchup.py` — REPLACE
- `core/shooting_efficiency.py` — NEW
- `streamlit_app.py` — REPLACE

Recommended:

- `tests/v2172_disjoint_h2h_shooting_smoke.py` — NEW
- `README_v2_17_2_disjoint_h2h_shooting_efficiency.md` — NEW

No change is required to `player_role_state.py`, `team_model.py`, `structural_calibration.py`, `minutes_engine.py`, `pace_engine.py`, providers or league config.

## Regression tests

The patch was checked against the existing v2.14, v2.15, v2.16.1, v2.17 and v2.17.1 smoke tests plus the new v2.17.2 test.
