# Basketball Pricing Engine v2.16.0 patch

Apply on top of v2.15.0 / v2.15.1. The v2.15.1 stale Streamlit state fix is retained in `streamlit_app.py`.

## 1) Player near-state: relevant OUT set by player + stat

The player near-state no longer conditions equally on every selected OUT.

For each focal player/stat, v2.16 scores current OUTs using event volume and stat-specific position/role compatibility, removes weak tail absences, then renormalizes the state over the materially relevant OUTs. The audit now shows:

- `Relevant OUT state`
- `Excluded low-relevance OUT`
- normalized `Relevance`
- `Raw relevance`

This is intended to stop a Harrison REB sample, for example, from being diluted by unrelated creator absences.

## 2) Stat-specific event redistribution

Vacated player events are no longer routed mainly through the generic minute-replacement matrix.

Routing is now:

`minute replacement relationship × stat-specific role/position priority × player event propensity`

with different role-family priorities for AST, REB, 3PA, FGA and FTA.

## 3) Sparse synthetic fallback cap

Low empirical confidence no longer hands 100% control to the synthetic player-role fallback. The residual fallback is capped at 65% structural credibility until walk-forward calibration is completed.

## 4) Current defensive roster bridge for Team Markets

For every confirmed OUT separately:

- `MIN >= 10` = ON game
- `MIN = 0` after first team appearance = OFF game
- `0 < MIN < 10` = excluded from BOTH ON and OFF groups
- current H2H opponent is excluded to keep the explicit H2H layer disjoint

The engine compares opponent offensive outcomes ON vs OFF, partial-pools each split, and combines multiple current OUTs with overlap protection. The bridge affects only the opponent offense and is stat-specific for 3P share, FTA, TOV, OREB, AST, 3P%, and 2P%.

The Streamlit model audit shows the individual ON/OFF opponent PTS/100 splits plus the final combined modifiers.

## 5) Possession-consistent Team FGA chain

The old chain generated FGA from `(POSS - TOV)` and then separately generated FTA and OREB. Because the historical possession estimator is:

`POSS = FGA - OREB + TOV + 0.44*FTA`

that could produce too few FGA for the stated pace.

v2.16 now uses:

`POSS -> TOV + FTA possession component -> initial shot endings -> OREB recycling -> total FGA -> 3P share`

so in expectation the simulated box score reconciles back to the shared possession state.

A new `Possession identity audit` reports:

`FGA - OREB + TOV + 0.44*FTA - Sim POSS`

The target is approximately zero.

## Important

The new structural rules are not yet walk-forward calibrated. Re-test Team Markets and Player Props before treating fair odds as calibrated true probabilities.
