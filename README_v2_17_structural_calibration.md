# Basketball Pricing Engine v2.17 — Structural Calibration

## Why this version exists

v2.16.1 could remain too close to team averages in several structural markets even when repeat-matchup / opponent evidence consistently pointed in another direction. The main issue was not necessarily game pace; it was the allocation of possessions into turnovers, free throws, field-goal attempts and shot mix, plus the assisted-make rate.

v2.17 therefore changes **only the main Team-Market structural-rate layer** first:

- `3P_SHARE = 3PA / FGA`
- `FTA_RATE = FTA / possessions`
- `TOV_RATE = TOV / possessions`
- `AST_PER_MAKE = AST / FGM` for WNBA/NBA

The possession simulator, roster bridge, defensive-OUT bridge, location layer, shooting-efficiency shrinkage and player-prop engine remain intact.

## What changed

### 1. No fixed opponent/H2H weight for the four structural rates

For every historical team-game the new calibrator constructs **pregame-only** inputs:

- own `Old / G6-10 / L5`, non-overlapping;
- opponent-allowed `Old / G6-10 / L5`;
- same-season H2H, kept disjoint from the two baselines;
- league baseline available before that game.

Current H2H rotation similarity attenuates the H2H signal, but the numerical response to H2H is learned from league history rather than fixed at 5%, 8% or 10%.

### 2. Coefficients are data-learned

Targets are learned in transformed space:

- logit: `3P_SHARE`, `TOV_RATE`, `AST_PER_MAKE`
- log: `FTA_RATE`

A ridge model is fitted league-wide. The ridge penalty is selected by expanding-window validation from a broad logarithmic candidate grid.

### 3. Model only activates when it wins out of sample

For each structural rate, v2.17 compares walk-forward RMSE against the existing stable Old/G6-10/L5 baseline. If the learned model does **not** improve unseen-game RMSE, that rate falls back to the existing baseline rather than forcing a new model.

This is deliberate protection against changing the engine because one matchup looks unusual.

### 4. No second hand-picked cap after learning

The old combined context caps for `3P_SHARE`, `FTA`, `TOV`, and `AST` are removed. Physical/statistical bounds are enforced inside the simulator instead:

- 3P share stays in the simulator's feasible probability range;
- TOV probability stays in its feasible range;
- AST probability stays in its feasible range;
- FTA/possession receives a direct feasible-rate bound before Poisson sampling.

Roster, current defensive-OUT and location modifiers remain separate and auditable.

### 5. H2H double counting removed

The fixed `h2h_profile_blend()` no longer blends `three_share`, `fta_pp`, `tov_pp`, or `assist_per_make` in v2.17. Those four H2H signals are handled only by the new walk-forward structural model.

Other team statistics can continue using the old disjoint H2H layer for now.

## Important assist-stat rule

The current activated app is WNBA.

For **WNBA/NBA official statistics**, team assists are modeled as assists on made field goals (`AST / FGM`). The NBA rule language defines an assist as the last pass leading directly to a made field goal, and WNBA Stats publishes assisted-field-goal measures (`FGM %AST`, `2FGM %AST`, `3FGM %AST`).

**FIBA / EuroLeague statistics use a different convention**: scoring includes free throws, and a pass leading directly to a shooting foul can earn an assist if the receiver makes at least one free throw. The league configuration now records this distinction explicitly. EuroLeague/EuroCup are not yet activated, and proper FIBA free-throw-assist modeling will require play-by-play foul/pass events rather than inferring it from box-score FTA.

## Audit panels

The Streamlit Team Markets tab now exposes:

- structural model activation by market;
- training-row count;
- selected ridge lambda;
- walk-forward RMSE vs existing baseline RMSE;
- learned coefficient table;
- current matchup predicted structural rate and resulting modifier;
- current H2H sample and rotation similarity.

## New file

`core/structural_calibration.py`

## Validation

A new smoke test (`tests/v217_structural_smoke.py`) builds synthetic league history with known opponent effects and verifies that:

- structural models learn from walk-forward history;
- the learned layer can move the current matchup materially;
- the four structural rates are skipped by the old fixed H2H blend.
