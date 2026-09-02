# v2.17.1 — Adaptive player role-state

This patch keeps the v2.17 team structural calibration intact and addresses one
specific player-prop failure mode: a genuine within-season role change can remain
too anchored to the fixed Old/G6-10/L5 55/20/25 opportunity profile when there is
no confirmed teammate OUT event to activate the availability bridge.

## What changes

A new `core/player_role_state.py` layer is applied to **opportunity rates only**:

- 2PA/min
- 3PA/min
- FTA/min
- REB/min
- AST/min

Minutes remain controlled by the minutes engine / trader override. Shooting skill
(3P%, 2P%, FT%) still uses the larger regularized sample and is **not** made dynamic
by this patch. H2H, opponent-by-position and availability-state logic remain
separate, avoiding double counting.

## Statistical model

For each game `t`, opportunity counts `C_t` in `M_t` minutes are treated as a
Poisson-rate time series. A local-level state-space approximation is fit on the
log-rate scale:

```
z_t = log((C_t + 0.5) / M_t)
z_t ~ Normal(theta_t, 1 / (C_t + 0.5))
theta_t = theta_(t-1) + eta_t
eta_t ~ Normal(0, q)
```

The `+0.5` is a Jeffreys-style small-count stabilization. The process variance
`q` is **not hard-coded**. A data-scaled candidate grid is evaluated by one-step-
ahead walk-forward predictive likelihood. `q=0` is always included as the static
model.

The dynamic model has one extra parameter (`q`). To stop a noisy recent stretch
from automatically overriding the longer history, v2.17.1 uses Akaike model
averaging between:

- static latent state (`q=0`), and
- best walk-forward dynamic state (`q>0`).

The resulting dynamic/static latent-rate ratio is applied as a modifier to the
existing non-overlapping Old/G6-10/L5 baseline. Therefore:

- stable role -> modifier stays close to 1.00;
- sustained role change -> recent latent state receives more influence;
- no arbitrary `L5 = 45%` switch is needed;
- the old bucket model remains the stable prior rather than being discarded.

Availability-similarity game weights are reused as information weights inside the
state-space model, so a confirmed OUT state is not counted twice.

## Why this is literature-backed

The architecture follows the state-space / dynamic generalized linear model
tradition for non-Gaussian count processes and time-varying latent parameters:

1. West, M., Harrison, P. J., & Migon, H. S. (1985), *Dynamic Generalized Linear
   Models and Bayesian Forecasting*, Journal of the American Statistical
   Association, 80(389), 73-83. DOI: 10.1080/01621459.1985.10477131.
2. Yamada, K. & Fujii, K. (2025), *Estimation of temporal changes in basketball
   players' contribution and compatibility based on a state-space model*, JSAI
   2025. DOI: 10.11517/pjsai.JSAI2025.0_2E1GS1003.
3. Macrì-Demartino, R., Egidi, L., & Torelli, N. (2026), *Bayesian dynamic
   Bradley-Terry model with commensurate spike-and-slab priors*, Journal of Big
   Data. DOI: 10.1186/s40537-026-01486-6. The key principle used here is adaptive
   temporal borrowing: strong shrinkage when the state is stable, weaker borrowing
   when the data support a change.
4. Sandholtz, N. & Bornn, L. (2018), *Markov Decision Processes with Dynamic
   Transition Probabilities: An Analysis of Shooting Strategies in Basketball*.
   Their basketball framework explicitly models non-stationary transition behavior
   and borrows strength across players and through time.
5. Terner, Z. & Franks, A. (2021), *Modeling Player and Team Performance in
   Basketball*, Annual Review of Statistics and Its Application 8:1-23, for the
   broader basketball-analytics role of hierarchical regularization and
   spatio-temporal modeling.

## Why this should fix the Citron / Austin diagnostic

The old player model always gave 55% weight to the `old` bucket unless the trader
explicitly selected a role-change regime. That is appropriate for stationary
players but can lag a tactical role change.

For a Citron-like synthetic sequence matching the observed audit pattern (old AST
rate near 0.11/min, G6-10 around 0.135/min, L5 around 0.22/min), the v2.17.1 test
moves the opportunity profile materially above the fixed 55/20/25 baseline while
leaving shooting percentages unchanged. The exact live output remains data-driven
and should be checked in the new **Adaptive player role-state audit**.

## New audit

Player deep-dive -> `Model audit` now shows an **Adaptive player role-state audit**
with:

- static latent rate
- dynamic latent rate
- applied profile rate
- applied modifier
- selected process variance `q`
- static and dynamic walk-forward predictive NLL
- dynamic model weight
- state-change z-score
- active / neutral flag

This makes the correction fully inspectable rather than hidden.

## Files changed

Required for GitHub upload:

- `core/player_role_state.py` **NEW**
- `core/player_model.py` **REPLACE**
- `streamlit_app.py` **REPLACE**

Recommended:

- `tests/v2171_player_role_state_smoke.py` **NEW**
- `README_v2_17_1_adaptive_player_role_state.md` **NEW**

No change is required to team-market calibration, pace, matchup, minutes, provider,
or league config files for this patch.
