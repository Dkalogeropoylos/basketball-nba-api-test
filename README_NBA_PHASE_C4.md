# NBA V2 Phase C4 — Structural variance A/B

This phase does **not** change any mean-model coefficient, opponent beta, H2H shrinkage, pace weight, shooting calibration, roster logic, or the 2025-26 final test.

## Scientific design

- League/mean architecture: frozen from the existing 2023-24 TRAIN calibration.
- 2024-25 eligible games: split chronologically in half.
- First half (497 of 995 eligible games): development/calibration only.
- Last half (498 games): untouched A/B evaluation.
- 2025-26: still locked.

## B1 — FGA possession/rebound chain

Frozen C3 baseline:

`FGA ~ Poisson(FGA_mean)` after pace, TOV, FTA and rebound state already made `FGA_mean` random.

C4 B1 removes this additional full-count Poisson redraw. It simulates:

1. initial shot endings from `POSS - TOV - 0.44*FTA`;
2. shot type and make/miss;
3. OREB from realized misses;
4. each realized OREB creates one continuation shot opportunity;
5. repeat until the rebound chain ends.

This is the event-chain interpretation of the standard possession identity

`POSS ≈ FGA - OREB + TOV + 0.44*FTA`.

No B1 variance parameter is fitted.

## B2 — FTA mean-preserving overdispersion

The existing FTA process is already Poisson-lognormal. C4 keeps that family and fits only the latent log-rate sigma using early-half method of moments:

`lambda = mu * exp(sigma*z - sigma^2/2)` with `z~N(0,1)`.

The correction term `-sigma^2/2` keeps the multiplicative shock mean at 1.

Early-half C3 audit (994 team rows):

- old sigma = 0.120000
- FTA bias = +0.278051 (projection - actual)
- residual variance = 42.683152
- mean C3 MC variance = 29.142369
- mean projection^2 = 471.672857
- fitted sigma = **0.205681**

Approximate MoM equation:

`exp(s1^2) = exp(s0^2) + (V_obs - V_mc) / E[mu^2]`

The fitted sigma is judged only on the untouched final 498 games.

## Acceptance criteria

Do not accept a variant because one coverage number improves. Require jointly:

- centers (Bias/MAE/RMSE) not materially worse;
- 50/80/90 coverage closer to nominal;
- lower CRPS;
- reliability/probability bins closer to actual frequencies;
- no material damage to unrelated control markets (especially TOV/OREB/STL/BLK);
- zero technical failures.

Use B1 and B2 separately first. B3 (combined) is only a final interaction check.

## Statistical / basketball basis

- Kubatko, Oliver, Pelton & Rosenbaum (2007), *A Starting Point for Analyzing Basketball Statistics*, Journal of Quantitative Analysis in Sports, DOI 10.2202/1559-0410.1070 — possession-based accounting as the natural unit for basketball box-score analysis.
- Gneiting, Balabdaoui & Raftery (2007), *Probabilistic Forecasts, Calibration and Sharpness*, JRSS B, DOI 10.1111/j.1467-9868.2007.00587.x — maximize sharpness subject to calibration; use predictive-distribution diagnostics and proper scores.
- Cameron & Trivedi (1986), *Econometric Models Based on Count Data*, Journal of Applied Econometrics, DOI 10.1002/jae.3950010104 — Poisson assumptions, dispersion diagnostics, and richer count-data models.
- Aitchison & Ho (1989), *The Multivariate Poisson-Log Normal Distribution*, Biometrika, DOI 10.1093/biomet/76.4.643 — latent lognormal Poisson mixtures support overdispersion and dependence in multivariate counts.
