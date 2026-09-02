# Basketball Pricing Engine v2.18.2

## Consolidated patch: OT/DREB fixes + continuous residual player H2H

This release is designed to be installed **directly on top of v2.18.0**. It already includes all v2.18.1 changes, so there is no need to install v2.18.1 first.

## What is included

### A. v2.18.1 team-market fixes retained

1. **Regulation-equivalent OT exposure**
   - Count/exposure quantities from overtime games are converted to a 40-minute regulation equivalent.
   - For one overtime, the exposure factor is `40 / 45`.
   - Opportunity/rate quantities such as 3P share, TOV/POSS, OREB/miss and DREB/chance are not mechanically scaled.
   - Historical player minutes are likewise converted to regulation-equivalent minutes instead of using the old fixed `OT weight = 0.78` penalty.

2. **DREB capture in the residualized team-H2H structural layer**
   - DREB capture is treated as an opportunity rate based on opponent missed-shot/rebound chances.
   - H2H can affect DREB only through a residual matchup term after the own-state and opponent-state expectation has been formed.
   - Raw team REB does not receive a second independent H2H boost.

### B. v2.18.2 player-H2H correction

The v2.17.3/v2.18.0 player H2H architecture correctly residualized H2H but then used a **binary held-out gate**:

`held-out residual model wins -> keep H2H`

`held-out residual model loses -> H2H modifier = 1.00`

That was too aggressive. It removed pair information rather than merely shrinking uncertain pair information.

v2.18.2 keeps the residualization but removes the hard drop.

For each primitive player opportunity statistic:

- 2PA/min
- 3PA/min
- FTA/min
- REB/min
- AST/min

historical H2H is first compared with what would already have been expected from:

`historical non-pair player rate × historical leave-pair-out opponent modifier`

Thus the H2H layer contains only the **player × opponent residual**, avoiding double counting of the generic opponent effect.

### Gamma-Poisson partial pooling

For a residual H2H sample:

`full_EB_modifier = (observed_effective_events + K) / (expected_effective_events + K)`

`K` is selected on the earlier chronological calibration block. It is measured in expected-event mass, so small samples are shrunk strongly toward 1.00.

Minute relevance `tau` is also learned from historical prediction performance. `tau = inf` means no minute-distance penalty.

### Continuous predictive shrinkage instead of ON/OFF

The later chronological holdout no longer decides whether H2H exists. Instead it determines a continuous predictive model weight:

`w_pred = L_H2H / (L_H2H + L_noH2H)`

which is equivalent to:

`w_pred = sigmoid(NLL_noH2H - NLL_H2H)`

The final multiplier is geometrically shrunk toward neutral:

`final_H2H_modifier = exp(w_pred × log(full_EB_modifier))`

Consequences:

- if residual H2H predicts much better, `w_pred -> 1`;
- if both models are similar, both receive material weight;
- if residual H2H predicts worse, `w_pred` becomes small but the pair signal is not abruptly deleted;
- pair sample size, rotation similarity and minute relevance still shrink the evidence before this model-averaging step.

This is deliberately different from both the old fixed 5-10% H2H weight and the later binary `Active=False` rule.

### FTA was added to the residual player-H2H layer

FTA/min now uses the same architecture as 2PA, 3PA, REB and AST and is wired into the player Monte Carlo simulation through `h2h_fta`.

No independent H2H correction is added to PTS, PRA, PA, PR or AR because those are derived outcomes and doing so would re-count component information.

## Audit changes

The player H2H audit now exposes:

- raw residual ratio;
- prior residual event mass `K`;
- learned minute-relevance `tau`;
- predictive H2H model weight;
- posterior evidence weight from the Gamma-Poisson update;
- combined applied H2H weight;
- full EB modifier before predictive model averaging;
- final residual H2H modifier.

Therefore a stat such as AST should no longer disappear merely because the later holdout is slightly worse. The audit will show **how strongly it was shrunk**.

## Statistical / literature basis

The architecture follows established ideas rather than assigning a desired projection target:

1. **Possession, pace, per-minute and rebound-rate normalization**: Kubatko, Oliver, Pelton & Rosenbaum (2007), *A Starting Point for Analyzing Basketball Statistics*, Journal of Quantitative Analysis in Sports. DOI: 10.2202/1559-0410.1070.

2. **Opponent quality should be explicitly accounted for and its influence is stat-specific**: Dong et al. (2021), *Addressing opposition quality in basketball performance evaluation*, International Journal of Performance Analysis in Sport. DOI: 10.1080/24748668.2021.1877938.

3. **Opposition interaction can materially affect technical performance**: Cui, Gonçalves & Gómez (2019), *Performance profiles and opposition interaction during game-play in elite basketball: evidences from National Basketball Association*. DOI: 10.1080/24748668.2018.1555738.

4. **Hierarchical/mixed-effects modelling is appropriate for nested/repeated athlete data and supports partial pooling**: Oliveira & Newell (2024), *A hierarchical approach for evaluating athlete performance with an application in elite basketball*, Scientific Reports 14, 1717.

5. **Predictive-likelihood model averaging** provides a statistical basis for assigning continuous weights to competing predictive models rather than relying on binary model selection. The exact holdout implementation here is an engineering approximation of that principle, not a claim that a paper prescribes this exact formula for WNBA H2H.

## What remains an engineering/calibration choice

The following should not be described as constants dictated by the literature:

- 70/30 chronological calibration split;
- minimum calibration-event thresholds;
- the finite search grids for `K` and `tau`;
- exact outer historical bucket weights;
- physical clipping limits and Monte Carlo dispersion parameters.

They are testable modelling choices and should continue to be judged by walk-forward/backtest performance.

## Verification

v2.18.2 passes:

- v2.14 smoke test
- v2.15 smoke test
- v2.16.1 stabilization test
- v2.17.1 player role-state test
- v2.17.2 disjoint H2H/shooting test
- v2.17.3 residual H2H/minutes test
- v2.18 structural compatibility test
- v2.18 team-state/market-prior test
- v2.18.1 OT/DREB test
- v2.18.2 continuous player-H2H + FTA test

