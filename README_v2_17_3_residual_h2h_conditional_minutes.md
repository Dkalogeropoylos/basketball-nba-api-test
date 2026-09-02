# v2.17.3 — Residualized player H2H + conditional minute uncertainty

This patch sits on top of v2.17.2. It does **not** target any particular player projection. The goal is to remove the remaining opponent/H2H overlap and replace two hand-chosen minute assumptions with data-driven uncertainty/calibration.

## 1) Player H2H is now a residual, not a second matchup boost

The live player opportunity chain is now conceptually:

```
current non-H2H player role rate
× generic opponent / opponent-by-position effect
× pair-specific H2H residual
```

The explicit H2H term is **not** estimated from raw H2H rate divided by the player's ordinary baseline.

For each historical H2H game, v2.17.3 reconstructs the no-H2H expectation that existed before that game:

```
historical expected rate
= historical non-pair player rate
  × historical generic opponent modifier
```

Important disjointness rules:

- only rows strictly before the historical H2H date are used;
- the focal team is excluded from the opponent-specific overall sample;
- the focal team is excluded from the opponent-by-position sample;
- the current opponent H2H rows are already excluded from the player's Old/G6-10/L5 + adaptive role-state baseline by v2.17.2;
- shooting percentages remain outside the H2H opportunity layer.

The H2H evidence therefore becomes a player×opponent residual:

```
raw residual = observed H2H events / no-H2H expected H2H events
```

The residual is partial-pooled toward 1.00.

## 2) Gamma-Poisson residual shrinkage is in expected-event units

Because a 30-minute H2H for a high-volume creator contains more information than 30 minutes for a low-volume creator, the prior is no longer expressed only as equivalent player minutes.

For a stat-specific pair residual `r`:

```
count ~ Poisson(expected_no_H2H_events × r)
r ~ Gamma(K, K)
```

so the prior mean is exactly 1.00.

With effective historical observed events `C` and effective no-H2H expected events `E`:

```
posterior residual = (C + K) / (E + K)
```

`K = infinity` is always available and means the data rejected an explicit H2H layer for that stat.

## 3) The minute-relevance decay is learned, not fixed at `/10`

v2.17.2 used:

```
exp(-abs(H2H_MIN - ProjectedMIN) / 10)
```

The denominator 10 was a manual calibration choice.

v2.17.3 jointly learns a stat-specific minute-relevance scale `tau` with the residual prior mass `K` from chronological repeat-matchup prediction:

```
minute relevance = exp(-abs(H2H_MIN - current/projected MIN) / tau)
```

The candidate set includes:

```
tau = infinity
```

so the model can decide that a 27-minute H2H should **not** be penalized merely because today's projection is 31.5 minutes.

The H2H calibration is blocked chronologically:

1. choose `K` and `tau` on the earlier 70% of repeat-matchup prediction events;
2. require the chosen residual model to beat no-H2H on the later 30%;
3. otherwise set `K = infinity` and the H2H modifier is 1.00.

This reduces the chance that the two calibration parameters activate only because they overfit the same games on which they were selected.

## 4) Live generic opponent context is leave-pair-out

For Player Props only, the current focal team is removed from:

- opponent overall allowed sample;
- opponent-by-position sample.

That avoids letting, for example, Citron's own Phoenix H2Hs partly define the generic Phoenix AST environment and then re-enter through the explicit Citron×Phoenix residual.

Team Markets are unchanged.

## 5) Monte-Carlo minutes: no downward penalty

The simulation still draws minutes symmetrically around the central projection:

```
MIN_sim = ProjectedMIN + SD_MIN × Z
```

with the existing safety clipping.

There is no transformation such as `31.5 -> 30.0` because minutes are uncertain.

### Trader / metadata overrides

v2.17.2 capped the simulated minute SD for trader/metadata overrides at 2.25 minutes.

v2.17.3 removes that hard cap. A trader override changes the **central minute estimate only**. The uncertainty remains the historical role-conditioned `Raw SD` produced by the minutes engine.

For AUTO minutes, the existing mild allocation-scale adjustment remains:

```
SD_final = RawSD × sqrt(allocation ratio)
```

with the same safety bounds.

This means:

- central trader estimate stays unbiased;
- uncertainty is not artificially reduced just because a human supplied the mean;
- an actually stable rotation can still have a small historical SD;
- a volatile rotation can legitimately have a larger SD.

No asymmetric return/restriction distribution is added in this patch because the current metadata schema does not yet encode a reliable downside-mixture state. That should be a separate change and backtest.

## 6) Audits added

### H2H calibration audit

Shows by 2PA / 3PA / REB / AST:

- calibration events;
- train / held-out events;
- learned residual-event prior `K`;
- learned minute-relevance `tau`;
- train no-H2H vs residual NLL;
- held-out no-H2H vs residual NLL;
- held-out NLL gain;
- whether H2H is active.

### Player H2H audit

Shows:

- historical observed effective events;
- historical no-H2H expected effective events;
- raw residual ratio;
- rotation similarity;
- learned `tau`;
- mean minute relevance;
- residual prior `K`;
- posterior H2H weight;
- current non-H2H player rate;
- current generic opponent modifier;
- final residual modifier.

The deep-dive also exposes the historical H2H expectations used to build the residual.

### Minutes audit

Adds `Minutes SD Method`, making it explicit whether SD comes from:

- historical conditional SD + AUTO allocation scale;
- historical conditional SD with trader/metadata mean override;
- no simulation for OUT/outside-rotation players.

## Files changed

Required:

- `core/matchup.py` — REPLACE
- `core/minutes_engine.py` — REPLACE
- `streamlit_app.py` — REPLACE

New:

- `tests/v2173_residual_h2h_minutes_smoke.py`
- `README_v2_17_3_residual_h2h_conditional_minutes.md`

Everything from v2.17.2 shooting efficiency and v2.17 team structural calibration remains in the package.

## Regression checks

Passed:

- v2.14 smoke
- v2.15 smoke
- v2.16.1 stabilization smoke
- v2.17 structural smoke
- v2.17.1 adaptive player role-state smoke
- v2.17.2 disjoint H2H + shooting smoke
- v2.17.3 residual H2H + conditional minute SD smoke

The new residual calibration was also benchmarked on a synthetic ~1.5k player-event repeat-matchup set and completed in a few seconds after cumulative-stat/vectorized calibration optimization.

## Still required before treating fair odds as calibrated truth

This is an architectural/statistical correction, not a completed out-of-sample betting calibration. Re-run historical player-prop backtests and compare v2.17.2 vs v2.17.3 on:

- mean error / MAE / RMSE by stat;
- Poisson/log-score or CRPS-type distribution score;
- line hit-probability calibration;
- fair-odds reliability curves;
- H2H-active vs H2H-inactive subsets;
- stable-minute vs volatile-minute subsets.

Do not judge the patch by whether one player moves toward a preselected number.
