# Basketball Pricing Engine v2.18.1

**Base:** v2.18.0 consolidated package.

This patch deliberately changes only two architectural points that were exposed by the Atlanta–Portland audit. It does **not** insert a manual rebound subtraction and does **not** target any specific Dream/Fire projection.

## 1. Overtime is now treated as extra exposure, not as extra 40-minute pace

A 45-minute WNBA game naturally contains more possessions and more player minutes than a regulation game. Using the raw team possessions as if they came from 40 minutes makes an OT game look artificially fast inside recent pace buckets.

v2.18.1 therefore computes a regulation-equivalent factor:

```text
regulation_factor = 40 / GAME_LENGTH_MIN
```

For 1OT:

```text
40 / 45 = 0.888888...
```

For 2OT:

```text
40 / 50 = 0.80
```

`GAME_LENGTH_MIN` is derived from the player-minute accounting when SportsDataverse WNBA data are loaded. Team player minutes sum to 200 in regulation, 225 in 1OT, 250 in 2OT, etc. The legacy `OT_FLAG` remains as a conservative fallback.

### What is normalized

- historical team possessions used by the pace engine;
- league pace calibration targets;
- the historical pace environment used by Player Props;
- historical player minutes used to estimate the normal 40-minute rotation state.

### What is NOT mechanically scaled

Opportunity rates remain rates:

- `3PA / FGA`;
- `TOV / POSS`;
- `FTA / POSS`;
- `OREB / missed FGA`;
- `DREB / available defensive-rebound chances`;
- `AST / made FG`.

The OT game therefore does not get an arbitrary global penalty. Extra clock exposure is removed from pace/minute state, while within-game event rates retain their real observed opportunities.

The old player-minutes rule

```text
if OT: historical weight *= 0.78
```

has been removed. It was an engineering constant, not a literature-derived quantity. Historical minutes are instead converted to a regulation equivalent.

### Basketball-analysis basis

Kubatko, Oliver, Pelton & Rosenbaum (2007), *A Starting Point for Analyzing Basketball Statistics*, Journal of Quantitative Analysis in Sports 3(3), DOI `10.2202/1559-0410.1070`, establishes possessions, per-minute statistics, pace adjustment and rebound rates as core basketball-analysis tools. The paper does not prescribe the exact `40/game_length` implementation used here; that factor is the direct exposure conversion implied by comparing a 40-minute regulation game with a longer overtime game.

---

## 2. DREB capture joins the residualized structural H2H model

The Atlanta–Portland audit exposed an asymmetry in v2.18.0:

- `OREB_PER_MISS` could receive a validated residual H2H term;
- `DREB_CAPTURE` was still audit-only and therefore its H2H value could not affect production.

v2.18.1 adds:

```text
DREB_CAPTURE
    = team DREB
      / (opponent missed FGA - opponent OREB)
```

as a sixth structural feature alongside:

- `3P_SHARE`;
- `FTA/POSS`;
- `TOV/POSS`;
- `OREB/MISS`;
- `AST/MAKE`.

The same v2.18 decomposition is used:

```text
transformed prediction
    = own non-H2H state
    + beta * opponent-allowed deviation from league
    + shrunk H2H residual
```

The H2H component is **not** the raw Atlanta–Portland DREB rate. For each historical repeat matchup it is the residual after the own + opponent expectation. Its current influence is then shrunk by the chronologically selected prior mass `K` and by current rotation similarity.

```text
H2H effect
    = mean residual * N/(N+K) * rotation similarity
```

`K = infinity` is still a valid candidate and means H2H gets zero production weight.

The current app audit now shows, for DREB capture as well:

- own non-H2H state;
- opponent-allowed state;
- opponent beta;
- prediction without H2H;
- usable residual H2H games;
- H2H prior K;
- effective H2H weight;
- raw-unit H2H delta;
- final learned DREB-capture prediction.

### Basketball/statistical basis

Dong et al. (2021), *Addressing opposition quality in basketball performance evaluation*, International Journal of Performance Analysis in Sport 21(2), 263–276, DOI `10.1080/24748668.2021.1877938`, explicitly finds that basketball indicators respond differently to opposition quality and reports defensive rebounds among the indicators related to relative opposition quality.

Cui, Gonçalves & Gómez (2019), *Performance profiles and opposition interaction during game-play in elite basketball: evidences from National Basketball Association*, International Journal of Performance Analysis in Sport 19(1), 28–48, DOI `10.1080/24748668.2018.1555738`, documents opposition-interaction effects and identifies defensive rebounds/turnovers among important game-related indicators.

Oliveira & Newell (2024), *A hierarchical approach for evaluating athlete performance with an application in elite basketball*, Scientific Reports 14:1717, DOI `10.1038/s41598-024-51232-2`, supports multilevel treatment of repeated/contextual basketball observations. v2.18.1 uses the same statistical principle through residualization and partial pooling rather than treating two H2Hs as a full-strength independent sample.

These papers support the **architecture** (opponent context, repeated-context shrinkage, opportunity/pace normalization). They do not dictate our exact beta grid, K grid, 70/30 split or holdout thresholds; those remain engineering/calibration choices and therefore stay subject to chronological out-of-sample validation.

---

## 3. Rebounds are still not manually forced downward

There is no `-1 REB`, `-2 REB`, line cap or Atlanta-specific correction.

The simulation still follows:

```text
pace + TOV + FTA + OREB
        -> FGA
        -> missed shots
        -> available DREB chances
        -> DREB capture
        -> total REB
```

v2.18.1 can reduce projected rebounds for two principled reasons only:

1. OT no longer inflates the historical 40-minute pace state;
2. validated DREB matchup residuals can now alter the capture probability.

If those changes leave Atlanta at 38+ rebounds, the model keeps 38+. The patch is not calibrated to a desired answer.

---

## 4. Files changed from v2.18.0

- `core/exposure.py` — new regulation-equivalent exposure helper.
- `providers/sportsdataverse_wnba.py` — derives `GAME_LENGTH_MIN` and `OT_COUNT` from player-minute accounting.
- `core/pace_engine.py` — regulation-equivalent OT possessions for team/player historical pace and pace calibration.
- `core/minutes_engine.py` — replaces fixed OT weight `0.78` with regulation-equivalent historical minutes.
- `core/team_model.py` — bucket audit exposes raw vs regulation-equivalent pace; H2H audit includes OT metadata.
- `core/structural_calibration.py` — adds `DREB_CAPTURE` to opponent-adjusted residual-H2H calibration.
- `streamlit_app.py` — v2.18.1 production integration and updated audits.
- `tests/v2181_ot_dreb_h2h_smoke.py` — OT and DREB structural tests.

## 5. Validation performed

The package was compiled and the following tests passed after the changes:

- `v2181_ot_dreb_h2h_smoke.py`;
- `v2180_team_state_market_prior_smoke.py` (updated to include DREB capture);
- `v2173_residual_h2h_minutes_smoke.py`;
- `v2172_disjoint_h2h_shooting_smoke.py`;
- `v2161_stabilization_smoke.py`;
- `v215_smoke.py`;
- `v214_smoke.py`.

The new synthetic structural test confirms that DREB capture is available as a calibrated structural feature. The OT test confirms that a 45-minute game with 90 raw possessions but an 80-possession regulation-equivalent pace enters the pace state as 80 rather than 90.
