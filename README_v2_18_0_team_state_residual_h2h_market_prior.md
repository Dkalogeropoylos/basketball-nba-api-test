# Basketball Pricing Engine v2.18.0

**Base:** apply this as the consolidated successor to v2.17.2. It already contains the full v2.17.3 player patch, so there is no need to apply v2.17.3 separately.

## What is bundled

### Player Props — v2.17.3 retained

- Player H2H is residualized against the historical no-H2H expectation: current-role player rate + generic opponent/position environment.
- H2H residual shrinkage and minute-relevance `tau` are selected chronologically; `tau = inf` is allowed, meaning no penalty merely because an old H2H had different minutes.
- Trader projected minutes move the central mean. Their Monte Carlo SD is the historical role-conditioned uncertainty; there is no automatic downward minutes penalty and no fixed 2.25-minute trader cap.

See `README_v2_17_3_residual_h2h_conditional_minutes.md` for the full player-side implementation.

---

# Team Markets — v2.18 structural change

## 1. Why the v2.17 ridge model was replaced

v2.17 put Own Old/G6-10/L5, Opponent Old/G6-10/L5 and two H2H terms into one ridge regression. Those recency variables are naturally highly correlated. Ridge stabilizes variance, but it does not make the contribution of the opponent or H2H easy to interpret and it can produce a globally good fit that behaves poorly for an extreme opponent.

v2.18 uses the simpler decomposition:

```text
transformed expected rate
    = own non-H2H state
    + beta * opponent-allowed deviation from league
    + shrunk repeat-matchup residual
```

The model is applied to:

- `3P_SHARE = 3PA / FGA`
- `FTA_RATE = FTA / possessions`
- `TOV_RATE = TOV / possessions`
- `OREB_PER_MISS = OREB / missed FGA`
- `AST_PER_MAKE = AST / made FG`

The own and opponent histories are built from non-overlapping Old / G6-10 / L5 buckets, with the current pair removed from both baselines.

### Statistical rationale

Basketball performance research supports accounting explicitly for opposition quality rather than evaluating game indicators as if the opponent were irrelevant. Dong et al. (2021) specifically model basketball indicators with opposition quality and show that different indicators react differently to opponent context. This is why v2.18 estimates a **stat-specific** opponent response rather than a universal opponent weight.

Reference: Dong, R., Lian, B., Zhang, S., Zhang, M., Huang, S. Z. Y. & O'Donoghue, P. (2021), *Addressing opposition quality in basketball performance evaluation*, International Journal of Performance Analysis in Sport, 21(2), 263–276. DOI: `10.1080/24748668.2021.1877938`.

Hierarchical/mixed-effect approaches are also a natural basis for separating player/team baseline effects from contextual effects instead of treating every small subgroup as an independent full-strength sample. See Oliveira & Newell (2024), *A hierarchical approach for evaluating athlete performance with an application in elite basketball*, Scientific Reports 14, 1717.

---

## 2. Opponent strength is learned chronologically and cannot reverse sign

For each structural stat, v2.18 selects:

```text
beta in [0, 2]
```

from earlier chronological validation blocks.

`beta = 0` is explicitly allowed. Therefore the data can say the opponent adds no predictive information.

The opponent signal is defined as the amount that the opponent historically **allows** above/below league average. Its coefficient is constrained non-negative. Thus a defense that historically allows more of a stat cannot reduce the prediction purely because correlated Old/Mid/L5 coefficients happened to flip sign.

This is a shape/structural constraint, not a manual Atlanta/Portland adjustment. The magnitude remains data-selected.

---

## 3. Extreme-opponent stress test

A model is not activated merely because it improves global RMSE.

After parameter selection on earlier data, the last 30% of chronological rows is held out. Activation requires:

1. lower RMSE than the own-state baseline on that later holdout; and
2. no deterioration versus baseline on the most extreme quartile of opponent contexts, when there is enough evidence.

This specifically protects against a model that works for average defenses but breaks on a team with an unusually strong turnover-forcing or 3PA-suppressing profile.

---

## 4. Team H2H: fixed 6–10% weights are removed

The previous production path could use:

```text
0.20 * N/(N+2) * rotation_similarity
```

capped at 10% for some team H2H rates. That was an engineering prior, not a literature-derived weight.

v2.18 removes fixed percentage H2H influence from production Team Markets.

For the five calibrated structural rates, H2H is now a **residual**:

```text
historical residual
    = actual H2H transformed rate
      - historical no-H2H own+opponent expectation
```

The current H2H term is:

```text
mean residual * N/(N + K) * current rotation similarity
```

where `K` is selected chronologically from:

```text
0.5, 1, 2, 4, 8, 16, 32, infinity
```

`K = infinity` means the validation says H2H should have **zero** numerical influence.

For FGA, PF, DREB, STL and BLK, H2H is currently audit-only until a residual specification is validated. There is no arbitrary fallback H2H percentage.

The UI now shows for every calibrated team stat:

- own non-H2H state;
- opponent-allowed state;
- learned opponent beta;
- prediction **without H2H**;
- raw H2H games;
- usable residual H2H games;
- selected H2H prior `K`;
- effective H2H weight;
- H2H delta in the original stat units;
- final prediction;
- later holdout and extreme-opponent validation errors.

This makes the exact importance of H2H observable rather than hidden.

---

## 5. FGA is no longer given a second generic matchup multiplier

The simulator already enforces the possession identity approximately:

```text
POSS = FGA - OREB + TOV + 0.44*FTA
```

Therefore production v2.18 sets the generic opponent FGA modifier to neutral. FGA is generated mainly from:

- shared possessions;
- turnovers;
- free-throw possession component;
- offensive-rebound recycling.

This is important for the Portland-style diagnostic: if the old model projected too few turnovers, the identity itself generated too many FGA, misses and therefore opponent DREB. v2.18 fixes the upstream structural response rather than forcing a manual rebound subtraction.

The possession/rebound architecture follows the standard possession-based basketball-analysis framework discussed by Kubatko, Oliver, Pelton & Rosenbaum (2007), *A Starting Point for Analyzing Basketball Statistics*, Journal of Quantitative Analysis in Sports 3(3), DOI `10.2202/1559-0410.1070`.

---

## 6. Rebounds are NOT manually reduced

There is no `-2 REB` patch and no artificial rebound cap.

DREB still comes from opponent missed-FG opportunities not recovered as OREB, multiplied by the team's empirical DREB capture probability. Therefore if the new TOV / FTA / OREB / 3P-share chain reduces opponent shot/miss volume, rebounds fall automatically.

This preserves the causal/box-score chain:

```text
TOV / FTA / OREB / pace
        -> FGA
        -> misses
        -> available defensive rebounds
        -> DREB
        -> total REB
```

If rebounds remain too high after the upstream changes, the next audit should focus on miss generation and capture calibration rather than editing the final REB line by hand.

---

# Optional sportsbook handicap prior

## 7. Why the spread can be useful

A bookmaker spread is itself a forecast of scoring margin. The forecast-combination literature shows that combining forecasts can improve out-of-sample performance when the second forecast contains independent information.

Key references:

- Bates, J. M. & Granger, C. W. J. (1969), *The Combination of Forecasts*, Journal of the Operational Research Society 20(4), 451–468. DOI `10.1057/jors.1969.103`.
- Wang, X., Hyndman, R. J., Li, F. & Kang, Y. (2023), *Forecast combinations: An over 50-year review*, International Journal of Forecasting 39(4), 1518–1547. DOI `10.1016/j.ijforecast.2022.11.005`.
- Sports-forecasting literature also treats betting-market prices/spreads as informative forecasts; see Stekler, Sendor & Verlander (2010), *Issues in sports forecasting*, International Journal of Forecasting 26(3), 606–621.

## 8. No arbitrary market weight

The current handicap receives **zero numerical weight by default**.

To activate market augmentation, upload a historical CSV with:

```text
GAME_DATE                 optional
MODEL_HOME_MARGIN         required
MARKET_HOME_SPREAD        required   # sportsbook sign, e.g. home -6.5
ACTUAL_HOME_MARGIN        required
```

The first 70% chronologically estimates the convex forecast-combination weight:

```text
combined_margin
    = model_margin + w*(market_margin - model_margin)
```

with `0 <= w <= 1`.

The later 30% is untouched during fitting. The market prior activates only if the combined margin has lower holdout RMSE than the pure model. Otherwise `w = 0`.

The current market total remains audit-only because the user's own total model is already a separate forecast and there is not yet a calibrated reason to force it toward the book.

## 9. How an active handicap prior enters the Monte Carlo

It does **not** add/subtract arbitrary points or edit individual team stats.

The existing coherent joint simulations are exponentially tilted toward the validated blended margin and then resampled using the same row index for both teams. This preserves the natural correlations and box-score identities within each simulated game.

The UI reports:

- pure-model margin before the prior;
- market-implied margin;
- learned market weight;
- blended target margin;
- margin after conditioning;
- game-total drift.

---

# Files changed from v2.17.2

The consolidated patch contains both the previously uninstalled v2.17.3 player work and v2.18 Team changes:

- `core/matchup.py` — v2.17.3 player residual H2H calibration.
- `core/minutes_engine.py` — v2.17.3 conditional minutes SD.
- `core/structural_calibration.py` — v2.18 team own/opponent/residual-H2H state model.
- `core/team_model.py` — production fixed-H2H path disabled/audited.
- `core/market_prior.py` — optional chronologically calibrated handicap forecast combination.
- `streamlit_app.py` — v2.18 integration + audits.
- `tests/v2173_residual_h2h_minutes_smoke.py`
- `tests/v217_structural_smoke.py`
- `tests/v2180_team_state_market_prior_smoke.py`

## Validation performed

- All source files compile.
- Legacy smoke tests through v2.17.3 were rerun individually.
- v2.18 synthetic structural tests verify non-negative opponent beta, residual-H2H audit fields and the five structural modifiers.
- v2.18 market-prior tests verify that no historical calibration means a guaranteed no-op and that an active calibrated prior moves the *joint* simulation margin rather than editing isolated stats.

## Important calibration note

This is a methodological architecture upgrade, not proof that bookmaker probabilities are already calibrated. Run the same games under v2.17.2 and v2.18.0 and retain a walk-forward history of model projections, bookmaker lines and actual outcomes before treating changes in fair odds as production-calibrated probabilities.
