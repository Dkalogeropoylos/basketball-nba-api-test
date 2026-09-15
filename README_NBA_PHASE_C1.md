# NBA V2 Phase C1 — 48-minute player exposure fix + backtest

This patch is intentionally narrow.

## Changes
1. `core/minutes_engine.py`
   - Final player `Low Min` / `High Min` stress bounds now clip at **48**, not 40.
   - The existing NBA central allocation remains **240 team minutes** with **48 max per player**.

2. `core/player_model.py`
   - Player Monte Carlo keeps its historical +/-8 minute stress band, but the upper tail is now physically capped at **48 regulation minutes**.
   - No rate, matchup, shooting, H2H, or pricing coefficient is changed.

3. `tests/nba_minutes_48_smoke.py`
   - Checks that simulated player minutes never exceed 48 and that the final stress interval is wired to 48.

4. Phase C backtest files are included unchanged from the prior backtest patch.

## Deliberately NOT changed
- Player H2H remains on the current empirical-Bayes prior-minutes wiring for now.
- Player shooting priors are unchanged.
- Team OREB/DREB structural wiring is unchanged.

Reason: the next validation should test the current production-equivalent architecture after fixing only the clear NBA port bug. We can A/B-test alternative H2H or rebound wiring later instead of changing multiple layers before the baseline backtest.
