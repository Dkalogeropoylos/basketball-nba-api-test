# NBA V2 — Phase C6 Untouched 2025-26 Final Exam

This patch integrates the final untouched 2025-26 exam into the existing `nba_variance_ab_app.py`.

## Files
- Replace: `nba_variance_ab_app.py`
- Add: `backtest/final_exam.py`

No `core/` file is changed. No model coefficient is changed.

## Frozen protocol
- TRAIN league calibration: 2023-24 only
- Final exam season: 2025-26
- Minimum prior current-season games/team: 15
- Simulations/game: 1500
- Pregame rotation similarity: ON
- A: frozen C3 baseline (`fga_process=poisson`, FTA log sigma `0.12`)
- B3: frozen rebound-chain FGA + frozen FTA dispersion from the 2024-25 development process
- All eligible 2025-26 games are evaluated; there is no development split in the final season.

With the current normalized 2025-26 regular-season dataset there are 1,230 games total and 998 games eligible after the fixed 15-prior-games/team rule.

## What one FINAL batch computes
For the same historical game, before tip and using only history dated before the game:
1. Frozen baseline A simulation.
2. Frozen B3 simulation.
3. Center/distribution rows for A and B3.
4. A-defined half-point lines (pregame only; actual outcome never defines the line).
5. A and B3 probabilities on those exact same lines.
6. Most-market probabilities for A and B3.

Absolute game-index seeds make batching computational only: changing batch size does not change the intended final protocol.

## How to run
1. Upload this patch to the same NBA repo.
2. Keep the same Streamlit main file: `nba_variance_ab_app.py`.
3. Press `1) Load seasons + fit frozen TRAIN calibration` as before.
4. Scroll to Section E.
5. Tick the freeze confirmation and type `FREEZE B3`.
6. Press `4) Load untouched 2025-26 FINAL season`.
7. Run `5) Run NEXT untouched FINAL batch...` until 998/998 and failures=0.
8. Download the final checkpoint/detail/shared-line/Most files and evaluate once.

## Scientific guard
Do not tune B3 from the 2025-26 results. If the final season exposes a weakness, record it as a limitation. Any revised architecture would require a new future untouched test set.
