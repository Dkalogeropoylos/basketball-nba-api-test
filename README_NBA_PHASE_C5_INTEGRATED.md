# NBA V2 Phase C5 — integrated shared-line A vs B3 test

This patch adds the final 2024-25 probability A/B test **inside the existing C4 app**. No new Streamlit app is required.

## Files

Replace:
- `nba_variance_ab_app.py`

Add:
- `backtest/shared_line_eval.py`
- `nba_c5_A_fixed_lines_late498.csv.gz`

Do not change any `core/` model file for this phase.

## What section D does

- Reuses the frozen baseline-A half-point lines and A probabilities from the late 498-game holdout.
- Does **not** rerun baseline A.
- Reruns only B3 with the frozen C4 settings and the same absolute-game seeds.
- Evaluates `P_B3` at the exact same lines as A.
- Compares Brier and log loss on identical binary events.
- Uses a GAME_ID-clustered bootstrap for the Brier-difference confidence interval because multiple synthetic lines from one game are correlated.

## Frozen settings

- TRAIN: 2023-24
- evaluation: 2024-25 late 498 games
- min prior games/team: 15
- B3 simulations/game: 1500
- rotation similarity: ON
- B3 FGA process: rebound chain
- B3 FTA log sigma: frozen C4 development value
- 2025-26: still untouched

## Run order

1. Open the same `nba_variance_ab_app.py` Streamlit app.
2. Click `1) Load seasons + fit frozen TRAIN calibration`.
3. Scroll to `D. Shared-line probability A/B — SAME exact lines`.
4. Verify the reference audit says 498 games, zero non-half lines, zero push-probability rows.
5. Click `3) Run NEXT B3 shared-line batch` until 498/498 with 0 failures.
6. Download `nba_c5_shared_line_rows.csv` and `nba_c5_shared_line_summary.csv`.

Negative `Brier Δ B3-A` / `LogLoss Δ B3-A` means B3 is better on the same lines.
