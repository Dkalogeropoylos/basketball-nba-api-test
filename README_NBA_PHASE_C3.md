# NBA V2 Phase C3 — batched/resumable validation

No model coefficient is changed versus Phase C2. This patch changes only backtest execution.

Why: the full 2024-25 one-shot validation can require a long continuous Streamlit Cloud run and holds hundreds of thousands of synthetic-line rows before rendering results.

Changes:
- `backtest/engine.py`: adds deterministic `start_index` batching, absolute-index simulation seeds, and small filtering optimizations.
- `nba_backtest_app.py`: runs 100-game batches by default, accumulates results, offers checkpoint ZIP download/restore, and only exposes final CSV downloads when all eligible games are complete.
- Includes the Phase C2 `3P_SHARE` H2H attribute fix in `backtest/calibration.py`.

Recommended settings: TRAIN 2023-24, validation 2024-25, min prior games 15, 1500 simulations, 100 games/batch, rotation similarity ON.
