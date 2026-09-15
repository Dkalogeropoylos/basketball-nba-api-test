# NBA V2 Walk-Forward Backtest — Phase C

This is a separate validation sandbox. It does **not** modify the existing NBA Team Markets app.

## Protocol

- TRAIN: 2023-24 (`season=2024`) — fit league-level parameters only.
- VALIDATION: 2024-25 (`season=2025`) — walk-forward current-team state; use to judge/tune the probability layer.
- FINAL TEST: 2025-26 (`season=2026`) — untouched until the validation protocol is frozen.

For each historical game, all team/player rows on or after tip date are removed before building the projection. Current team identity, Old/G6-10/L5, opponent allowed, location, H2H, and optional rotation similarity therefore use only pregame history.

### Frozen from TRAIN

- pace fast/slow control weights and pace residual SD
- opponent elasticities
- structural opponent beta + H2H shrink K
- shooting-efficiency defense beta + shrinkage masses

### Recomputed from current-season pregame history

- team identity / non-overlapping recency buckets
- league current baseline
- opponent-allowed state
- same-season H2H residual evidence
- home/away state
- current pregame rotation similarity

Historical injury overrides are deliberately disabled in Test A because postgame DNP status is not a valid pregame injury feed.

## Streamlit

Main file: `nba_backtest_app.py`

1. Run a 40-game smoke test at 2,500 simulations/game.
2. Inspect failures and center metrics.
3. If clean, set Max games to 0 and run full 2024-25 validation.
4. Do not run 2025-26 final test until the validation settings are frozen.
