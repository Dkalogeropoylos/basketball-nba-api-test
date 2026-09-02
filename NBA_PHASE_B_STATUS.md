# NBA Phase B status — 2025-26 data audit

Validated from full SportsDataverse exports supplied by the user:

- raw team boxscores: 2,652 rows / 1,326 games across all ESPN season types;
- raw player boxscores: 34,883 rows / 1,326 games;
- season_type=2 contains the 30 NBA franchises plus All-Star mini-game teams;
- after keeping team IDs 1–30, there are 1,231 season_type=2 games;
- San Antonio and New York have 83 such games because the NBA Cup championship is included as an extra game;
- removing game 401809839 restores exactly 1,230 regular-season games and 82 games for every NBA franchise;
- normalized participating-player rows after DNP removal: 26,547;
- regulation/OT exposure observed after normalization: 2,352 team-rows at 48 minutes, 92 at 53 minutes, 16 at 58 minutes.

NBA sandbox smoke checks already passed locally:

- Boston projected team minutes sum to exactly 240.0;
- New York projected team minutes sum to exactly 240.0;
- NBA pace calibration for BOS @ NY produced a central pace around 97.70 possessions from NBA history;
- all Python modules and the NBA Streamlit entrypoint compile successfully.

Important remaining Phase B work:

1. benchmark/optimize the walk-forward structural-rate calibration on the larger NBA dataset;
2. run full coupled Monte Carlo on a frozen hypothetical matchup;
3. compare NBA V2 output against an independently frozen manual projection;
4. only after validation consider production integration.

WNBA production is not modified by this sandbox.
