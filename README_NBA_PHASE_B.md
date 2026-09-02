# NBA V2 Phase B Sandbox

This is a **separate NBA sandbox** built from the frozen WNBA V2.18.2 + DREB framework.

What is changed for NBA:
- data provider -> SportsDataverse NBA GitHub release assets;
- only 30 NBA franchise team IDs are retained;
- All-Star mini-games are excluded;
- completed-season NBA Cup championship extra 83rd game is removed from regular-season calibration;
- regulation game length -> 48 minutes;
- team regulation minutes -> 240;
- player minute cap -> 48;
- pace calibration is re-fit from NBA history and NBA-safe pace bounds are used;
- structural/opponent/shooting/H2H calibration functions train on NBA data loaded into the app.

What is NOT changed:
- the WNBA production app/repository;
- the V2 modelling architecture (disjoint buckets, opponent context, residual H2H, coupled Monte Carlo, pricing);
- bookmaker input remains comparison-only.

Entrypoint on Streamlit Cloud:
`streamlit_app_nba.py`

Python:
3.12

Secrets:
none required for the historical NBA backbone.

Phase B status:
This is a validation sandbox, not yet a production betting model. It must be smoke-tested
for data identities, 240-minute conservation, pace, and market outputs before any betting use.
