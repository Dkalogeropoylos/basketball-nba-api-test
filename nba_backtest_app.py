from __future__ import annotations

import streamlit as st
import pandas as pd

from providers.sportsdataverse_nba import SportsDataverseNBA
from backtest.calibration import fit_league_calibration
from backtest.engine import run_walk_forward
from backtest.evaluation import metric_table, calibration_bins, most_summary
from backtest.data_loader import season_label

st.set_page_config(page_title="NBA V2 Walk-Forward Backtest", layout="wide")
st.title("NBA V2 — Walk-Forward Validation Sandbox")
st.caption(
    "Train league-level parameters on 2023-24, validate on 2024-25, then keep 2025-26 untouched for the final test. "
    "Team identity always comes from current-season games completed before tip."
)

@st.cache_data(show_spinner=False)
def load_season(season: int):
    return SportsDataverseNBA(timeout=60).load_season(int(season))

unlock_final = st.checkbox(
    "Unlock FINAL TEST 2025-26 (only after 2024-25 settings are frozen)",
    value=False,
)

c1, c2, c3 = st.columns(3)
train_season = c1.selectbox("TRAIN league parameters", [2024], format_func=lambda x: season_label(x))
_val_options = [2025, 2026] if unlock_final else [2025]
validation_season = c2.selectbox("Evaluation season", _val_options, index=0, format_func=lambda x: season_label(x))
min_games = c3.number_input("Minimum prior current-season games/team", min_value=8, max_value=30, value=15, step=1)
if validation_season == 2026:
    st.error("FINAL TEST unlocked. Do not tune any coefficient/setting after seeing these results.")

with st.expander("Run settings", expanded=True):
    a,b,c = st.columns(3)
    n_sims = a.select_slider("Simulations per historical game", options=[1000,1500,2500,5000,10000], value=2500)
    max_games = b.number_input("Max games (0 = all eligible)", min_value=0, max_value=1230, value=40, step=10)
    rotation = c.toggle("Pregame rotation-similarity weighting", value=True)
    st.caption("Start with 40 games as a smoke test. Once clean, set Max games = 0 for the full validation run.")

if st.button("1) Load seasons + fit TRAIN calibration", type="primary"):
    with st.spinner("Loading SportsDataverse seasons and fitting frozen league-level parameters..."):
        train_pack = load_season(int(train_season))
        val_pack = load_season(int(validation_season))
        cal = fit_league_calibration(train_pack["team"], int(train_season))
        st.session_state["bt_train_pack"] = train_pack
        st.session_state["bt_val_pack"] = val_pack
        st.session_state["bt_cal"] = cal
    st.success(
        f"Loaded TRAIN {season_label(train_season)} and VALIDATION {season_label(validation_season)}. "
        f"Frozen pace weights: fast={cal.pace['fast_weight']:.3f}, slow={cal.pace['slow_weight']:.3f}, SD={cal.pace['rmse']:.3f}."
    )

cal = st.session_state.get("bt_cal")
val_pack = st.session_state.get("bt_val_pack")

if cal is not None:
    with st.expander("Frozen TRAIN calibration audit"):
        st.markdown("**Structural rates**")
        st.dataframe(cal.structural_audit.round(4), use_container_width=True, hide_index=True)
        st.markdown("**Shooting efficiency**")
        st.dataframe(cal.shooting_audit.round(4), use_container_width=True, hide_index=True)
        st.markdown("**Legacy/generic opponent elasticities**")
        st.dataframe(cal.opponent_audit.round(4), use_container_width=True, hide_index=True)

if cal is not None and val_pack is not None:
    if st.button("2) Run leakage-safe walk-forward validation"):
        bar = st.progress(0.0, text="Preparing games...")
        def prog(done, total):
            bar.progress(done/max(total,1), text=f"Historical games: {done}/{total}")
        with st.spinner("Running historical pregame projections..."):
            res = run_walk_forward(
                val_pack["team"], val_pack["player"], cal,
                min_prior_games=int(min_games), n_sims=int(n_sims),
                max_games=(None if int(max_games)==0 else int(max_games)),
                use_rotation_similarity=bool(rotation), progress_callback=prog,
            )
        st.session_state["bt_result"] = res
        bar.empty()

res = st.session_state.get("bt_result")
if res is not None:
    st.success(
        f"Finished in {res.runtime_seconds/60:.1f} min. "
        f"Projected games: {res.detail['GAME_ID'].nunique() if not res.detail.empty else 0}; failures: {len(res.failures)}."
    )
    metrics = metric_table(res.detail)
    st.subheader("A. Center + distribution validation")
    st.caption("Bias/MAE/RMSE test the centers. Coverage and CRPS test the Monte Carlo distribution around those centers.")
    st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)

    st.subheader("B. Probability calibration — synthetic pregame lines")
    st.caption("These are NOT bookmaker backtests. Half-point lines are generated only from the pregame simulation, so we can test whether 60% behaves like 60% without needing historical odds.")
    over_bins = calibration_bins(res.probability_rows, "Over")
    if not over_bins.empty:
        st.dataframe(over_bins.round(4), use_container_width=True, hide_index=True)
    else:
        st.info("Need a larger run before probability bins have enough observations.")

    st.subheader("C. Team-with-most calibration")
    ms = most_summary(res.most_rows)
    st.dataframe(ms.round(4), use_container_width=True, hide_index=True)

    if not res.failures.empty:
        with st.expander("Failures / skipped projections"):
            st.dataframe(res.failures, use_container_width=True, hide_index=True)

    st.subheader("Downloads")
    d1,d2,d3 = st.columns(3)
    d1.download_button("Detailed game-market rows", res.detail.to_csv(index=False).encode(), "nba_backtest_detail.csv", "text/csv")
    d2.download_button("Probability calibration rows", res.probability_rows.to_csv(index=False).encode(), "nba_backtest_probability_rows.csv", "text/csv")
    d3.download_button("Most-market rows", res.most_rows.to_csv(index=False).encode(), "nba_backtest_most_rows.csv", "text/csv")

st.divider()
st.warning(
    "Protocol guard: do NOT use 2025-26 to tune parameters after looking at its final-test results. "
    "First tune/approve on 2024-25; only then run 2025-26 once as the untouched exam."
)
