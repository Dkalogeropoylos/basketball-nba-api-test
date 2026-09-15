from __future__ import annotations

import io
import json
import zipfile
import streamlit as st
import pandas as pd

from providers.sportsdataverse_nba import SportsDataverseNBA
from backtest.calibration import fit_league_calibration
from backtest.engine import run_walk_forward, eligible_game_count
from backtest.evaluation import metric_table, calibration_bins, most_summary
from backtest.data_loader import season_label

st.set_page_config(page_title="NBA V2 Walk-Forward Backtest", layout="wide")
st.title("NBA V2 — Walk-Forward Validation Sandbox")
st.caption(
    "BATCHED C3: identical frozen model logic, but the full season is evaluated in short resumable batches so "
    "Streamlit Cloud does not need one 20–30 minute continuous run."
)

@st.cache_data(show_spinner=False)
def load_season(season: int):
    return SportsDataverseNBA(timeout=60).load_season(int(season))


def _empty_state():
    st.session_state["bt_acc_detail"] = pd.DataFrame()
    st.session_state["bt_acc_prob"] = pd.DataFrame()
    st.session_state["bt_acc_most"] = pd.DataFrame()
    st.session_state["bt_acc_fail"] = pd.DataFrame()
    st.session_state["bt_next_index"] = 0
    st.session_state["bt_runtime_total"] = 0.0
    st.session_state["bt_batch_signature"] = None


def _append(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if new is None or new.empty:
        return old if isinstance(old, pd.DataFrame) else pd.DataFrame()
    if old is None or old.empty:
        return new.reset_index(drop=True)
    return pd.concat([old, new], ignore_index=True)


def _checkpoint_bytes(detail, prob, most, fail, meta) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("detail.csv", detail.to_csv(index=False))
        z.writestr("probability_rows.csv", prob.to_csv(index=False))
        z.writestr("most_rows.csv", most.to_csv(index=False))
        z.writestr("failures.csv", fail.to_csv(index=False))
        z.writestr("meta.json", json.dumps(meta, indent=2))
    return buf.getvalue()


def _restore_checkpoint(raw: bytes):
    with zipfile.ZipFile(io.BytesIO(raw), "r") as z:
        def read_csv(name):
            with z.open(name) as f:
                try:
                    return pd.read_csv(f)
                except pd.errors.EmptyDataError:
                    return pd.DataFrame()
        detail = read_csv("detail.csv")
        prob = read_csv("probability_rows.csv")
        most = read_csv("most_rows.csv")
        fail = read_csv("failures.csv")
        meta = json.loads(z.read("meta.json").decode("utf-8"))
    return detail, prob, most, fail, meta


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
    a, b, c = st.columns(3)
    n_sims = a.select_slider("Simulations per historical game", options=[1000,1500,2500,5000,10000], value=1500)
    batch_size = b.number_input("Games per batch", min_value=20, max_value=300, value=100, step=20)
    rotation = c.toggle("Pregame rotation-similarity weighting", value=True)
    st.caption("Recommended: 100 games/batch. Each click is a short independent slice; accumulated results are identical in logic to a one-shot run.")

if st.button("1) Load seasons + fit TRAIN calibration", type="primary"):
    with st.spinner("Loading SportsDataverse seasons and fitting frozen league-level parameters..."):
        train_pack = load_season(int(train_season))
        val_pack = load_season(int(validation_season))
        cal = fit_league_calibration(train_pack["team"], int(train_season))
        st.session_state["bt_train_pack"] = train_pack
        st.session_state["bt_val_pack"] = val_pack
        st.session_state["bt_cal"] = cal
        _empty_state()
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
    eligible = eligible_game_count(val_pack["team"], min_prior_games=int(min_games))
    signature = {
        "train": int(train_season), "validation": int(validation_season), "min_games": int(min_games),
        "n_sims": int(n_sims), "rotation": bool(rotation), "eligible": int(eligible),
    }
    saved_sig = st.session_state.get("bt_batch_signature")
    next_index = int(st.session_state.get("bt_next_index", 0))
    done_games = min(next_index, eligible)
    st.info(f"Eligible games: {eligible}. Completed/queued through index: {done_games}. Remaining: {max(eligible-done_games,0)}.")

    if saved_sig is not None and saved_sig != signature and next_index > 0:
        st.error("Run settings changed after batches were accumulated. Reset the accumulated validation before continuing.")
    else:
        if saved_sig is None:
            st.session_state["bt_batch_signature"] = signature

        left, right = st.columns([2,1])
        run_batch = left.button("2) Run NEXT validation batch", type="primary", disabled=(next_index >= eligible))
        reset = right.button("Reset accumulated validation")
        if reset:
            _empty_state()
            st.session_state["bt_batch_signature"] = signature
            st.rerun()

        if run_batch:
            bar = st.progress(0.0, text=f"Preparing games {next_index+1} onward...")
            def prog(done, total):
                bar.progress(done/max(total,1), text=f"This batch: {done}/{total} | season index starts at {next_index}")
            with st.spinner("Running one resumable historical batch..."):
                res = run_walk_forward(
                    val_pack["team"], val_pack["player"], cal,
                    min_prior_games=int(min_games), n_sims=int(n_sims),
                    max_games=int(batch_size), start_index=int(next_index),
                    use_rotation_similarity=bool(rotation), progress_callback=prog,
                )
            st.session_state["bt_acc_detail"] = _append(st.session_state.get("bt_acc_detail"), res.detail)
            st.session_state["bt_acc_prob"] = _append(st.session_state.get("bt_acc_prob"), res.probability_rows)
            st.session_state["bt_acc_most"] = _append(st.session_state.get("bt_acc_most"), res.most_rows)
            st.session_state["bt_acc_fail"] = _append(st.session_state.get("bt_acc_fail"), res.failures)
            st.session_state["bt_next_index"] = int(res.end_index)
            st.session_state["bt_runtime_total"] = float(st.session_state.get("bt_runtime_total",0.0)) + float(res.runtime_seconds)
            st.session_state["bt_batch_signature"] = signature
            bar.empty()
            st.rerun()

    # Optional restore after a Streamlit session restart.
    with st.expander("Checkpoint / restore", expanded=False):
        up = st.file_uploader("Restore a C3 checkpoint ZIP", type=["zip"], key="bt_restore_zip")
        if up is not None and st.button("Restore checkpoint"):
            d,p,m,f,meta = _restore_checkpoint(up.getvalue())
            if meta.get("signature") != signature:
                st.error("Checkpoint settings do not match the current train/validation/min-games/sims/rotation settings.")
            else:
                st.session_state["bt_acc_detail"] = d
                st.session_state["bt_acc_prob"] = p
                st.session_state["bt_acc_most"] = m
                st.session_state["bt_acc_fail"] = f
                st.session_state["bt_next_index"] = int(meta.get("next_index",0))
                st.session_state["bt_runtime_total"] = float(meta.get("runtime_total",0.0))
                st.session_state["bt_batch_signature"] = signature
                st.success("Checkpoint restored.")
                st.rerun()

    detail = st.session_state.get("bt_acc_detail", pd.DataFrame())
    prob = st.session_state.get("bt_acc_prob", pd.DataFrame())
    most = st.session_state.get("bt_acc_most", pd.DataFrame())
    fail = st.session_state.get("bt_acc_fail", pd.DataFrame())
    next_index = int(st.session_state.get("bt_next_index",0))

    if isinstance(detail, pd.DataFrame) and not detail.empty:
        projected_games = detail["GAME_ID"].astype(str).nunique()
        st.success(
            f"Accumulated projected games: {projected_games}/{eligible}; failures: {len(fail)}; "
            f"compute time: {float(st.session_state.get('bt_runtime_total',0.0))/60:.1f} min."
        )
        metrics = metric_table(detail)
        st.subheader("A. Center + distribution validation — accumulated batches")
        st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)

        st.subheader("B. Probability calibration — accumulated synthetic pregame lines")
        over_bins = calibration_bins(prob, "Over")
        if not over_bins.empty:
            st.dataframe(over_bins.round(4), use_container_width=True, hide_index=True)

        st.subheader("C. Team-with-most calibration — accumulated")
        st.dataframe(most_summary(most).round(4), use_container_width=True, hide_index=True)

        if isinstance(fail, pd.DataFrame) and not fail.empty:
            with st.expander("Failures / skipped projections"):
                st.dataframe(fail, use_container_width=True, hide_index=True)

        meta = {
            "signature": signature,
            "next_index": next_index,
            "runtime_total": float(st.session_state.get("bt_runtime_total",0.0)),
        }
        st.download_button(
            "Download resumable checkpoint ZIP",
            _checkpoint_bytes(detail, prob, most, fail, meta),
            file_name=f"nba_backtest_checkpoint_{next_index}_of_{eligible}.zip",
            mime="application/zip",
        )
        if next_index >= eligible:
            st.success("FULL VALIDATION COMPLETE. Freeze/tune only on this 2024-25 result before unlocking 2025-26.")
            d1,d2,d3 = st.columns(3)
            d1.download_button("Detailed game-market rows", detail.to_csv(index=False).encode(), "nba_backtest_detail.csv", "text/csv")
            d2.download_button("Probability calibration rows", prob.to_csv(index=False).encode(), "nba_backtest_probability_rows.csv", "text/csv")
            d3.download_button("Most-market rows", most.to_csv(index=False).encode(), "nba_backtest_most_rows.csv", "text/csv")

st.divider()
st.warning(
    "Protocol guard: do NOT use 2025-26 to tune parameters after looking at its final-test results. "
    "First tune/approve on 2024-25; only then run 2025-26 once as the untouched exam."
)
