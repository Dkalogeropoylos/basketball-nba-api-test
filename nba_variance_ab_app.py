from __future__ import annotations

import io
import json
import zipfile

import pandas as pd
import streamlit as st

from providers.sportsdataverse_nba import SportsDataverseNBA
from backtest.calibration import fit_league_calibration
from backtest.engine import run_walk_forward, eligible_game_count
from backtest.evaluation import metric_table, calibration_bins, most_summary
from backtest.data_loader import season_label
from backtest.variance_experiment import VARIANTS, late_half_start_index


st.set_page_config(page_title="NBA V2 C4 Variance A/B", layout="wide")
st.title("NBA V2 — Phase C4 Variance A/B Sandbox")
st.caption(
    "Mean architecture and 2023-24 league calibration remain frozen. C4 changes ONLY the stochastic data-generating layer. "
    "The first half of 2024-25 is development/calibration; the late half is the untouched A/B evaluation set. 2025-26 stays locked."
)


@st.cache_data(show_spinner=False)
def load_season(season: int):
    return SportsDataverseNBA(timeout=60).load_season(int(season))


def _state_prefix(variant_key: str) -> str:
    return f"c4_{variant_key}_"


def _empty_variant_state(variant_key: str, start_index: int):
    p = _state_prefix(variant_key)
    st.session_state[p + "detail"] = pd.DataFrame()
    st.session_state[p + "prob"] = pd.DataFrame()
    st.session_state[p + "most"] = pd.DataFrame()
    st.session_state[p + "fail"] = pd.DataFrame()
    st.session_state[p + "next_index"] = int(start_index)
    st.session_state[p + "runtime"] = 0.0
    st.session_state[p + "signature"] = None


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
        return (
            read_csv("detail.csv"), read_csv("probability_rows.csv"),
            read_csv("most_rows.csv"), read_csv("failures.csv"),
            json.loads(z.read("meta.json").decode("utf-8")),
        )


st.warning("2025-26 FINAL TEST is deliberately unavailable in this app.")

c1, c2, c3 = st.columns(3)
train_season = c1.selectbox("TRAIN league parameters", [2024], format_func=lambda x: season_label(x))
validation_season = c2.selectbox("DEVELOPMENT / A-B season", [2025], format_func=lambda x: season_label(x))
min_games = c3.number_input("Minimum prior current-season games/team", min_value=8, max_value=30, value=15, step=1)

variant_labels = {v.label: k for k, v in VARIANTS.items()}
variant_label = st.selectbox(
    "Variance variant",
    list(variant_labels.keys()),
    index=1,  # start scientifically with FGA-chain only
)
variant = VARIANTS[variant_labels[variant_label]]
st.info(variant.description)

with st.expander("Run settings", expanded=True):
    a, b, c = st.columns(3)
    n_sims = a.select_slider("Simulations per historical game", options=[1000,1500,2500,5000], value=1500)
    batch_size = b.number_input("Games per batch", min_value=20, max_value=200, value=100, step=20)
    rotation = c.toggle("Pregame rotation-similarity weighting", value=True)

with st.expander("C4 mathematical audit", expanded=False):
    st.markdown(
        """
**FGA B1 — possession/rebound chain**

The frozen baseline draws `FGA ~ Poisson(FGA_mean)` after pace, TOV, FTA and OREB-related state have already made `FGA_mean` stochastic. B1 removes that extra full-count redraw. It starts from the possession identity

`POSS ≈ FGA - OREB + TOV + 0.44*FTA`

and simulates initial shot endings; every realized offensive rebound creates exactly one continuation shot opportunity. Randomized rounding preserves the expected initial count. There is no fitted B1 variance parameter.

**FTA B2 — mean-preserving Poisson-lognormal intensity**

`lambda = mu * exp(sigma*z - sigma^2/2)`, so the multiplicative shock has expectation 1. The C3 baseline uses `sigma=0.12`. The C4 development value `sigma=0.205681` was estimated by method of moments using ONLY the first 497 eligible 2024-25 games (994 team rows); the final 498 games are untouched for evaluation.

B1 and B2 are deliberately separable. Do not accept B3 merely because it is combined: each component should first improve its intended distribution without damaging centers or unrelated markets.
        """
    )

if st.button("1) Load seasons + fit frozen TRAIN calibration", type="primary"):
    with st.spinner("Loading SportsDataverse seasons and fitting frozen 2023-24 league parameters..."):
        train_pack = load_season(int(train_season))
        val_pack = load_season(int(validation_season))
        cal = fit_league_calibration(train_pack["team"], int(train_season))
        st.session_state["c4_train_pack"] = train_pack
        st.session_state["c4_val_pack"] = val_pack
        st.session_state["c4_cal"] = cal
        eligible = eligible_game_count(val_pack["team"], min_prior_games=int(min_games))
        split = late_half_start_index(eligible)
        for key in VARIANTS:
            _empty_variant_state(key, split)
    st.success(
        f"Loaded {season_label(train_season)} TRAIN + {season_label(validation_season)} development. "
        f"Eligible games={eligible}; early calibration half={split}; untouched A/B half={eligible-split}."
    )

cal = st.session_state.get("c4_cal")
val_pack = st.session_state.get("c4_val_pack")

if cal is not None:
    with st.expander("Frozen 2023-24 TRAIN calibration audit", expanded=False):
        st.markdown("**Structural rates**")
        st.dataframe(cal.structural_audit.round(4), use_container_width=True, hide_index=True)
        st.markdown("**Shooting efficiency**")
        st.dataframe(cal.shooting_audit.round(4), use_container_width=True, hide_index=True)
        st.markdown("**Legacy/generic opponent elasticities**")
        st.dataframe(cal.opponent_audit.round(4), use_container_width=True, hide_index=True)

if cal is not None and val_pack is not None:
    eligible = eligible_game_count(val_pack["team"], min_prior_games=int(min_games))
    split = late_half_start_index(eligible)
    eval_games = eligible - split
    p = _state_prefix(variant.key)
    if p + "next_index" not in st.session_state:
        _empty_variant_state(variant.key, split)

    signature = {
        "variant": variant.key,
        "train": int(train_season), "validation": int(validation_season),
        "min_games": int(min_games), "n_sims": int(n_sims), "rotation": bool(rotation),
        "eligible": int(eligible), "split": int(split),
        "fga_process": variant.fga_process, "fta_log_sigma": float(variant.fta_log_sigma),
    }
    next_index = int(st.session_state.get(p + "next_index", split))
    completed = max(0, min(next_index, eligible) - split)
    st.info(
        f"Eligible season games: {eligible}. Development split: first {split} calibrate / last {eval_games} evaluate. "
        f"Variant {variant.label}: completed {completed}/{eval_games}; remaining {max(eval_games-completed,0)}."
    )

    saved_sig = st.session_state.get(p + "signature")
    if saved_sig is not None and saved_sig != signature and completed > 0:
        st.error("Settings changed after this variant accumulated batches. Reset this variant before continuing.")
    else:
        if saved_sig is None:
            st.session_state[p + "signature"] = signature

        left, right = st.columns([2,1])
        run_batch = left.button("2) Run NEXT untouched late-half A/B batch", type="primary", disabled=(next_index >= eligible))
        reset = right.button("Reset selected variant")
        if reset:
            _empty_variant_state(variant.key, split)
            st.session_state[p + "signature"] = signature
            st.rerun()

        if run_batch:
            bar = st.progress(0.0, text=f"Preparing absolute season index {next_index}...")
            def prog(done, total):
                bar.progress(done/max(total,1), text=f"This batch: {done}/{total} | absolute start index {next_index}")
            with st.spinner(f"Running {variant.label} on untouched late-half games..."):
                res = run_walk_forward(
                    val_pack["team"], val_pack["player"], cal,
                    min_prior_games=int(min_games), n_sims=int(n_sims),
                    max_games=int(batch_size), start_index=int(next_index),
                    use_rotation_similarity=bool(rotation),
                    fga_process=variant.fga_process,
                    fta_log_sigma=float(variant.fta_log_sigma),
                    progress_callback=prog,
                )
            st.session_state[p + "detail"] = _append(st.session_state.get(p + "detail"), res.detail)
            st.session_state[p + "prob"] = _append(st.session_state.get(p + "prob"), res.probability_rows)
            st.session_state[p + "most"] = _append(st.session_state.get(p + "most"), res.most_rows)
            st.session_state[p + "fail"] = _append(st.session_state.get(p + "fail"), res.failures)
            st.session_state[p + "next_index"] = int(res.end_index)
            st.session_state[p + "runtime"] = float(st.session_state.get(p + "runtime",0.0)) + float(res.runtime_seconds)
            st.session_state[p + "signature"] = signature
            bar.empty()
            st.rerun()

    with st.expander("Checkpoint / restore selected variant", expanded=False):
        up = st.file_uploader("Restore C4 variant checkpoint ZIP", type=["zip"], key=f"restore_{variant.key}")
        if up is not None and st.button("Restore selected-variant checkpoint"):
            d, pr, m, f, meta = _restore_checkpoint(up.getvalue())
            if meta.get("signature") != signature:
                st.error("Checkpoint signature does not match current variant/settings.")
            else:
                st.session_state[p + "detail"] = d
                st.session_state[p + "prob"] = pr
                st.session_state[p + "most"] = m
                st.session_state[p + "fail"] = f
                st.session_state[p + "next_index"] = int(meta.get("next_index", split))
                st.session_state[p + "runtime"] = float(meta.get("runtime_total", 0.0))
                st.session_state[p + "signature"] = signature
                st.success("Checkpoint restored.")
                st.rerun()

    detail = st.session_state.get(p + "detail", pd.DataFrame())
    prob = st.session_state.get(p + "prob", pd.DataFrame())
    most = st.session_state.get(p + "most", pd.DataFrame())
    fail = st.session_state.get(p + "fail", pd.DataFrame())
    next_index = int(st.session_state.get(p + "next_index", split))

    if isinstance(detail, pd.DataFrame) and not detail.empty:
        projected_games = detail["GAME_ID"].astype(str).nunique()
        st.success(
            f"{variant.label}: projected {projected_games}/{eval_games} late-half games; failures={len(fail)}; "
            f"compute time={float(st.session_state.get(p+'runtime',0.0))/60:.1f} min."
        )
        st.subheader("A. Center + distribution validation — selected variant")
        st.dataframe(metric_table(detail).round(4), use_container_width=True, hide_index=True)

        st.subheader("B. Probability calibration — selected variant")
        bins = calibration_bins(prob)
        st.dataframe(bins.round(4), use_container_width=True, hide_index=True)

        st.subheader("C. Most-market calibration — selected variant")
        st.dataframe(most_summary(most).round(4), use_container_width=True, hide_index=True)

        meta = {
            "signature": signature,
            "next_index": int(next_index),
            "runtime_total": float(st.session_state.get(p + "runtime",0.0)),
        }
        st.download_button(
            "Download resumable C4 checkpoint ZIP",
            _checkpoint_bytes(detail, prob, most, fail, meta),
            file_name=f"nba_c4_{variant.key}_checkpoint_{projected_games}games.zip",
            mime="application/zip",
        )
        st.download_button("Download C4 detail CSV", detail.to_csv(index=False), file_name=f"nba_c4_{variant.key}_detail.csv", mime="text/csv")
        st.download_button("Download C4 probability rows CSV", prob.to_csv(index=False), file_name=f"nba_c4_{variant.key}_probability_rows.csv", mime="text/csv")
        st.download_button("Download C4 most rows CSV", most.to_csv(index=False), file_name=f"nba_c4_{variant.key}_most_rows.csv", mime="text/csv")
        if isinstance(fail, pd.DataFrame) and not fail.empty:
            st.error("Failures detected — do not interpret A/B metrics until failures are resolved.")
            st.dataframe(fail, use_container_width=True, hide_index=True)
