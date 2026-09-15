from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from providers.sportsdataverse_nba import SportsDataverseNBA
from backtest.calibration import fit_league_calibration
from backtest.engine import run_walk_forward, eligible_game_count
from backtest.evaluation import metric_table, calibration_bins, most_summary
from backtest.data_loader import season_label
from backtest.variance_experiment import VARIANTS, late_half_start_index
from backtest.shared_line_eval import (
    B3_FGA_PROCESS,
    B3_FTA_LOG_SIGMA,
    checkpoint_bytes as shared_checkpoint_bytes,
    load_fixed_a_lines,
    reference_audit,
    restore_checkpoint as restore_shared_checkpoint,
    run_shared_line_batch,
    shared_line_calibration_bins,
    shared_line_summary,
)
from backtest.final_exam import (
    FINAL_MIN_PRIOR_GAMES,
    FINAL_N_SIMS,
    FINAL_ROTATION,
    center_head_to_head as final_center_head_to_head,
    checkpoint_bytes as final_checkpoint_bytes,
    final_most_summary,
    final_shared_calibration,
    final_shared_summary,
    restore_checkpoint as restore_final_checkpoint,
    run_final_exam_batch,
    variant_metric_table as final_variant_metric_table,
)


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


st.info("2025-26 FINAL TEST stays locked until you explicitly freeze B3 in Section E. The 2024-25 controls above remain development-only.")

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

    # ---------------------------------------------------------------------
    # D. FINAL 2024-25 SHARED-LINE PROBABILITY TEST
    # ---------------------------------------------------------------------
    st.divider()
    st.subheader("D. Shared-line probability A/B — SAME exact lines")
    st.caption(
        "This is the final 2024-25 probability test before the untouched 2025-26 exam. "
        "Frozen baseline A is NOT rerun. B3 is rerun only to evaluate P(Over/Under) at the exact same A half-point lines. "
        "Because several lines come from one game, uncertainty is clustered by GAME_ID."
    )

    fixed_path = Path(__file__).resolve().parent / "nba_c5_A_fixed_lines_late498.csv.gz"
    if not fixed_path.exists():
        st.error(
            "Missing nba_c5_A_fixed_lines_late498.csv.gz in the repo root. "
            "Upload the bundled frozen-A reference file before running section D."
        )
    else:
        a_lines = load_fixed_a_lines(fixed_path)
        audit = reference_audit(a_lines)
        st.info(
            f"Frozen A reference: {audit['games']} late-half games, {audit['rows']:,} exact lines; "
            f"non-half lines={audit['non_half_lines']}, rows with push probability>0={audit['push_rows']}. "
            "Expected: 498 games, only half-point lines, zero pushes."
        )

        shared_sig = {
            "train": int(train_season),
            "validation": int(validation_season),
            "min_games": 15,
            "n_sims": 1500,
            "rotation": True,
            "eligible": int(eligible),
            "split": int(split),
            "fga_process": B3_FGA_PROCESS,
            "fta_log_sigma": float(B3_FTA_LOG_SIGMA),
            "a_reference_games": int(audit["games"]),
            "a_reference_rows": int(audit["rows"]),
        }

        if "c5_next_index" not in st.session_state:
            st.session_state["c5_rows"] = pd.DataFrame()
            st.session_state["c5_fail"] = pd.DataFrame()
            st.session_state["c5_next_index"] = int(split)
            st.session_state["c5_signature"] = shared_sig

        saved_shared_sig = st.session_state.get("c5_signature")
        shared_next = int(st.session_state.get("c5_next_index", split))
        shared_done = max(0, min(shared_next, eligible) - split)

        if saved_shared_sig is not None and saved_shared_sig != shared_sig and shared_done > 0:
            st.error("Shared-line settings/signature changed after batches were accumulated. Reset section D before continuing.")
        else:
            if saved_shared_sig is None:
                st.session_state["c5_signature"] = shared_sig

            c5a, c5b, c5c = st.columns([1, 1, 1])
            c5a.metric("Shared-line completed", f"{shared_done}/{eval_games}")
            c5b.metric("B3 sims/game", "1500")
            shared_batch_size = c5c.number_input(
                "Shared-line games per batch",
                min_value=20,
                max_value=200,
                value=100,
                step=20,
                key="c5_batch_size",
            )

            left_d, right_d = st.columns([2, 1])
            run_shared = left_d.button(
                "3) Run NEXT B3 shared-line batch",
                type="primary",
                disabled=(shared_next >= eligible),
                key="run_c5_shared",
            )
            reset_shared = right_d.button("Reset shared-line test", key="reset_c5_shared")

            if reset_shared:
                st.session_state["c5_rows"] = pd.DataFrame()
                st.session_state["c5_fail"] = pd.DataFrame()
                st.session_state["c5_next_index"] = int(split)
                st.session_state["c5_signature"] = shared_sig
                st.rerun()

            if run_shared:
                bar_d = st.progress(0.0, text=f"Preparing shared-line season index {shared_next}...")

                def prog_shared(done, total):
                    bar_d.progress(
                        done / max(total, 1),
                        text=f"Shared-line batch: {done}/{total} | absolute start index {shared_next}",
                    )

                with st.spinner("Rerunning B3 only and scoring the frozen A lines..."):
                    rr, ff, end_idx, _ = run_shared_line_batch(
                        val_pack["team"],
                        val_pack["player"],
                        cal,
                        a_lines,
                        start_index=int(shared_next),
                        max_games=int(shared_batch_size),
                        min_prior_games=15,
                        n_sims=1500,
                        use_rotation_similarity=True,
                        progress_callback=prog_shared,
                    )
                st.session_state["c5_rows"] = _append(st.session_state.get("c5_rows"), rr)
                st.session_state["c5_fail"] = _append(st.session_state.get("c5_fail"), ff)
                st.session_state["c5_next_index"] = int(end_idx)
                st.session_state["c5_signature"] = shared_sig
                bar_d.empty()
                st.rerun()

        with st.expander("Shared-line checkpoint / restore", expanded=False):
            up_shared = st.file_uploader(
                "Restore shared-line checkpoint ZIP", type=["zip"], key="restore_c5_integrated"
            )
            if up_shared is not None and st.button("Restore shared-line checkpoint", key="restore_c5_btn"):
                sr, sf, smeta = restore_shared_checkpoint(up_shared.getvalue())
                if smeta.get("signature") != shared_sig:
                    st.error("Shared-line checkpoint signature does not match the frozen settings/reference.")
                else:
                    st.session_state["c5_rows"] = sr
                    st.session_state["c5_fail"] = sf
                    st.session_state["c5_next_index"] = int(smeta.get("next_index", split))
                    st.session_state["c5_signature"] = shared_sig
                    st.success("Shared-line checkpoint restored.")
                    st.rerun()

        shared_rows = st.session_state.get("c5_rows", pd.DataFrame())
        shared_fail = st.session_state.get("c5_fail", pd.DataFrame())
        shared_next = int(st.session_state.get("c5_next_index", split))

        if isinstance(shared_rows, pd.DataFrame) and not shared_rows.empty:
            shared_games = shared_rows["GAME_ID"].astype(str).nunique()
            st.success(
                f"B3 scored on frozen A lines for {shared_games}/{eval_games} games; failures={len(shared_fail)}."
            )

            summary_shared = shared_line_summary(shared_rows)
            st.markdown("**Paired scoring on identical lines**")
            st.dataframe(summary_shared.round(5), use_container_width=True, hide_index=True)
            st.caption(
                "For Δ columns, negative B3-A is better for B3. The Brier 95% interval is bootstrapped after clustering rows by GAME_ID."
            )

            tab_a, tab_b = st.tabs(["Calibration — A", "Calibration — B3"])
            with tab_a:
                st.dataframe(
                    shared_line_calibration_bins(shared_rows, "P_A_Over").round(4),
                    use_container_width=True,
                    hide_index=True,
                )
            with tab_b:
                st.dataframe(
                    shared_line_calibration_bins(shared_rows, "P_B3_Over").round(4),
                    use_container_width=True,
                    hide_index=True,
                )

            shared_meta = {
                "signature": shared_sig,
                "next_index": int(shared_next),
            }
            st.download_button(
                "Download shared-line checkpoint ZIP",
                shared_checkpoint_bytes(shared_rows, shared_fail, shared_meta),
                file_name=f"nba_c5_shared_lines_{shared_games}games.zip",
                mime="application/zip",
                key="download_c5_checkpoint",
            )
            st.download_button(
                "Download shared-line detail CSV",
                shared_rows.to_csv(index=False),
                file_name="nba_c5_shared_line_rows.csv",
                mime="text/csv",
                key="download_c5_rows",
            )
            st.download_button(
                "Download shared-line summary CSV",
                summary_shared.to_csv(index=False),
                file_name="nba_c5_shared_line_summary.csv",
                mime="text/csv",
                key="download_c5_summary",
            )

            if isinstance(shared_fail, pd.DataFrame) and not shared_fail.empty:
                st.error("Shared-line failures detected — do not interpret probability A/B until resolved.")
                st.dataframe(shared_fail, use_container_width=True, hide_index=True)



# -------------------------------------------------------------------------
# E. UNTOUCHED 2025-26 FINAL EXAM — FROZEN A vs FROZEN B3
# -------------------------------------------------------------------------
st.divider()
st.subheader("E. UNTOUCHED 2025-26 FINAL EXAM — frozen B3")
st.caption(
    "This is the one-shot generalization exam. No 2025-26 result is used to tune parameters. "
    "The 2023-24 league calibration, B3 FGA rebound-chain, B3 FTA dispersion, 15-game minimum, "
    "1500 simulations and rotation weighting are all frozen before the season is loaded."
)
st.warning(
    "Protocol guard: once you inspect 2025-26 results, do NOT change the architecture or parameters because of them. "
    "Any later production refit must reuse the validated methodology and be evaluated on future data."
)

confirm_final = st.checkbox(
    "I confirm B3 is frozen from the 2024-25 development/A-B process and I will not tune it from 2025-26 results.",
    key="c6_confirm_final",
)
freeze_phrase = st.text_input(
    "Type FREEZE B3 to unlock the final exam",
    key="c6_freeze_phrase",
    placeholder="FREEZE B3",
)
final_unlocked = bool(confirm_final and freeze_phrase.strip().upper() == "FREEZE B3")

if cal is None:
    st.info("First use button 1 above to load 2023-24 and fit the frozen TRAIN calibration.")
else:
    load_final = st.button(
        "4) Load untouched 2025-26 FINAL season",
        type="primary",
        disabled=not final_unlocked,
        key="c6_load_final",
    )
    if load_final:
        with st.spinner("Loading untouched 2025-26 regular-season data..."):
            final_pack = load_season(2026)
            final_eligible = eligible_game_count(
                final_pack["team"], min_prior_games=FINAL_MIN_PRIOR_GAMES
            )
            st.session_state["c6_final_pack"] = final_pack
            st.session_state["c6_detail"] = pd.DataFrame()
            st.session_state["c6_shared"] = pd.DataFrame()
            st.session_state["c6_most"] = pd.DataFrame()
            st.session_state["c6_fail"] = pd.DataFrame()
            st.session_state["c6_next_index"] = 0
            st.session_state["c6_runtime"] = 0.0
            st.session_state["c6_signature"] = {
                "train_calibration": "2023-24 frozen",
                "final_season": "2025-26",
                "min_prior_games": FINAL_MIN_PRIOR_GAMES,
                "n_sims": FINAL_N_SIMS,
                "rotation_similarity": FINAL_ROTATION,
                "A_fga_process": "poisson",
                "A_fta_log_sigma": 0.12,
                "B3_fga_process": B3_FGA_PROCESS,
                "B3_fta_log_sigma": float(B3_FTA_LOG_SIGMA),
                "eligible_games": int(final_eligible),
            }
        st.success(
            f"FINAL season loaded. Eligible 2025-26 games after the fixed {FINAL_MIN_PRIOR_GAMES}-game minimum: {final_eligible}."
        )
        st.rerun()

final_pack = st.session_state.get("c6_final_pack")
if cal is not None and final_pack is not None:
    final_eligible = eligible_game_count(
        final_pack["team"], min_prior_games=FINAL_MIN_PRIOR_GAMES
    )
    final_sig = {
        "train_calibration": "2023-24 frozen",
        "final_season": "2025-26",
        "min_prior_games": FINAL_MIN_PRIOR_GAMES,
        "n_sims": FINAL_N_SIMS,
        "rotation_similarity": FINAL_ROTATION,
        "A_fga_process": "poisson",
        "A_fta_log_sigma": 0.12,
        "B3_fga_process": B3_FGA_PROCESS,
        "B3_fta_log_sigma": float(B3_FTA_LOG_SIGMA),
        "eligible_games": int(final_eligible),
    }

    if st.session_state.get("c6_signature") != final_sig:
        st.error("FINAL-test signature mismatch. Do not continue; reload the final season with the frozen settings.")
    else:
        next_final = int(st.session_state.get("c6_next_index", 0))
        done_final = min(next_final, final_eligible)
        st.info(
            f"FINAL 2025-26: completed {done_final}/{final_eligible}; remaining {max(final_eligible-done_final, 0)}. "
            f"Frozen protocol: min prior={FINAL_MIN_PRIOR_GAMES}, sims/game={FINAL_N_SIMS}, rotation=ON, variants=A and B3."
        )

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Eligible final games", f"{final_eligible}")
        cc2.metric("Completed", f"{done_final}")
        final_batch_size = cc3.number_input(
            "FINAL games per batch",
            min_value=20,
            max_value=200,
            value=100,
            step=20,
            key="c6_batch_size",
        )

        run_final = st.button(
            "5) Run NEXT untouched FINAL batch (A + B3 + same A lines)",
            type="primary",
            disabled=(next_final >= final_eligible),
            key="c6_run_final",
        )
        if run_final:
            bar_f = st.progress(0.0, text=f"Preparing FINAL season index {next_final}...")

            def prog_final(done, total):
                bar_f.progress(
                    done / max(total, 1),
                    text=f"FINAL batch: {done}/{total} | absolute start index {next_final}",
                )

            with st.spinner("Running frozen A and frozen B3 on untouched 2025-26..."):
                fr = run_final_exam_batch(
                    final_pack["team"],
                    final_pack["player"],
                    cal,
                    start_index=int(next_final),
                    max_games=int(final_batch_size),
                    progress_callback=prog_final,
                )
            st.session_state["c6_detail"] = _append(st.session_state.get("c6_detail"), fr["detail"])
            st.session_state["c6_shared"] = _append(st.session_state.get("c6_shared"), fr["shared"])
            st.session_state["c6_most"] = _append(st.session_state.get("c6_most"), fr["most"])
            st.session_state["c6_fail"] = _append(st.session_state.get("c6_fail"), fr["failures"])
            st.session_state["c6_next_index"] = int(fr["end_index"])
            st.session_state["c6_runtime"] = float(st.session_state.get("c6_runtime", 0.0)) + float(fr["runtime_seconds"])
            bar_f.empty()
            st.rerun()

        with st.expander("FINAL checkpoint / restore", expanded=False):
            up_final = st.file_uploader("Restore FINAL checkpoint ZIP", type=["zip"], key="restore_c6_final")
            if up_final is not None and st.button("Restore FINAL checkpoint", key="restore_c6_final_btn"):
                fd, fs, fm, ff, fmeta = restore_final_checkpoint(up_final.getvalue())
                if fmeta.get("signature") != final_sig:
                    st.error("FINAL checkpoint signature does not match the frozen final-test protocol.")
                else:
                    st.session_state["c6_detail"] = fd
                    st.session_state["c6_shared"] = fs
                    st.session_state["c6_most"] = fm
                    st.session_state["c6_fail"] = ff
                    st.session_state["c6_next_index"] = int(fmeta.get("next_index", 0))
                    st.session_state["c6_runtime"] = float(fmeta.get("runtime_total", 0.0))
                    st.session_state["c6_signature"] = final_sig
                    st.success("FINAL checkpoint restored.")
                    st.rerun()

        final_detail = st.session_state.get("c6_detail", pd.DataFrame())
        final_shared = st.session_state.get("c6_shared", pd.DataFrame())
        final_most = st.session_state.get("c6_most", pd.DataFrame())
        final_fail = st.session_state.get("c6_fail", pd.DataFrame())
        next_final = int(st.session_state.get("c6_next_index", 0))

        if isinstance(final_detail, pd.DataFrame) and not final_detail.empty:
            final_games_done = final_detail["GAME_ID"].astype(str).nunique()
            st.success(
                f"FINAL accumulated games: {final_games_done}/{final_eligible}; failures={len(final_fail)}; "
                f"compute time={float(st.session_state.get('c6_runtime',0.0))/60:.1f} min."
            )

            st.markdown("**E1. Center + distribution — frozen A vs frozen B3**")
            st.dataframe(final_variant_metric_table(final_detail).round(5), use_container_width=True, hide_index=True)

            st.markdown("**E2. Same-game center head-to-head**")
            st.dataframe(final_center_head_to_head(final_detail).round(5), use_container_width=True, hide_index=True)

            st.markdown("**E3. Probability A/B on the SAME A-defined half-point lines**")
            final_prob_summary = final_shared_summary(final_shared)
            st.dataframe(final_prob_summary.round(5), use_container_width=True, hide_index=True)
            st.caption("Negative B3-A is better for B3. Lines are generated from A pregame simulations only; actual outcomes never define the lines.")

            tab_fa, tab_fb = st.tabs(["FINAL calibration — A", "FINAL calibration — B3"])
            with tab_fa:
                st.dataframe(final_shared_calibration(final_shared, "A").round(4), use_container_width=True, hide_index=True)
            with tab_fb:
                st.dataframe(final_shared_calibration(final_shared, "B3").round(4), use_container_width=True, hide_index=True)

            st.markdown("**E4. Most-market calibration — frozen A vs frozen B3**")
            st.dataframe(final_most_summary(final_most).round(5), use_container_width=True, hide_index=True)

            final_meta = {
                "signature": final_sig,
                "next_index": int(next_final),
                "runtime_total": float(st.session_state.get("c6_runtime", 0.0)),
            }
            st.download_button(
                "Download resumable FINAL checkpoint ZIP",
                final_checkpoint_bytes(final_detail, final_shared, final_most, final_fail, final_meta),
                file_name=f"nba_c6_final_2025_26_{final_games_done}games.zip",
                mime="application/zip",
                key="download_c6_checkpoint",
            )
            st.download_button(
                "Download FINAL detail CSV",
                final_detail.to_csv(index=False),
                file_name="nba_c6_final_detail.csv",
                mime="text/csv",
                key="download_c6_detail",
            )
            st.download_button(
                "Download FINAL shared-line rows CSV",
                final_shared.to_csv(index=False),
                file_name="nba_c6_final_shared_line_rows.csv",
                mime="text/csv",
                key="download_c6_shared",
            )
            st.download_button(
                "Download FINAL Most rows CSV",
                final_most.to_csv(index=False),
                file_name="nba_c6_final_most_rows.csv",
                mime="text/csv",
                key="download_c6_most",
            )

            if isinstance(final_fail, pd.DataFrame) and not final_fail.empty:
                st.error("FINAL-test failures detected. Do not interpret the exam until every failure is resolved.")
                st.dataframe(final_fail, use_container_width=True, hide_index=True)

            if next_final >= final_eligible and (final_fail is None or final_fail.empty):
                st.success(
                    "FINAL 2025-26 exam complete with zero failures. Export everything and evaluate once; do not tune B3 from this season."
                )
