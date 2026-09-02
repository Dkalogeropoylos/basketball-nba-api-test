from __future__ import annotations

import copy
import json
import os
import numpy as np
import pandas as pd
import streamlit as st

from providers.sportsdataverse_wnba import SportsDataverseWNBA
from providers.router import get_advanced_provider

from core.cleaning import clean_player_log, clean_team_log
from core.buckets import WeightConfig
from core.player_model import PlayerContext, build_player_profile, simulate_player
from core.team_model import (
    TeamContext, build_team_profile, simulate_game,
    team_location_modifiers, h2h_team_audit, h2h_profile_blend,
)
from core.pricing import (
    price, auto_market_table, model_line, most_market, most_market_calibrated, line_ladder,
    required_odds_for_ev,
)
from core.matchup import (
    opponent_allowed_profile,
    player_matchup_modifiers, player_h2h_modifiers,
    team_matchup_modifiers,
    position_environment,
    block_position_susceptibility_modifier,
    fit_opponent_elasticities, fit_player_h2h_prior_minutes,
)
from core.structural_calibration import (
    fit_structural_rate_models, predict_structural_modifiers, coefficient_audit,
)
from core.shooting_efficiency import (
    fit_shooting_efficiency_models, predict_shooting_efficiency_modifiers,
)
from core.minutes_engine import (
    project_team_minutes, rotation_regime_for_team, rotation_similarity_weights,
    residual_rotation_similarity_weights, h2h_rotation_similarity,
)
from core.pace_engine import (
    project_game_pace,
    player_historical_pace_environment,
)
from core.role_splits import current_out_teammates, same_role_game_weights
from core.availability import (
    confirmed_out_players, availability_state_weights, combine_game_weights,
    availability_similarity_weight_maps, confidence_by_stat, combine_stat_weight_maps,
)
from core.availability_impact import (
    recent_team_player_names, augment_current_pool, build_rotation_state_impact,
    defensive_absence_bridge,
)


st.set_page_config(
    page_title="Basketball Pricing Engine",
    page_icon="🏀",
    layout="wide",
)
st.title("🏀 Basketball Pricing Engine v2.17.2 — disjoint H2H + shooting efficiency")
st.caption(
    "v2.17.2: player H2H is fully disjoint from adaptive role-state and uses league-learned empirical-Bayes "
    "prior minutes instead of a universal 5% cap. Team 3P%/2P% use attempt-weighted held-out binomial "
    "offense-vs-defense calibration; v2.17 possession/shot-allocation structure remains unchanged."
)


@st.cache_data(show_spinner=False)
def cached_opponent_elasticities(team_logs: pd.DataFrame):
    return fit_opponent_elasticities(team_logs)


@st.cache_data(show_spinner=False)
def cached_structural_rate_models(team_logs: pd.DataFrame):
    return fit_structural_rate_models(team_logs)


@st.cache_data(show_spinner=False)
def cached_shooting_efficiency_models(team_logs: pd.DataFrame):
    return fit_shooting_efficiency_models(team_logs)


@st.cache_data(show_spinner=False)
def cached_player_h2h_prior_minutes(player_logs: pd.DataFrame):
    return fit_player_h2h_prior_minutes(player_logs)


def _render_player_deep_analysis_impl(board, detail_store, target_ev, reference_odds):
    """Fast UI-only player pricing panel.

    When Streamlit supports fragments, changing player/market reruns only this
    panel. The 50k simulation arrays and model audits are reused from session
    state; no minutes/state/profile/Monte-Carlo work is repeated.
    """
    st.markdown("### 4. Price one selected player")
    pname = st.selectbox(
        "Deep-dive player", board["Player"].tolist(), key="deep_player"
    )
    detail = detail_store[pname]
    sim = detail["sim"]
    markets = ["PTS","REB","AST","3PM","3PA","FTA","PRA","PR","PA","AR"]

    st.markdown("#### Automatic model lines + fair prices")
    cached_table = detail.get("auto_market_table")
    if not isinstance(cached_table, pd.DataFrame):
        cached_table = auto_market_table(
            sim, markets, target_ev=target_ev, reference_odds=reference_odds
        )
    st.dataframe(cached_table.round(3), use_container_width=True, hide_index=True)

    with st.expander("Model audit", expanded=False):
        st.write({
            "opponent": detail["opp_abbr"],
            "projected_minutes": detail["ctx"].projected_minutes,
            "minutes_sd": detail["ctx"].minutes_sd,
            "pace_multiplier": detail["ctx"].pace_multiplier,
        })
        st.markdown("**Non-overlapping sample audit**")
        st.dataframe(detail["profile_audit"], use_container_width=True)
        st.markdown("**Adaptive player role-state audit (walk-forward state-space)**")
        rsa = detail.get("role_state_audit")
        if isinstance(rsa, pd.DataFrame) and not rsa.empty:
            st.dataframe(rsa.round(4), use_container_width=True, hide_index=True)
            st.caption(
                "This layer changes opportunity rates only. q is selected from the player's own history by "
                "one-step-ahead predictive likelihood; Akaike model averaging shrinks back to the static state "
                "when the extra dynamic parameter does not earn predictive support."
            )
        else:
            st.caption("Neutral: insufficient history for adaptive role-state inference.")
        st.markdown("**Opponent / position audit**")
        st.dataframe(detail["matchup_audit"].round(4), use_container_width=True)
        st.markdown("**Confirmed-OUT exact/near-state player-role audit**")
        st.dataframe(detail["same_role_audit"].round(3), use_container_width=True, hide_index=True)
        st.markdown("**Residual role fallback audit**")
        fb = detail.get("availability_fallback_audit")
        if isinstance(fb, pd.DataFrame) and not fb.empty:
            st.dataframe(fb.round(4), use_container_width=True, hide_index=True)
        else:
            st.caption("Neutral fallback: no roster-state role redistribution applied.")

        h2h = detail["plog"][
            detail["plog"]["OPP_ABBR"].astype(str).str.upper()
            == str(detail["opp_abbr"]).upper()
        ]
        st.markdown("**H2H — disjoint empirical-Bayes opportunity layer**")
        h2h_cal = detail.get("h2h_calibration_audit")
        if isinstance(h2h_cal, pd.DataFrame) and not h2h_cal.empty:
            st.caption("League-wide next-H2H predictive calibration; K is prior-equivalent player minutes. K=∞ means the stat did not earn a repeat-matchup layer.")
            st.dataframe(h2h_cal.round(4), use_container_width=True, hide_index=True)
        h2h_audit = detail.get("h2h_audit")
        if isinstance(h2h_audit, pd.DataFrame) and not h2h_audit.empty:
            st.dataframe(h2h_audit.round(4), use_container_width=True, hide_index=True)
        if h2h.empty:
            st.caption("No same-season H2H.")
        else:
            cols = [c for c in [
                "GAME_DATE","OT_FLAG","MIN","PTS","REB","AST","FG3M","FG3A","FTA"
            ] if c in h2h.columns]
            st.dataframe(h2h[cols].sort_values("GAME_DATE", ascending=False), use_container_width=True)

    st.markdown("#### Price ladder — no bookmaker input required")
    ladder_market = st.selectbox("Market ladder", markets, key="player_ladder_market")
    st.caption(
        f"Play-from price = the minimum decimal price that gives at least "
        f"{target_ev:.0%} model EV. Fair price alone is NOT labeled value. "
        f"The {reference_odds:.2f} columns above also show the line needed at a common market price."
    )
    st.dataframe(
        line_ladder(
            sim[ladder_market],
            center_line=model_line(sim[ladder_market])["line"],
            radius=3, target_ev=target_ev,
        ).round(3),
        use_container_width=True, hide_index=True,
    )
    st.info(
        "Use the sportsbook only as a comparison after the model is frozen: "
        "match its offered line to this ladder, then require at least the displayed play-from price."
    )


# Streamlit >=1.37: selectbox changes inside this panel no longer rerun the full app.
_render_player_deep_analysis = (
    st.fragment(_render_player_deep_analysis_impl)
    if hasattr(st, "fragment") else _render_player_deep_analysis_impl
)


def get_secret(name):
    try:
        return st.secrets.get(name, None)
    except Exception:
        return os.getenv(name)


def ci_lookup(d, name, default=None):
    if not isinstance(d, dict):
        return default
    target = str(name).strip().casefold()
    for k, v in d.items():
        if str(k).strip().casefold() == target:
            return v
    return default


def reset_context_widgets():
    prefixes = (
        "deep_min_",
        "usage_",
        "creation_",
        "rebrole_",
        "3role_",
        "ftarole_",
        "sel_override_",
        "home_team_mod_",
        "away_team_mod_",
        "confirmed_out_",
        "global_confirmed_out_",
        "shared_min_names_",
        "shared_min_value_",
    )
    for key in list(st.session_state.keys()):
        if any(str(key).startswith(p) for p in prefixes):
            del st.session_state[key]
    st.session_state.pop("selected_player_board", None)
    st.session_state.pop("team_game_sim", None)
    st.session_state.pop("home_team_weighting_mode", None)
    st.session_state.pop("away_team_weighting_mode", None)


def context_role(manual_context, player_name):
    return ci_lookup(
        manual_context.get("role_adjustments", {}),
        player_name,
        {},
    ) or {}


def _context_out_names_for_team(manual_context, pool, team_abbr):
    return confirmed_out_players(manual_context or {}, pool, team_abbr)


def _apply_confirmed_out_selection(manual_context, pool, team_abbr, selected_names):
    """Update only CONFIRMED OUT state for current-team players.

    GTD/questionable entries may remain in JSON for notes, but Team Markets do
    not probabilistically model them. A player affects availability only after
    the trader explicitly selects OUT.
    """
    ctx = copy.deepcopy(manual_context or {})
    injuries = ctx.setdefault("injuries", {})
    team_names = set(
        pool[pool["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())]
        ["PLAYER_NAME"].astype(str).tolist()
    )
    selected_cf = {str(x).casefold() for x in selected_names}
    # Remove stale OUT flags for this current team when user deselects them.
    for name in list(injuries.keys()):
        if str(name) not in team_names:
            continue
        info = injuries.get(name, {})
        status = str(info.get("status", "") if isinstance(info, dict) else info).upper()
        if status == "OUT" and str(name).casefold() not in selected_cf:
            injuries.pop(name, None)
    for name in selected_names:
        old = injuries.get(str(name), {})
        note = old.get("note") if isinstance(old, dict) else None
        injuries[str(name)] = {"status": "OUT", "team": str(team_abbr).upper()}
        if note:
            injuries[str(name)]["note"] = note
    return ctx


def _context_pool_for_selector(current_pool, player_db, team_abbr):
    """Pool used only for availability UI; includes recent ex-roster names.

    Calculations themselves re-add only players explicitly selected OUT, so an
    old traded player cannot accidentally receive current minutes.
    """
    recent = recent_team_player_names(player_db, team_abbr)
    return augment_current_pool(current_pool, player_db, team_abbr, recent)


def _apply_shared_minute_overrides(manual_context, pool, team_abbr, selected_values):
    ctx = copy.deepcopy(manual_context or {})
    block = dict(ctx.get("projected_minutes", {}) or {})
    team_names = set(
        pool[pool["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())]
        ["PLAYER_NAME"].astype(str).tolist()
    )
    selected_cf = {str(k).casefold(): float(v) for k, v in (selected_values or {}).items()}
    # Deselecting a current-team override removes it from the shared state.
    for name in list(block.keys()):
        if str(name) in team_names and str(name).casefold() not in selected_cf:
            block.pop(name, None)
    for name_cf, value in selected_cf.items():
        actual = next((n for n in team_names if str(n).casefold() == name_cf), None)
        if actual is not None:
            block[str(actual)] = float(value)
    ctx["projected_minutes"] = block
    return ctx


def _clear_team_projected_minutes(manual_context, pool, team_abbr):
    ctx = copy.deepcopy(manual_context or {})
    block = dict(ctx.get("projected_minutes", {}) or {})
    names = set(
        pool[pool["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())]
        ["PLAYER_NAME"].astype(str).tolist()
    )
    ctx["projected_minutes"] = {k: v for k, v in block.items() if str(k) not in names}
    return ctx


def concat_without_attrs(frames, **kwargs):
    """
    pandas propagates/compares DataFrame.attrs during concat. v2.7 stores
    audit DataFrames in attrs, and pandas 2.x/3.x can raise ValueError while
    comparing those nested objects. Strip attrs only on temporary concat
    copies; the original frames keep their audit metadata.
    """
    clean = []
    for frame in frames:
        tmp = frame.copy()
        tmp.attrs = {}
        clean.append(tmp)
    return pd.concat(clean, **kwargs)


def current_game_pace():
    pp = st.session_state.get("pace_projection")
    if not pp:
        return None
    mode = st.session_state.get("pace_mode", "AUTO")
    if mode == "TRADER OVERRIDE":
        return float(st.session_state.get("pace_override", pp.central))
    return float(pp.central)


bdl_key = get_secret("BALLDONTLIE_API_KEY")

with st.sidebar:
    st.header("Data / Security")
    st.write("WNBA historical:", "✅ SportsDataverse")
    st.write("stats.nba.com:", "🚫 not used")
    st.write(
        "BALLDONTLIE advanced:",
        "✅ configured" if bdl_key else "⚪ optional"
    )
    st.divider()
    st.subheader("Pricing discipline")
    target_ev_pct = st.select_slider(
        "Minimum model EV before calling a price playable",
        options=[4,5,6,7,8,10],
        value=6,
        help="Fair odds are break-even. Play-from odds include this extra EV buffer.",
    )
    reference_odds = st.selectbox(
        "Reference price for automatic line thresholds",
        [1.80,1.85,1.90,1.95,2.00],
        index=2,
    )
    target_ev = float(target_ev_pct) / 100.0
    st.caption(
        f"Example: fair 1.70 becomes play-from {1.70*(1+target_ev):.2f} at {target_ev:.0%} target EV."
    )
    st.divider()
    league = st.selectbox(
        "League",
        ["WNBA", "NBA (later)", "EuroLeague (later)", "EuroCup (later)"]
    )
    season = st.number_input(
        "Season", min_value=2002, max_value=2100, value=2026, step=1
    )
    if league != "WNBA":
        st.info("v2.6 is activated for WNBA first.")
        st.stop()


@st.cache_data(ttl=21600, show_spinner=False)
def load_sportsdataverse_season(season):
    return SportsDataverseWNBA(timeout=30).load_season(int(season))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_advanced_player(season, player_id, measure):
    provider, _ = get_advanced_provider("WNBA", bdl_key)
    if provider is None:
        return []
    return provider.player_season_advanced(
        int(season), int(player_id), measure
    )


data_pack = st.session_state.get("sdv_data")
player_db = data_pack["player"] if data_pack else pd.DataFrame()
team_db = data_pack["team"] if data_pack else pd.DataFrame()

tab_game, tab_team, tab_player, tab_audit = st.tabs(
    ["🎯 Game Setup", "🏟️ Team Markets", "👤 Player Props", "🔎 Data Audit"]
)


# =====================================================================
# GAME SETUP
# =====================================================================
with tab_game:
    st.subheader("Game Setup")

    if data_pack is None:
        st.info(
            "Load the season database once. Two static release files are cached "
            "for 6 hours."
        )
        if st.button(f"Load WNBA {int(season)} database", type="primary"):
            try:
                with st.spinner("Loading WNBA historical database..."):
                    pack = load_sportsdataverse_season(int(season))
                st.session_state["sdv_data"] = pack
                for k in [
                    "game_setup", "opp_profile_home", "opp_profile_away",
                    "team_opp_profile_home", "team_opp_profile_away",
                    "pace_projection", "team_sim", "team_game_sim", "player_sim",
                    "selected_player_board",
                ]:
                    st.session_state.pop(k, None)
                st.rerun()
            except Exception as e:
                st.error(f"Database load failed: {e}")
    else:
        provider = SportsDataverseWNBA()
        teams = provider.teams(team_db)

        a,b,c,d = st.columns(4)
        a.metric("Player-game rows", f"{len(player_db):,}")
        b.metric("Team-game rows", f"{len(team_db):,}")
        c.metric("Players", f"{player_db['PLAYER_ID'].nunique():,}")
        d.metric(
            "Through",
            str(pd.Timestamp(player_db["GAME_DATE"].max()).date())
        )

        team_names = teams["TEAM_NAME"].astype(str).tolist()
        c1,c2 = st.columns(2)
        home_name = c1.selectbox("Home team", team_names, key="home_team")
        away_name = c2.selectbox(
            "Away team",
            [x for x in team_names if x != home_name],
            key="away_team",
        )

        lookup = teams.set_index("TEAM_NAME")

        if st.button(
            "Set matchup & build game inputs",
            type="primary",
        ):
            hr, ar = lookup.loc[home_name], lookup.loc[away_name]
            setup = {
                "home_name": home_name,
                "home_id": int(hr["TEAM_ID"]),
                "home_abbr": str(hr["TEAM_ABBR"]),
                "away_name": away_name,
                "away_id": int(ar["TEAM_ID"]),
                "away_abbr": str(ar["TEAM_ABBR"]),
                "season": int(season),
            }
            st.session_state["game_setup"] = setup
            reset_context_widgets()
            # Full opponent profiles remain available for Player Props.
            st.session_state["opp_profile_home"] = opponent_allowed_profile(
                team_db, setup["home_abbr"]
            )
            st.session_state["opp_profile_away"] = opponent_allowed_profile(
                team_db, setup["away_abbr"]
            )
            # Team Markets keep opponent-allowed evidence disjoint from the
            # explicit same-season H2H sample.
            st.session_state["team_opp_profile_home"] = opponent_allowed_profile(
                team_db, setup["home_abbr"], exclude_team_abbr=setup["away_abbr"]
            )
            st.session_state["team_opp_profile_away"] = opponent_allowed_profile(
                team_db, setup["away_abbr"], exclude_team_abbr=setup["home_abbr"]
            )
            st.session_state["pace_projection"] = project_game_pace(
                team_db,
                setup["home_abbr"],
                setup["away_abbr"],
                WeightConfig.stable(),
            )
            st.session_state["pace_mode"] = "AUTO"
            st.session_state.pop("team_sim", None)
            st.session_state.pop("team_game_sim", None)
            st.session_state.pop("selected_player_board", None)
            st.success(
                f"Loaded {setup['away_abbr']} @ {setup['home_abbr']}"
            )

    setup = st.session_state.get("game_setup")
    pace = st.session_state.get("pace_projection")

    if setup and pace:
        st.markdown("### Shared Pace / Possessions Engine")
        st.caption(
            "The same central possessions feed Team Markets and Player Props. "
            "Fast/slow control weights are fitted from completed WNBA games; "
            "they are not a fixed 50/50 midpoint."
        )

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric(f"{setup['home_abbr']} pace", f"{pace.home_pace:.2f}")
        m2.metric(f"{setup['away_abbr']} pace", f"{pace.away_pace:.2f}")
        m3.metric("Auto projection", f"{pace.central:.2f}")
        m4.metric(
            "Fast / Slow weight",
            f"{pace.fast_weight:.2f} / {pace.slow_weight:.2f}"
        )
        m5.metric("Calibration games", f"{pace.calibration_games}")

        p1,p2 = st.columns(2)
        mode = p1.radio(
            "Pace source",
            ["AUTO", "TRADER OVERRIDE"],
            horizontal=True,
            key="pace_mode",
        )
        if mode == "TRADER OVERRIDE":
            p2.number_input(
                "Trader projected possessions",
                min_value=60.0,
                max_value=105.0,
                value=float(round(pace.central, 1)),
                step=0.5,
                key="pace_override",
            )
        else:
            p2.write(
                f"Stress band: **{pace.low:.1f} – {pace.high:.1f}** "
                f"(fit RMSE {pace.rmse:.2f})"
            )

        shared = current_game_pace()
        st.success(f"Shared game possessions used by models: **{shared:.2f}**")

        with st.expander("Market total / handicap — audit only"):
            st.caption(
                "These are NOT fed into pace or projections in v2.6. "
                "Using the sportsbook total to create the same team projections "
                "would be circular. Spread/total remain external cross-checks until "
                "we calibrate a separate market-prior/blowout layer."
            )
            c1,c2 = st.columns(2)
            c1.number_input(
                "Market total (optional)",
                min_value=0.0,
                max_value=250.0,
                value=0.0,
                step=0.5,
                key="market_total_audit",
            )
            c2.number_input(
                "Home handicap (optional)",
                min_value=-40.0,
                max_value=40.0,
                value=0.0,
                step=0.5,
                key="market_spread_audit",
            )

    # ---------------------------------------------------------------
    # SHARED CONFIRMED-OUT STATE — single source of truth for BOTH
    # Team Markets and Player Props. Keeping one selector avoids the two
    # tabs silently disagreeing about availability.
    # ---------------------------------------------------------------
    if setup and data_pack is not None:
        provider = SportsDataverseWNBA()
        pool = provider.current_player_pool(player_db)
        # Availability selector is allowed to show recently departed players so
        # a fresh buyout/trade can still be marked unavailable. Calculations
        # only re-add names that the trader actually selects OUT.
        away_selector_pool = _context_pool_for_selector(pool, player_db, setup["away_abbr"])
        home_selector_pool = _context_pool_for_selector(pool, player_db, setup["home_abbr"])
        selector_pool = concat_without_attrs([away_selector_pool, home_selector_pool], ignore_index=True)
        manual_context = st.session_state.get("game_context", {})

        with st.expander(
            "Confirmed OUT availability — shared by Team Markets + Player Props",
            expanded=True,
        ):
            st.caption(
                "Select only confirmed OUT/unavailable players. Recent ex-roster names are shown too, so a buyout/trade "
                "can move the historical baseline even before the team has played a game without that player. "
                "QUESTIONABLE/GTD stays neutral until you explicitly mark OUT."
            )
            ac1, ac2, ac3 = st.columns([1, 1, 0.7])
            away_opts = sorted(
                away_selector_pool[away_selector_pool["TEAM_ABBR"].astype(str).str.upper().eq(setup["away_abbr"].upper())]
                ["PLAYER_NAME"].astype(str).unique().tolist(), key=str.casefold,
            )
            home_opts = sorted(
                home_selector_pool[home_selector_pool["TEAM_ABBR"].astype(str).str.upper().eq(setup["home_abbr"].upper())]
                ["PLAYER_NAME"].astype(str).unique().tolist(), key=str.casefold,
            )
            away_default = [
                x for x in _context_out_names_for_team(manual_context, away_selector_pool, setup["away_abbr"])
                if x in away_opts
            ]
            home_default = [
                x for x in _context_out_names_for_team(manual_context, home_selector_pool, setup["home_abbr"])
                if x in home_opts
            ]
            away_selected_out = ac1.multiselect(
                f"{setup['away_abbr']} confirmed OUT", away_opts, default=away_default,
                key=f"global_confirmed_out_{setup['away_abbr']}",
            )
            home_selected_out = ac2.multiselect(
                f"{setup['home_abbr']} confirmed OUT", home_opts, default=home_default,
                key=f"global_confirmed_out_{setup['home_abbr']}",
            )
            ac3.number_input(
                "State shrink K", min_value=3.0, max_value=15.0, value=6.0, step=1.0,
                help=(
                    "v2.15 near-state evidence is partial-pooled with K. Five fully comparable games is the maturity point; "
                    "1–4 games still count with strong shrinkage. Synthetic rotation fallback fills only residual uncertainty."
                ), key="availability_state_k",
            )
            manual_context = _apply_confirmed_out_selection(
                manual_context, away_selector_pool, setup["away_abbr"], away_selected_out
            )
            manual_context = _apply_confirmed_out_selection(
                manual_context, home_selector_pool, setup["home_abbr"], home_selected_out
            )
            st.session_state["game_context"] = manual_context
            st.caption(
                "Overlap guard: each historical game gets one stat-specific availability similarity score only. "
                "Near-state relevance is an INNER weight inside Old/G6–10/L5; synthetic fallback fills only the remaining gap."
            )

        # Shared minute restrictions/returns are current-state information and
        # must feed BOTH team markets and player props. Local player-tab overrides
        # remain scenario-only; use this block for a real restriction such as Plum.
        with st.expander("Shared minute overrides / restrictions", expanded=False):
            st.caption(
                "Use this for a real current restriction/return. These minutes are written to game_context and affect "
                "the full 200-minute rotation, Team Markets and Player Props."
            )
            # Calculation pools include only selected OUT historical names; no
            # unrelated old player is allowed back into today's rotation.
            away_calc_pool = augment_current_pool(pool, player_db, setup["away_abbr"], away_selected_out)
            home_calc_pool = augment_current_pool(pool, player_db, setup["home_abbr"], home_selected_out)
            calc_pool = concat_without_attrs([away_calc_pool, home_calc_pool], ignore_index=True)

            # AUTO baselines ignore existing shared minute overrides, but retain OUTs.
            away_auto_ctx = _clear_team_projected_minutes(manual_context, away_calc_pool, setup["away_abbr"])
            home_auto_ctx = _clear_team_projected_minutes(manual_context, home_calc_pool, setup["home_abbr"])
            away_auto_min = project_team_minutes(
                player_db, team_db, away_calc_pool, setup["away_abbr"], setup["away_name"], away_auto_ctx
            )
            home_auto_min = project_team_minutes(
                player_db, team_db, home_calc_pool, setup["home_abbr"], setup["home_name"], home_auto_ctx
            )
            auto_min_map = {
                str(r["Player"]): float(r["Projected Min"])
                for _, r in concat_without_attrs([away_auto_min, home_auto_min], ignore_index=True).iterrows()
            }
            existing_pm = dict(manual_context.get("projected_minutes", {}) or {})
            c1, c2 = st.columns(2)
            away_active = sorted(
                [x for x in pool[pool["TEAM_ABBR"].astype(str).str.upper().eq(setup["away_abbr"].upper())]["PLAYER_NAME"].astype(str).unique().tolist()
                 if x not in set(away_selected_out)], key=str.casefold
            )
            home_active = sorted(
                [x for x in pool[pool["TEAM_ABBR"].astype(str).str.upper().eq(setup["home_abbr"].upper())]["PLAYER_NAME"].astype(str).unique().tolist()
                 if x not in set(home_selected_out)], key=str.casefold
            )
            away_existing = [x for x in away_active if x in existing_pm]
            home_existing = [x for x in home_active if x in existing_pm]
            away_min_names = c1.multiselect(
                f"{setup['away_abbr']} minute overrides", away_active, default=away_existing,
                key=f"shared_min_names_{setup['away_abbr']}",
            )
            home_min_names = c2.multiselect(
                f"{setup['home_abbr']} minute overrides", home_active, default=home_existing,
                key=f"shared_min_names_{setup['home_abbr']}",
            )
            away_values, home_values = {}, {}
            if away_min_names:
                st.markdown(f"**{setup['away_abbr']}**")
                cols = st.columns(min(4, len(away_min_names)))
                for i, name in enumerate(away_min_names):
                    default = float(existing_pm.get(name, auto_min_map.get(name, 20.0)))
                    away_values[name] = cols[i % len(cols)].number_input(
                        f"{name} expected MIN", min_value=0.0, max_value=40.0,
                        value=float(np.clip(default, 0.0, 40.0)), step=0.5,
                        key=f"shared_min_value_{setup['away_abbr']}_{name}",
                        help=f"AUTO baseline ≈ {auto_min_map.get(name, np.nan):.1f}",
                    )
            if home_min_names:
                st.markdown(f"**{setup['home_abbr']}**")
                cols = st.columns(min(4, len(home_min_names)))
                for i, name in enumerate(home_min_names):
                    default = float(existing_pm.get(name, auto_min_map.get(name, 20.0)))
                    home_values[name] = cols[i % len(cols)].number_input(
                        f"{name} expected MIN", min_value=0.0, max_value=40.0,
                        value=float(np.clip(default, 0.0, 40.0)), step=0.5,
                        key=f"shared_min_value_{setup['home_abbr']}_{name}",
                        help=f"AUTO baseline ≈ {auto_min_map.get(name, np.nan):.1f}",
                    )
            manual_context = _apply_shared_minute_overrides(
                manual_context, away_calc_pool, setup["away_abbr"], away_values
            )
            manual_context = _apply_shared_minute_overrides(
                manual_context, home_calc_pool, setup["home_abbr"], home_values
            )
            st.session_state["game_context"] = manual_context
            if away_values or home_values:
                st.success(
                    "Shared minute context active: "
                    + "; ".join([f"{k} {v:.1f}'" for k, v in {**away_values, **home_values}.items()])
                )

    st.markdown("### Trader context")
    st.caption(
        "The model does not guess injuries. Trader marks confirmed OUT and can "
        "override minutes/roles. If minutes are omitted, the Minutes Engine "
        "projects them automatically."
    )

    upload = st.file_uploader(
        "Upload game_context.json",
        type=["json"],
        key="context_uploader",
    )
    if upload is not None:
        try:
            raw = upload.getvalue()
            sig = (upload.name, len(raw), hash(raw))
            if st.session_state.get("_context_sig") != sig:
                parsed = json.loads(raw.decode("utf-8"))
                st.session_state["game_context"] = parsed
                st.session_state["context_editor"] = json.dumps(
                    parsed, indent=2, ensure_ascii=False
                )
                st.session_state["_context_sig"] = sig
                reset_context_widgets()
                st.rerun()
        except Exception as e:
            st.error(f"Invalid context JSON: {e}")

    if "context_editor" not in st.session_state:
        st.session_state["context_editor"] = json.dumps(
            st.session_state.get("game_context", {}),
            indent=2,
            ensure_ascii=False,
        ) if st.session_state.get("game_context") else ""

    st.text_area(
        "Paste / edit trader context",
        height=230,
        key="context_editor",
        placeholder=(
            '{"injuries":{"Player X":{"status":"OUT"}},'
            '"projected_minutes":{"Player Y":31},'
            '"rotation_regime":{"LAS":"role_change"}}'
        ),
    )

    if st.button("Apply trader context"):
        try:
            txt = st.session_state.get("context_editor", "")
            st.session_state["game_context"] = (
                json.loads(txt) if txt.strip() else {}
            )
            reset_context_widgets()
            st.success("Trader context applied.")
            st.rerun()
        except Exception as e:
            st.error(f"JSON error: {e}")


# =====================================================================
# TEAM MARKETS
# =====================================================================
with tab_team:
    st.subheader("Team Markets — coupled game engine")

    setup = st.session_state.get("game_setup")
    shared_pace = current_game_pace()

    if data_pack is None:
        st.info("Load the database first.")
    elif not setup or shared_pace is None:
        st.info("Set the matchup in Game Setup first.")
    else:
        manual_context = st.session_state.get("game_context", {})
        provider = SportsDataverseWNBA()
        pool = provider.current_player_pool(player_db)

        home_log = clean_team_log(
            team_db[team_db["TEAM_ID"] == setup["home_id"]].copy()
        )
        away_log = clean_team_log(
            team_db[team_db["TEAM_ID"] == setup["away_id"]].copy()
        )

        st.markdown(
            f"### {setup['away_abbr']} @ {setup['home_abbr']}"
        )
        st.info(
            f"Shared projected possessions: **{shared_pace:.2f}**. "
            "Both teams are simulated in the SAME game state, so totals, "
            "rebounds, steals, blocks and 'team with most' markets are coherent."
        )

        # -----------------------------------------------------------------
        # CONFIRMED OUT state comes from the single shared selector in Game Setup.
        # Team Markets only READ it here, so Team and Player tabs cannot disagree.
        # -----------------------------------------------------------------
        away_selected_out = _context_out_names_for_team(
            manual_context, pool, setup["away_abbr"]
        )
        home_selected_out = _context_out_names_for_team(
            manual_context, pool, setup["home_abbr"]
        )
        availability_k = float(st.session_state.get("availability_state_k", 6.0))
        st.caption(
            "Shared confirmed OUT state — "
            f"{setup['away_abbr']}: {', '.join(away_selected_out) if away_selected_out else '—'} | "
            f"{setup['home_abbr']}: {', '.join(home_selected_out) if home_selected_out else '—'} | "
            f"shrink K={availability_k:.1f}. Change this only in Game Setup."
        )

        # AUTO regime is separate from confirmed OUT state. Use role_change only
        # for a broader structural role/rotation change, not merely because the
        # same OUT players were selected above.
        home_regime_auto = rotation_regime_for_team(
            manual_context, setup["home_name"], setup["home_abbr"]
        )
        away_regime_auto = rotation_regime_for_team(
            manual_context, setup["away_name"], setup["away_abbr"]
        )

        with st.expander("Team sample weighting / trader override", expanded=False):
            st.caption(
                "AUTO reads rotation_regime from game_context.json. Stable = "
                "55/20/25; role_change = 35/20/45. Confirmed OUT state is a separate INNER-bucket relevance layer. "
                "Use role_change only for residual structural change so the same absence is not counted twice."
            )
            c1, c2 = st.columns(2)
            home_mode = c1.selectbox(
                f"{setup['home_abbr']} weighting",
                ["AUTO", "Stable 55/20/25", "Role change 35/20/45"],
                key="home_team_weighting_mode",
            )
            away_mode = c2.selectbox(
                f"{setup['away_abbr']} weighting",
                ["AUTO", "Stable 55/20/25", "Role change 35/20/45"],
                key="away_team_weighting_mode",
            )
            rotation_similarity_enabled = st.toggle(
                "AUTO: residual rotation similarity after OUT-state matching",
                value=True,
                help=(
                    "Confirmed OUT players are removed from this Jaccard comparison because their effect is already "
                    "handled by exact availability-state weighting. Only the residual rotation receives a weak 0.85–1.00 inner weight."
                ),
                key="team_rotation_similarity_toggle",
            )

        def resolve_cfg(mode, auto_regime, has_confirmed_out):
            if mode.startswith("Stable"):
                return WeightConfig.stable(), "stable"
            if mode.startswith("Role"):
                return WeightConfig.role_change(), "role_change"
            # Overlap guard: when confirmed OUT identity is already modeled by
            # exact-state historical matching, AUTO does not ALSO switch to the
            # 35/20/45 injury-style recency profile. If there is a broader role
            # change (trade/coaching shift), trader can explicitly choose Role change.
            if has_confirmed_out:
                return WeightConfig.stable(), "stable (OUT-state guard)"
            return (
                (WeightConfig.role_change(), "role_change")
                if auto_regime == "role_change"
                else (WeightConfig.stable(), "stable")
            )

        home_cfg, home_regime = resolve_cfg(home_mode, home_regime_auto, bool(home_selected_out))
        away_cfg, away_regime = resolve_cfg(away_mode, away_regime_auto, bool(away_selected_out))
        if (home_selected_out and home_regime_auto == "role_change" and home_mode == "AUTO") or \
           (away_selected_out and away_regime_auto == "role_change" and away_mode == "AUTO"):
            st.caption(
                "OUT-state overlap guard active: AUTO keeps Stable 55/20/25 when confirmed OUT is selected. "
                "Choose Role change explicitly only if there is an additional structural rotation change beyond those OUTs."
            )

        # -----------------------------------------------------------------
        # INNER historical relevance: exact OUT state + weak residual rotation.
        # These two layers are deliberately separated to avoid injury overlap.
        # -----------------------------------------------------------------
        home_out = confirmed_out_players(manual_context, pool, setup["home_abbr"])
        away_out = confirmed_out_players(manual_context, pool, setup["away_abbr"])

        team_state_stats = [
            "FGA", "3PA", "FTA", "TOV", "OREB", "DREB",
            "AST", "STL", "BLK", "PF",
        ]
        home_avail_maps, home_avail_audit, home_state_scores = availability_similarity_weight_maps(
            player_db, home_log, setup["home_abbr"], home_out, team_state_stats,
            current_pool=pool, focal_player=None, k=float(availability_k),
            maturity_games=5.0, exclude_opponent_abbr=setup["away_abbr"],
        )
        away_avail_maps, away_avail_audit, away_state_scores = availability_similarity_weight_maps(
            player_db, away_log, setup["away_abbr"], away_out, team_state_stats,
            current_pool=pool, focal_player=None, k=float(availability_k),
            maturity_games=5.0, exclude_opponent_abbr=setup["home_abbr"],
        )

        home_rot_w = (
            residual_rotation_similarity_weights(
                player_db, pool, setup["home_abbr"], manual_context,
                out_players=home_out, residual_strength=0.15,
            ) if rotation_similarity_enabled else {}
        )
        away_rot_w = (
            residual_rotation_similarity_weights(
                player_db, pool, setup["away_abbr"], manual_context,
                out_players=away_out, residual_strength=0.15,
            ) if rotation_similarity_enabled else {}
        )
        home_game_weights_by_stat = combine_stat_weight_maps(home_avail_maps, home_rot_w)
        away_game_weights_by_stat = combine_stat_weight_maps(away_avail_maps, away_rot_w)
        # Compatibility/audit view only. Team profile itself uses the stat-specific maps above.
        home_game_weights = home_game_weights_by_stat.get("FGA", home_rot_w)
        away_game_weights = away_game_weights_by_stat.get("FGA", away_rot_w)

        # Baseline excludes current-opponent H2H INSIDE the actual Old/G6-10/L5
        # buckets. Near-state similarity is also INNER-only, once per historical game.
        home_profile, home_audit = build_team_profile(
            home_log, home_cfg,
            league_team_logs=team_db,
            game_weights_by_stat=home_game_weights_by_stat,
            exclude_opponent_abbr=setup["away_abbr"],
        )
        away_profile, away_audit = build_team_profile(
            away_log, away_cfg,
            league_team_logs=team_db,
            game_weights_by_stat=away_game_weights_by_stat,
            exclude_opponent_abbr=setup["home_abbr"],
        )

        h2h_all = h2h_team_audit(team_db, setup["home_abbr"], setup["away_abbr"])
        home_h2h_ids = (
            h2h_all[h2h_all["TEAM_ABBR"].astype(str).str.upper().eq(setup["home_abbr"].upper())]["GAME_ID"].astype(str).tolist()
            if not h2h_all.empty else []
        )
        away_h2h_ids = (
            h2h_all[h2h_all["TEAM_ABBR"].astype(str).str.upper().eq(setup["away_abbr"].upper())]["GAME_ID"].astype(str).tolist()
            if not h2h_all.empty else []
        )
        home_h2h_sim = h2h_rotation_similarity(
            player_db, pool, setup["home_abbr"], manual_context, home_h2h_ids
        )
        away_h2h_sim = h2h_rotation_similarity(
            player_db, pool, setup["away_abbr"], manual_context, away_h2h_ids
        )
        # v2.17: 3P share / FTA / TOV / AST H2H are learned jointly from
        # historical repeat-matchup residuals, so the old fixed H2H blend is
        # skipped for those four rates to avoid double counting.
        _structural_h2h_skip = {"three_share", "fta_pp", "tov_pp", "assist_per_make"}
        home_profile, home_h2h_audit = h2h_profile_blend(
            team_db, setup["home_abbr"], setup["away_abbr"], home_profile,
            rotation_similarity=home_h2h_sim, skip_features=_structural_h2h_skip,
        )
        away_profile, away_h2h_audit = h2h_profile_blend(
            team_db, setup["away_abbr"], setup["home_abbr"], away_profile,
            rotation_similarity=away_h2h_sim, skip_features=_structural_h2h_skip,
        )

        # v2.12 roster-state bridge. Exact historical OUT matching remains the
        # primary empirical layer. When exact games are sparse/zero (e.g. a
        # player just left the roster), a synthetic 200-minute counterfactual
        # moves team style instead of leaving the line unchanged. Explicit
        # minute restrictions are a separate current-state layer.
        def _impact_confidence(audit):
            c = confidence_by_stat(audit)
            return {
                "FGA": c.get("FGA", 0.0),
                "3P_SHARE": c.get("3PA", 0.0),
                "3P_PCT": c.get("3PA", 0.0),
                "2P_PCT": c.get("FGA", 0.0),
                "FTA": c.get("FTA", 0.0), "TOV": c.get("TOV", 0.0),
                "OREB": c.get("OREB", 0.0), "DREB": c.get("DREB", 0.0),
                "AST": c.get("AST", 0.0), "STL": c.get("STL", 0.0),
                "BLK": c.get("BLK", 0.0), "PF": c.get("PF", 0.0),
            }

        home_rot_impact = build_rotation_state_impact(
            player_db, team_db, pool, setup["home_abbr"], setup["home_name"],
            manual_context, home_out, state_confidence_by_stat=_impact_confidence(home_avail_audit),
        )
        away_rot_impact = build_rotation_state_impact(
            player_db, team_db, pool, setup["away_abbr"], setup["away_name"],
            manual_context, away_out, state_confidence_by_stat=_impact_confidence(away_avail_audit),
        )
        home_roster_mod = home_rot_impact.modifiers
        away_roster_mod = away_rot_impact.modifiers

        # v2.16 current DEFENSIVE roster bridge.  Each current OUT is evaluated
        # separately from her first team appearance onward.  >=10 MIN is ON,
        # 0 MIN is OFF, and 0<MIN<10 is excluded from both groups.  The combined
        # effect is strongly shrunk/overlap-protected and modifies the OPPONENT
        # offense; it is not another historical outer sample.
        home_def_out_mod, home_def_out_audit = defensive_absence_bridge(
            player_db, team_db, pool, setup["home_abbr"], home_out,
            exclude_opponent_abbr=setup["away_abbr"], on_min_minutes=10.0,
        )
        away_def_out_mod, away_def_out_audit = defensive_absence_bridge(
            player_db, team_db, pool, setup["away_abbr"], away_out,
            exclude_opponent_abbr=setup["home_abbr"], on_min_minutes=10.0,
        )

        # Structural matchup blending should see the CURRENT offensive identity.
        # The final context still multiplies the roster modifier because the
        # simulator's base profile is historical; this is not double counting.
        def _roster_adjusted_profile(profile, roster_mod):
            q = dict(profile)
            q["three_share"] = float(np.clip(
                q.get("three_share", 0.35) * roster_mod.get("3P_SHARE", 1.0), 0.06, 0.75
            ))
            q["fta_pp"] = float(np.clip(
                q.get("fta_pp", 0.24) * roster_mod.get("FTA", 1.0), 0.05, 0.55
            ))
            return q

        home_match_profile = _roster_adjusted_profile(home_profile, home_roster_mod)
        away_match_profile = _roster_adjusted_profile(away_profile, away_roster_mod)

        # Opponent allowance for Team Markets excludes the same H2H rows, so the
        # explicit H2H layer above is genuinely non-overlapping.
        home_opp = st.session_state.get("team_opp_profile_away") or opponent_allowed_profile(
            team_db, setup["away_abbr"], exclude_team_abbr=setup["home_abbr"]
        )
        away_opp = st.session_state.get("team_opp_profile_home") or opponent_allowed_profile(
            team_db, setup["home_abbr"], exclude_team_abbr=setup["away_abbr"]
        )
        opponent_elasticities, opponent_elasticity_audit = cached_opponent_elasticities(team_db)
        home_auto = team_matchup_modifiers(
            home_opp, own_profile=home_match_profile, elasticities=opponent_elasticities
        )
        away_auto = team_matchup_modifiers(
            away_opp, own_profile=away_match_profile, elasticities=opponent_elasticities
        )

        # v2.17 structural layer.  These four modifiers replace the old fixed
        # recent/opponent/H2H response for the main possession-allocation rates.
        # Coefficients and ridge regularization are chosen from walk-forward WNBA
        # history; the learned model activates only if it beats the existing
        # Old/G6-10/L5 baseline out of sample.
        structural_models, structural_model_audit = cached_structural_rate_models(team_db)
        home_struct, home_struct_audit = predict_structural_modifiers(
            team_db, setup["home_abbr"], setup["away_abbr"], structural_models,
            home_cfg, h2h_rotation_similarity=home_h2h_sim,
        )
        away_struct, away_struct_audit = predict_structural_modifiers(
            team_db, setup["away_abbr"], setup["home_abbr"], structural_models,
            away_cfg, h2h_rotation_similarity=away_h2h_sim,
        )
        for _k in ("3P_SHARE", "FTA", "TOV", "AST"):
            home_auto[_k] = float(home_struct.get(_k, home_auto.get(_k, 1.0)))
            away_auto[_k] = float(away_struct.get(_k, away_auto.get(_k, 1.0)))

        # v2.17.2 team shooting efficiency.  Attempts/shot mix stay exactly in
        # the v2.17 structural chain; only conditional make probabilities move.
        shooting_models, shooting_model_audit = cached_shooting_efficiency_models(team_db)
        home_shoot, home_shoot_audit = predict_shooting_efficiency_modifiers(
            team_db, setup["home_abbr"], setup["away_abbr"], home_match_profile, shooting_models
        )
        away_shoot, away_shoot_audit = predict_shooting_efficiency_modifiers(
            team_db, setup["away_abbr"], setup["home_abbr"], away_match_profile, shooting_models
        )
        for _k in ("3P_PCT", "2P_PCT"):
            home_auto[_k] = float(home_shoot.get(_k, home_auto.get(_k, 1.0)))
            away_auto[_k] = float(away_shoot.get(_k, away_auto.get(_k, 1.0)))

        # Small location correction with the current H2H opponent excluded as well.
        home_loc, home_loc_audit = team_location_modifiers(
            home_log, True, league_team_logs=team_db,
            exclude_opponent_abbr=setup["away_abbr"],
        )
        away_loc, away_loc_audit = team_location_modifiers(
            away_log, False, league_team_logs=team_db,
            exclude_opponent_abbr=setup["home_abbr"],
        )

        def combined_mods(auto, loc, roster, opp_def_out):
            # Conditional layers are combined once each:
            # historical offense -> own roster state -> opponent baseline matchup
            # -> CURRENT opponent defensive roster -> small location.
            return {
                "FGA": float(np.clip(roster.get("FGA", 1.0) * auto.get("FGA", 1.0) * loc.get("FGA", 1.0), 0.90, 1.10)),
                # v2.17: do not re-shrink the learned structural response with
                # another hand-picked matchup cap.  Physical probability/rate
                # bounds live inside the simulator; roster/defensive-OUT/location
                # layers remain explicit and auditable here.
                "3P_SHARE": float(roster.get("3P_SHARE", 1.0) * auto.get("3P_SHARE", 1.0) * opp_def_out.get("3P_SHARE",1.0) * loc.get("3P_SHARE", 1.0)),
                "FTA": float(roster.get("FTA", 1.0) * auto.get("FTA", 1.0) * opp_def_out.get("FTA",1.0) * loc.get("FTA", 1.0)),
                "TOV": float(roster.get("TOV", 1.0) * auto.get("TOV", 1.0) * opp_def_out.get("TOV",1.0) * loc.get("TOV", 1.0)),
                "OREB": float(np.clip(roster.get("OREB", 1.0) * auto.get("OREB", 1.0) * opp_def_out.get("OREB",1.0) * loc.get("OREB", 1.0), 0.82, 1.18)),
                "AST": float(roster.get("AST", 1.0) * auto.get("AST", 1.0) * opp_def_out.get("AST",1.0) * loc.get("AST", 1.0)),
                "PF": float(np.clip(roster.get("PF", 1.0) * auto.get("PF", 1.0) * loc.get("PF", 1.0), 0.84, 1.16)),
                "DREB": float(np.clip(roster.get("DREB", 1.0) * auto.get("DREB", 1.0) * loc.get("DREB", 1.0), 0.88, 1.12)),
                "STL": float(np.clip(roster.get("STL", 1.0) * loc.get("STL", 1.0), 0.86, 1.14)),
                "BLK": float(np.clip(roster.get("BLK", 1.0) * auto.get("BLK", 1.0) * loc.get("BLK", 1.0), 0.82, 1.18)),
                "3P_PCT": float(np.clip(roster.get("3P_PCT", 1.0) * auto.get("3P_PCT", 1.0) * opp_def_out.get("3P_PCT",1.0) * loc.get("3P_PCT", 1.0), 0.92, 1.08)),
                "2P_PCT": float(np.clip(roster.get("2P_PCT", 1.0) * auto.get("2P_PCT", 1.0) * opp_def_out.get("2P_PCT",1.0) * loc.get("2P_PCT", 1.0), 0.92, 1.08)),
            }


        home_mod = combined_mods(home_auto, home_loc, home_roster_mod, away_def_out_mod)
        away_mod = combined_mods(away_auto, away_loc, away_roster_mod, home_def_out_mod)

        # v2.15.1: Streamlit widgets keep their session_state value after first
        # creation, so changing confirmed OUT / shared minutes / shrink K could
        # leave the visible trader modifier inputs (and therefore the simulator)
        # stuck on the PREVIOUS context. Re-seed AUTO values only when the
        # underlying model context changes. Manual edits remain sticky while the
        # context itself is unchanged.
        _team_auto_mod_sig = (
            setup["home_abbr"], setup["away_abbr"],
            tuple(sorted(str(x) for x in home_out)),
            tuple(sorted(str(x) for x in away_out)),
            round(float(availability_k), 4),
            tuple(sorted((str(k), round(float(v), 4)) for k, v in (manual_context.get("projected_minutes", {}) or {}).items())),
            tuple((k, round(float(home_roster_mod.get(k, 1.0)), 5)) for k in sorted(home_roster_mod)),
            tuple((k, round(float(away_roster_mod.get(k, 1.0)), 5)) for k in sorted(away_roster_mod)),
            tuple((k, round(float(home_def_out_mod.get(k, 1.0)), 5)) for k in sorted(home_def_out_mod)),
            tuple((k, round(float(away_def_out_mod.get(k, 1.0)), 5)) for k in sorted(away_def_out_mod)),
            tuple((k, round(float(home_mod.get(k, 1.0)), 5)) for k in sorted(home_mod)),
            tuple((k, round(float(away_mod.get(k, 1.0)), 5)) for k in sorted(away_mod)),
        )
        if st.session_state.get("_team_auto_mod_sig") != _team_auto_mod_sig:
            for _k, _v in home_mod.items():
                st.session_state[f"home_team_mod_{_k}"] = float(_v)
            for _k, _v in away_mod.items():
                st.session_state[f"away_team_mod_{_k}"] = float(_v)
            st.session_state["_team_auto_mod_sig"] = _team_auto_mod_sig
            # Any previous simulation belongs to the old context.
            st.session_state.pop("team_game_sim", None)
            st.session_state.pop("selected_player_board", None)

        # Optional positional BLK susceptibility. It is neutral unless the
        # loaded player data contains an actual blocked-attempt field.
        home_blk_pos, home_blk_pos_audit = block_position_susceptibility_modifier(
            player_db, setup["away_abbr"], current_pool=pool, out_players=away_out
        )
        away_blk_pos, away_blk_pos_audit = block_position_susceptibility_modifier(
            player_db, setup["home_abbr"], current_pool=pool, out_players=home_out
        )

        with st.expander("Automatic matchup/location modifiers — optional trader override", expanded=False):
            st.caption(
                "Leave these untouched for full AUTO. v2.17 learns 3P_SHARE, FTA, TOV and AST-per-made-FG from "
                "walk-forward WNBA history using disjoint Old/G6-10/L5 + opponent + H2H inputs. FGA remains a "
                "possession-identity consequence. WNBA/NBA official AST are tied to made field goals; free-throw "
                "assist credit is a FIBA/EuroLeague statistical rule and is not mixed into WNBA markets."
            )

            with st.expander("v2.17 structural-rate calibration audit", expanded=False):
                st.caption(
                    "No fixed opponent or H2H weight is used for 3P_SHARE / FTA / TOV / AST. Ridge strength is "
                    "chosen by expanding-window validation, and a structural model activates only when it improves "
                    "out-of-sample RMSE versus the existing Old/G6-10/L5 baseline."
                )
                if isinstance(structural_model_audit, pd.DataFrame) and not structural_model_audit.empty:
                    st.dataframe(structural_model_audit.round(4), use_container_width=True, hide_index=True)
                _coef = coefficient_audit(structural_models)
                if isinstance(_coef, pd.DataFrame) and not _coef.empty:
                    st.dataframe(_coef.round(4), use_container_width=True, hide_index=True)
                st.markdown(f"**{setup['away_abbr']} current structural prediction**")
                st.dataframe(away_struct_audit.round(4), use_container_width=True, hide_index=True)
                st.markdown(f"**{setup['home_abbr']} current structural prediction**")
                st.dataframe(home_struct_audit.round(4), use_container_width=True, hide_index=True)

            with st.expander("v2.17.2 shooting-efficiency calibration audit", expanded=False):
                st.caption(
                    "3P% and 2P% are fit as Binomial makes|attempts. Own-offense and opponent-defense shrinkage masses "
                    "plus the defensive response beta are selected from chronological data; the layer activates only if it "
                    "beats an own-offense-only model on later held-out games. This does not change FGA or shot share."
                )
                if isinstance(shooting_model_audit, pd.DataFrame) and not shooting_model_audit.empty:
                    st.dataframe(shooting_model_audit.round(4), use_container_width=True, hide_index=True)
                st.markdown(f"**{setup['away_abbr']} current shooting prediction**")
                st.dataframe(away_shoot_audit.round(4), use_container_width=True, hide_index=True)
                st.markdown(f"**{setup['home_abbr']} current shooting prediction**")
                st.dataframe(home_shoot_audit.round(4), use_container_width=True, hide_index=True)

            with st.expander("Opponent-elasticity calibration audit", expanded=False):
                st.caption(
                    "Legacy audit for the remaining generic opponent layers. In v2.17 its 3P_SHARE / FTA / TOV / AST "
                    "rows are NOT used in the simulator; those four are replaced by the structural-rate model above. "
                    "3P%/2P% rows are now audit-only as well; v2.17.2 replaces them with the held-out Binomial shooting layer."
                )
                if isinstance(opponent_elasticity_audit, pd.DataFrame) and not opponent_elasticity_audit.empty:
                    st.dataframe(opponent_elasticity_audit.round(4), use_container_width=True, hide_index=True)

            def modifier_editor(prefix, team_abbr, mods):
                cols = st.columns(5)
                out = {}
                keys = ["FGA","3P_SHARE","FTA","TOV","OREB","AST","PF","DREB","STL","BLK","3P_PCT","2P_PCT"]
                for i, key in enumerate(keys):
                    out[key] = cols[i % 5].number_input(
                        f"{team_abbr} {key}",
                        min_value=0.70, max_value=1.30,
                        value=float(mods[key]), step=0.01,
                        key=f"{prefix}_{key}",
                    )
                return out

            st.markdown(f"**{setup['away_abbr']}**")
            away_mod = modifier_editor("away_team_mod", setup["away_abbr"], away_mod)
            st.markdown(f"**{setup['home_abbr']}**")
            home_mod = modifier_editor("home_team_mod", setup["home_abbr"], home_mod)

        pace_obj = st.session_state["pace_projection"]
        c1, c2 = st.columns(2)
        poss_sd = c1.number_input(
            "Possession SD",
            min_value=1.0, max_value=8.0,
            value=float(pace_obj.sd), step=0.25,
            key="game_team_poss_sd",
        )
        n = c2.select_slider(
            "Game simulations",
            [25_000, 50_000, 100_000, 250_000, 500_000],
            50_000,
            key="game_team_sims",
        )

        def make_ctx(mod, blk_position=1.0):
            return TeamContext(
                projected_possessions=float(shared_pace),
                possessions_sd=float(poss_sd),
                fga=float(mod["FGA"]),
                three_share=float(mod["3P_SHARE"]),
                three_pct=float(mod.get("3P_PCT", 1.0)),
                two_pct=float(mod.get("2P_PCT", 1.0)),
                fta=float(mod["FTA"]),
                tov=float(mod["TOV"]),
                oreb=float(mod["OREB"]),
                ast=float(mod["AST"]),
                pf=float(mod["PF"]),
                dreb=float(mod["DREB"]),
                stl=float(mod["STL"]),
                blk=float(mod["BLK"]),
                blk_position=float(blk_position),
                blk_h2h=1.0,
            )

        home_ctx = make_ctx(home_mod, home_blk_pos)
        away_ctx = make_ctx(away_mod, away_blk_pos)

        fingerprint = (
            setup["home_abbr"], setup["away_abbr"], float(shared_pace),
            float(poss_sd), home_regime, away_regime, bool(rotation_similarity_enabled),
            tuple(home_out), tuple(away_out), float(availability_k),
            tuple(sorted((str(k), float(v)) for k, v in (manual_context.get("projected_minutes", {}) or {}).items())),
            tuple(round(home_roster_mod[k], 4) for k in sorted(home_roster_mod)),
            tuple(round(away_roster_mod[k], 4) for k in sorted(away_roster_mod)),
            tuple(round(home_def_out_mod[k], 4) for k in sorted(home_def_out_mod)),
            tuple(round(away_def_out_mod[k], 4) for k in sorted(away_def_out_mod)),
            round(home_h2h_sim, 4), round(away_h2h_sim, 4),
            round(home_blk_pos, 4), round(away_blk_pos, 4),
            tuple(round(home_mod[k], 4) for k in sorted(home_mod)),
            tuple(round(away_mod[k], 4) for k in sorted(away_mod)),
            int(n),
        )

        if st.button("Run full-game Monte Carlo", type="primary", key="run_full_team_game"):
            home_sim, away_sim = simulate_game(
                home_profile, away_profile, home_ctx, away_ctx,
                int(n), seed=801,
            )
            sn = min(80_000, max(25_000, int(n)//4))
            low_mult = float(pace_obj.low / shared_pace) if shared_pace else 0.96
            high_mult = float(pace_obj.high / shared_pace) if shared_pace else 1.04
            home_low, away_low = simulate_game(
                home_profile, away_profile, home_ctx, away_ctx,
                sn, seed=802, opportunity_mult=low_mult,
            )
            home_high, away_high = simulate_game(
                home_profile, away_profile, home_ctx, away_ctx,
                sn, seed=803, opportunity_mult=high_mult,
            )
            st.session_state["team_game_sim"] = {
                "fingerprint": fingerprint,
                "home": home_sim, "away": away_sim,
                "home_low": home_low, "away_low": away_low,
                "home_high": home_high, "away_high": away_high,
            }

        pack = st.session_state.get("team_game_sim")
        if pack and pack.get("fingerprint") == fingerprint:
            home_sim = pack["home"]
            away_sim = pack["away"]
            home_low = pack["home_low"]
            away_low = pack["away_low"]
            home_high = pack["home_high"]
            away_high = pack["away_high"]

            markets = [
                "PTS","FGA","FGM","3PA","3PM","2PA","2PM",
                "FTA","FTM","REB","OREB","DREB","AST","STL",
                "BLK","TOV","PF",
            ]

            total_sim = home_sim[markets].add(away_sim[markets], fill_value=0)
            total_low = home_low[markets].add(away_low[markets], fill_value=0)
            total_high = home_high[markets].add(away_high[markets], fill_value=0)

            st.markdown("### Automatic model lines + fair prices")
            st.caption(
                "Example: a projection such as 30.7 is converted from the full "
                "simulated distribution to the half-point line that is closest "
                "to a 50/50 model market. Fair prices come from the simulation, "
                "not from rounding 1 / mean."
            )

            subtabs = st.tabs([
                setup["away_abbr"], setup["home_abbr"], "GAME TOTALS", "TEAM WITH MOST"
            ])
            with subtabs[0]:
                st.dataframe(
                    auto_market_table(away_sim, markets, away_low, away_high, target_ev=target_ev, reference_odds=reference_odds).round(3),
                    use_container_width=True, hide_index=True,
                )
            with subtabs[1]:
                st.dataframe(
                    auto_market_table(home_sim, markets, home_low, home_high, target_ev=target_ev, reference_odds=reference_odds).round(3),
                    use_container_width=True, hide_index=True,
                )
            with subtabs[2]:
                st.dataframe(
                    auto_market_table(total_sim, markets, total_low, total_high, target_ev=target_ev, reference_odds=reference_odds).round(3),
                    use_container_width=True, hide_index=True,
                )
            with subtabs[3]:
                most_rows = []
                most_audit_rows = []
                st.caption(
                    "PTS / match-winner is intentionally omitted. Team-with-most keeps the simulated mean difference, "
                    "but calibrates the DIFFERENCE spread to actual same-game league history; bookmaker prices are never used."
                )
                for m in [
                    "3PM","3PA","2PM","2PA","FTM","FTA",
                    "REB","OREB","DREB","AST","STL","BLK","TOV","PF",
                ]:
                    pr = most_market_calibrated(
                        home_sim[m], away_sim[m], team_logs=team_db, market=m,
                        calibration_strength=0.60,
                    )
                    most_rows.append({
                        "Market": m,
                        f"P {setup['home_abbr']}": pr["p_home"],
                        f"Fair {setup['home_abbr']}": pr["fair_home"],
                        f"Play {setup['home_abbr']} from": required_odds_for_ev(pr["p_home"], 0.0, target_ev),
                        "P Tie": pr["p_tie"],
                        "Fair Tie": pr["fair_tie"],
                        f"P {setup['away_abbr']}": pr["p_away"],
                        f"Fair {setup['away_abbr']}": pr["fair_away"],
                        f"Play {setup['away_abbr']} from": required_odds_for_ev(pr["p_away"], 0.0, target_ev),
                    })
                    most_audit_rows.append({
                        "Market": m,
                        "Raw tie": pr.get("raw_p_tie", pr["p_tie"]),
                        "Calibrated tie": pr["p_tie"],
                        "League tie": pr.get("league_tie_rate", np.nan),
                        "Raw diff SD": pr.get("raw_diff_sd", np.nan),
                        "League target diff SD": pr.get("league_target_diff_sd", np.nan),
                        "Applied diff SD": pr.get("applied_diff_sd", np.nan),
                        "Calibration games": pr.get("calibration_games", 0),
                    })
                st.dataframe(
                    pd.DataFrame(most_rows).round(3),
                    use_container_width=True, hide_index=True,
                )
                with st.expander("Team-with-most calibration audit", expanded=False):
                    st.dataframe(
                        pd.DataFrame(most_audit_rows).round(3),
                        use_container_width=True, hide_index=True,
                    )


            st.markdown("### Exact bookmaker-line comparison / arb audit")
            st.caption(
                "Upload a CSV after the model is frozen. Two-way rows: Scope,Market,Line,Over Odds,Under Odds. "
                "MOST rows: Scope=MOST, Market, Away Odds,Tie Odds,Home Odds. The app prices the EXACT bookmaker line "
                "from the current simulation. 'Book arb' is a real same-book overround check; model-vs-book EV is NOT an executable arbitrage by itself."
            )
            bm_file = st.file_uploader("Bookmaker market CSV", type=["csv"], key="team_book_csv")
            if bm_file is not None:
                try:
                    bm = pd.read_csv(bm_file)
                    bm.columns = [str(c).strip() for c in bm.columns]
                    comp_rows = []
                    sim_by_scope = {
                        "TOTAL": total_sim,
                        str(setup["away_abbr"]).upper(): away_sim,
                        str(setup["home_abbr"]).upper(): home_sim,
                    }
                    for _, br in bm.iterrows():
                        scope = str(br.get("Scope", "")).strip().upper()
                        market = str(br.get("Market", "")).strip().upper()
                        if not market:
                            continue
                        if scope == "MOST":
                            if market not in home_sim.columns or market not in away_sim.columns:
                                continue
                            pr = most_market_calibrated(
                                home_sim[market], away_sim[market], team_logs=team_db, market=market,
                                calibration_strength=0.60,
                            )
                            ao = float(br.get("Away Odds", np.nan)); to = float(br.get("Tie Odds", np.nan)); ho = float(br.get("Home Odds", np.nan))
                            book_sum = sum(1.0/x for x in [ao,to,ho] if np.isfinite(x) and x > 1.0)
                            comp_rows.append({
                                "Scope":"MOST","Market":market,"Line":np.nan,
                                f"Book {setup['away_abbr']}":ao, f"Model fair {setup['away_abbr']}":pr["fair_away"], f"EV {setup['away_abbr']}":pr["p_away"]*ao-1 if np.isfinite(ao) else np.nan,
                                "Book Tie":to,"Model fair Tie":pr["fair_tie"],"EV Tie":pr["p_tie"]*to-1 if np.isfinite(to) else np.nan,
                                f"Book {setup['home_abbr']}":ho, f"Model fair {setup['home_abbr']}":pr["fair_home"], f"EV {setup['home_abbr']}":pr["p_home"]*ho-1 if np.isfinite(ho) else np.nan,
                                "Book implied sum":book_sum,"Actual book arb?": bool(book_sum < 1.0) if book_sum > 0 else False,
                            })
                            continue

                        sim_scope = sim_by_scope.get(scope)
                        if sim_scope is None or market not in sim_scope.columns:
                            continue
                        line = float(br.get("Line", np.nan)); oo = float(br.get("Over Odds", np.nan)); uo = float(br.get("Under Odds", np.nan))
                        if not (np.isfinite(line) and np.isfinite(oo) and np.isfinite(uo)):
                            continue
                        pr = price(sim_scope[market].to_numpy(), line, oo, uo)
                        book_sum = 1.0/oo + 1.0/uo
                        comp_rows.append({
                            "Scope":scope,"Market":market,"Line":line,
                            "Book O":oo,"Model fair O":pr["fair_over"],"Model P O":pr["p_over"],"EV O":pr["ev_over"],"Play O from":required_odds_for_ev(pr["p_over"],pr["p_push"],target_ev),
                            "Book U":uo,"Model fair U":pr["fair_under"],"Model P U":pr["p_under"],"EV U":pr["ev_under"],"Play U from":required_odds_for_ev(pr["p_under"],pr["p_push"],target_ev),
                            "Push P":pr["p_push"],"Book implied sum":book_sum,"Actual book arb?":book_sum < 1.0,
                        })
                    if comp_rows:
                        comp_df = pd.DataFrame(comp_rows)
                        st.dataframe(comp_df.round(4), use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download exact model-vs-book comparison CSV",
                            comp_df.to_csv(index=False).encode("utf-8"),
                            file_name="model_vs_book_exact.csv", mime="text/csv", key="download_exact_book_compare"
                        )
                    else:
                        st.warning("No valid comparison rows found. Check Scope/team abbreviations and column names.")
                except Exception as exc:
                    st.error(f"Could not read bookmaker comparison CSV: {exc}")

            st.caption("No bookmaker input is required for pricing. The upload above is comparison-only and never feeds back into the model.")

            with st.expander("Model audit: buckets / location / H2H / conservation"):
                st.markdown("**Possession identity audit**")
                _poss_rows = []
                for _abbr, _sim in [(setup["away_abbr"], away_sim), (setup["home_abbr"], home_sim)]:
                    _m = _sim.mean(numeric_only=True)
                    _recon = float(_m.get("FGA",0.0) - _m.get("OREB",0.0) + _m.get("TOV",0.0) + 0.44*_m.get("FTA",0.0))
                    _poss = float(_m.get("POSS", shared_pace))
                    _poss_rows.append({
                        "Team": _abbr, "Sim POSS": _poss,
                        "FGA-OREB+TOV+0.44FTA": _recon, "Identity gap": _recon-_poss,
                    })
                st.dataframe(pd.DataFrame(_poss_rows).round(3), use_container_width=True, hide_index=True)
                st.caption("v2.16 target: the box-score possession identity should reconcile to the shared pace in expectation; a persistent multi-possession gap is a model bug, not a pace opinion.")

                st.markdown("**Current defensive OUT bridge (individual ON/OFF splits)**")
                st.caption(
                    "For each current OUT: >=10 MIN = ON, 0 MIN = OFF, 0<MIN<10 excluded. "
                    "Splits begin at that player's first appearance for the team and exclude current H2H. "
                    "Combined modifiers are shrunk and overlap-protected; they adjust only the opponent offense."
                )
                if isinstance(home_def_out_audit, pd.DataFrame) and not home_def_out_audit.empty:
                    st.markdown(f"**{setup['home_abbr']} defense OUT audit → affects {setup['away_abbr']} offense**")
                    st.dataframe(home_def_out_audit.round(4), use_container_width=True, hide_index=True)
                if isinstance(away_def_out_audit, pd.DataFrame) and not away_def_out_audit.empty:
                    st.markdown(f"**{setup['away_abbr']} defense OUT audit → affects {setup['home_abbr']} offense**")
                    st.dataframe(away_def_out_audit.round(4), use_container_width=True, hide_index=True)
                st.markdown(f"**{setup['away_abbr']} buckets — {away_regime}**")
                st.dataframe(away_audit.round(4), use_container_width=True, hide_index=True)
                st.markdown(f"**{setup['home_abbr']} buckets — {home_regime}**")
                st.dataframe(home_audit.round(4), use_container_width=True, hide_index=True)

                st.markdown("**Confirmed OUT exact/near-state + residual rotation — overlap audit**")
                st.caption(
                    "Availability similarity and residual rotation are both INNER-bucket weights. Each game receives one availability score per stat. "
                    "Selected OUT names are removed from residual Jaccard, so the same absence is not counted twice. "
                    "GTD/Q is not modeled unless you explicitly select OUT."
                )
                st.dataframe(
                    pd.concat([away_avail_audit, home_avail_audit], ignore_index=True).round(3),
                    use_container_width=True, hide_index=True,
                )

                st.markdown("**Roster-state bridge — OUT fallback + minute restrictions**")
                st.caption(
                    "Healthy / OUT-only / current are 200-minute synthetic counterfactuals. The OUT bridge fades separately by stat as "
                    "exact/near-state confidence rises; shared minute restrictions are a separate current-state layer."
                )
                ra1, ra2 = st.columns(2)
                with ra1:
                    st.caption(setup["away_abbr"])
                    st.dataframe(away_rot_impact.team_audit.round(4), use_container_width=True, hide_index=True)
                    st.dataframe(
                        away_rot_impact.current_minutes[["Player","Status","Projected Min","Source"]].round(2),
                        use_container_width=True, hide_index=True,
                    )
                with ra2:
                    st.caption(setup["home_abbr"])
                    st.dataframe(home_rot_impact.team_audit.round(4), use_container_width=True, hide_index=True)
                    st.dataframe(
                        home_rot_impact.current_minutes[["Player","Status","Projected Min","Source"]].round(2),
                        use_container_width=True, hide_index=True,
                    )

                def _inner_weight_summary(team_abbr, game_weights, regime, out_names):
                    ws = np.asarray(list(game_weights.values()), dtype=float) if game_weights else np.asarray([])
                    return {
                        "Team": team_abbr,
                        "Regime": regime,
                        "Confirmed OUT": ", ".join(out_names) if out_names else "—",
                        "Residual rotation": "ON 0.85–1.00" if rotation_similarity_enabled else "OFF",
                        "Historical games weighted": int(len(ws)),
                        "Min combined inner weight": float(ws.min()) if ws.size else 1.0,
                        "Mean combined inner weight": float(ws.mean()) if ws.size else 1.0,
                        "Max combined inner weight": float(ws.max()) if ws.size else 1.0,
                    }

                st.dataframe(pd.DataFrame([
                    _inner_weight_summary(setup["away_abbr"], away_game_weights, away_regime, away_out),
                    _inner_weight_summary(setup["home_abbr"], home_game_weights, home_regime, home_out),
                ]).round(3), use_container_width=True, hide_index=True)

                st.markdown("**Shot architecture audit — FGA → 3P share → 3PA/2PA**")
                st.dataframe(pd.DataFrame([
                    {
                        "Team": setup["away_abbr"],
                        "Base FGA/live": away_profile.get("fga_live", np.nan),
                        "Base 3P share": away_profile.get("three_share", np.nan),
                        "Final FGA context mod": away_mod.get("FGA", 1.0),
                        "Final 3P-share context mod": away_mod.get("3P_SHARE", 1.0),
                        "Final FTA context mod": away_mod.get("FTA", 1.0),
                        "Roster 3P-share mod": away_roster_mod.get("3P_SHARE", 1.0),
                        "Roster FTA mod": away_roster_mod.get("FTA", 1.0),
                    },
                    {
                        "Team": setup["home_abbr"],
                        "Base FGA/live": home_profile.get("fga_live", np.nan),
                        "Base 3P share": home_profile.get("three_share", np.nan),
                        "Final FGA context mod": home_mod.get("FGA", 1.0),
                        "Final 3P-share context mod": home_mod.get("3P_SHARE", 1.0),
                        "Final FTA context mod": home_mod.get("FTA", 1.0),
                        "Roster 3P-share mod": home_roster_mod.get("3P_SHARE", 1.0),
                        "Roster FTA mod": home_roster_mod.get("FTA", 1.0),
                    },
                ]).round(4), use_container_width=True, hide_index=True)

                st.markdown("**BLK v3 audit — own ability / opponent suffered / H2H / position / tiny 2PA**")
                def _h2h_blk_weight(audit):
                    if audit is None or audit.empty:
                        return 0.0
                    hit = audit[audit["Feature"].astype(str).eq("blk_rate_pp")]
                    return float(hit.iloc[0]["Applied H2H weight"]) if not hit.empty else 0.0

                blk_audit = pd.DataFrame([
                    {
                        "Team": setup["away_abbr"],
                        "Base non-H2H BLK/poss": away_profile.get("blk_pp", np.nan),
                        "Final own BLK/poss after H2H": away_profile.get("blk_rate_pp", np.nan),
                        "Opponent BLK-suffered modifier": away_auto.get("BLK", 1.0),
                        "Roster BLK modifier": away_roster_mod.get("BLK", 1.0),
                        "Location BLK modifier": away_loc.get("BLK", 1.0),
                        "Positional relative modifier": away_blk_pos,
                        "H2H weight": _h2h_blk_weight(away_h2h_audit),
                        "2PA effect cap": "±2%",
                    },
                    {
                        "Team": setup["home_abbr"],
                        "Base non-H2H BLK/poss": home_profile.get("blk_pp", np.nan),
                        "Final own BLK/poss after H2H": home_profile.get("blk_rate_pp", np.nan),
                        "Opponent BLK-suffered modifier": home_auto.get("BLK", 1.0),
                        "Roster BLK modifier": home_roster_mod.get("BLK", 1.0),
                        "Location BLK modifier": home_loc.get("BLK", 1.0),
                        "Positional relative modifier": home_blk_pos,
                        "H2H weight": _h2h_blk_weight(home_h2h_audit),
                        "2PA effect cap": "±2%",
                    },
                ])
                st.dataframe(blk_audit.round(4), use_container_width=True, hide_index=True)

                hb1, hb2 = st.columns(2)
                with hb1:
                    st.caption(f"{setup['away_abbr']} disjoint H2H structural blend")
                    st.dataframe(away_h2h_audit.round(4), use_container_width=True, hide_index=True)
                with hb2:
                    st.caption(f"{setup['home_abbr']} disjoint H2H structural blend")
                    st.dataframe(home_h2h_audit.round(4), use_container_width=True, hide_index=True)

                with st.expander("Positional BLK susceptibility audit", expanded=False):
                    st.caption(
                        "Applied only if the loaded player dataset contains a real BLKA/blocked-attempt field. "
                        "Otherwise the modifier is exactly 1.00; the model never infers positional blocks from 2PA."
                    )
                    pb1, pb2 = st.columns(2)
                    pb1.dataframe(away_blk_pos_audit.round(4), use_container_width=True, hide_index=True)
                    pb2.dataframe(home_blk_pos_audit.round(4), use_container_width=True, hide_index=True)

                st.markdown("**Location correction (small shrink only)**")
                ca, ch = st.columns(2)
                ca.dataframe(away_loc_audit.round(4), use_container_width=True, hide_index=True)
                ch.dataframe(home_loc_audit.round(4), use_container_width=True, hide_index=True)

                st.markdown("**H2H raw rows — separated from baseline and opponent-allowed samples**")
                if h2h_all.empty:
                    st.caption("No same-season H2H rows found.")
                else:
                    st.dataframe(h2h_all, use_container_width=True, hide_index=True)

                # Explicit conservation checks from the actual simulations.
                checks = {
                    "Away FGA = 2PA + 3PA": bool(((away_sim["FGA"] - away_sim["2PA"] - away_sim["3PA"]) == 0).all()),
                    "Home FGA = 2PA + 3PA": bool(((home_sim["FGA"] - home_sim["2PA"] - home_sim["3PA"]) == 0).all()),
                    "Away REB = OREB + DREB": bool(((away_sim["REB"] - away_sim["OREB"] - away_sim["DREB"]) == 0).all()),
                    "Home REB = OREB + DREB": bool(((home_sim["REB"] - home_sim["OREB"] - home_sim["DREB"]) == 0).all()),
                    "Away STL <= Home TOV": bool((away_sim["STL"] <= home_sim["TOV"]).all()),
                    "Home STL <= Away TOV": bool((home_sim["STL"] <= away_sim["TOV"]).all()),
                    "Away BLK <= Home missed FGA": bool((away_sim["BLK"] <= (home_sim["FGA"]-home_sim["FGM"])).all()),
                    "Home BLK <= Away missed FGA": bool((home_sim["BLK"] <= (away_sim["FGA"]-away_sim["FGM"])).all()),
                }
                st.json(checks)


# =====================================================================
# PLAYER PROPS
# =====================================================================
with tab_player:
    st.subheader("Player Props")

    setup = st.session_state.get("game_setup")
    shared_pace = current_game_pace()

    if data_pack is None:
        st.info("Load the database first.")
    elif not setup or shared_pace is None:
        st.info("Set the matchup first.")
    else:
        provider = SportsDataverseWNBA()
        raw_pool = provider.current_player_pool(player_db)
        manual_context = st.session_state.get("game_context", {})
        away_prop_out = _context_out_names_for_team(manual_context, raw_pool, setup["away_abbr"])
        home_prop_out = _context_out_names_for_team(manual_context, raw_pool, setup["home_abbr"])
        pool = augment_current_pool(raw_pool, player_db, setup["away_abbr"], away_prop_out)
        pool = augment_current_pool(pool, player_db, setup["home_abbr"], home_prop_out)
        player_availability_k = float(st.session_state.get("availability_state_k", 6.0))
        st.info(
            "Confirmed OUT used by Player Props — "
            f"{setup['away_abbr']}: {', '.join(away_prop_out) if away_prop_out else '—'} | "
            f"{setup['home_abbr']}: {', '.join(home_prop_out) if home_prop_out else '—'}. "
            "Change OUT players in Game Setup; the same state is used here automatically."
        )

        # Background minutes engine always projects the whole rotation so the
        # team stays at 200 regulation minutes.
        base_away = project_team_minutes(
            player_db, team_db, pool,
            setup["away_abbr"], setup["away_name"],
            manual_context,
        )
        base_home = project_team_minutes(
            player_db, team_db, pool,
            setup["home_abbr"], setup["home_name"],
            manual_context,
        )
        base_minutes = concat_without_attrs(
            [base_away, base_home], ignore_index=True
        )

        away_role_impact = build_rotation_state_impact(
            player_db, team_db, raw_pool, setup["away_abbr"], setup["away_name"],
            manual_context, away_prop_out, exact_state_confidence=0.0,
        )
        home_role_impact = build_rotation_state_impact(
            player_db, team_db, raw_pool, setup["home_abbr"], setup["home_name"],
            manual_context, home_prop_out, exact_state_confidence=0.0,
        )
        raw_role_mods = pd.concat([
            away_role_impact.raw_player_role_modifiers,
            home_role_impact.raw_player_role_modifiers,
        ], ignore_index=True) if (
            not away_role_impact.raw_player_role_modifiers.empty
            or not home_role_impact.raw_player_role_modifiers.empty
        ) else pd.DataFrame()

        st.markdown("### 1. Select players")
        all_options = []
        for _, r in base_minutes.sort_values(
            ["Team","Projected Min"], ascending=[True,False]
        ).iterrows():
            all_options.append(f"{r['Player']} — {r['Team']}")

        selected_labels = st.multiselect(
            "Click only the players you want to price",
            all_options,
            default=st.session_state.get("last_selected_players", []),
            key="selected_players",
        )
        st.session_state["last_selected_players"] = selected_labels

        selected_names = [
            label.rsplit(" — ", 1)[0]
            for label in selected_labels
        ]

        if selected_names:
            st.markdown("### 2. Minutes: AUTO or trader override")
            st.caption(
                "AUTO uses rotation similarity, starter history, OT/blowout "
                "downweighting, non-overlapping buckets and a 200-minute team "
                "constraint. Enter 0 to keep AUTO; any positive value is a LOCAL scenario override. "
                "For a real injury restriction/return that must also move Team Markets, set it in Game Setup → Shared minute overrides."
            )

            override_cols = st.columns(
                min(4, max(1, len(selected_names)))
            )
            ui_overrides = {}
            for i, pname in enumerate(selected_names):
                current_row = base_minutes[
                    base_minutes["Player"] == pname
                ].iloc[0]
                with override_cols[i % len(override_cols)]:
                    val = st.number_input(
                        f"{pname} override (0=AUTO)",
                        min_value=0.0,
                        max_value=40.0,
                        value=0.0,
                        step=0.5,
                        key=f"sel_override_{pname}",
                        help=(
                            f"AUTO currently {current_row['Projected Min']:.1f} "
                            f"({current_row['Source']})"
                        ),
                    )
                    if val > 0:
                        ui_overrides[pname] = val

            # Rebalance each whole team after selected trader overrides.
            away_overrides = {
                k:v for k,v in ui_overrides.items()
                if not base_away[base_away["Player"] == k].empty
            }
            home_overrides = {
                k:v for k,v in ui_overrides.items()
                if not base_home[base_home["Player"] == k].empty
            }

            away_min = project_team_minutes(
                player_db, team_db, pool,
                setup["away_abbr"], setup["away_name"],
                manual_context, away_overrides,
            )
            home_min = project_team_minutes(
                player_db, team_db, pool,
                setup["home_abbr"], setup["home_name"],
                manual_context, home_overrides,
            )
            final_minutes = concat_without_attrs(
                [away_min, home_min], ignore_index=True
            )

            selected_min = final_minutes[
                final_minutes["Player"].isin(selected_names)
            ][[
                "Team","Player","Status","Auto Baseline Min",
                "Projected Min","Override Delta","Low Min",
                "High Min","Minutes SD","Source","Starter P",
                "Rotation Similarity","Regime"
            ]].copy()

            st.dataframe(
                selected_min.round({
                    "Projected Min":1,
                    "Low Min":1,
                    "High Min":1,
                    "Minutes SD":2,
                    "Starter P":2,
                    "Rotation Similarity":2,
                }),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("Full rotation minute audit"):
                st.caption(
                    "Each team is constrained to 200 regulation minutes. "
                    "OUT/trader/metadata minutes are fixed first; AUTO minutes "
                    "absorb the remaining allocation."
                )
                minute_cols = [c for c in [
                    "Team","Player","Status","In Active Rotation",
                    "DNP-aware L5 Min","DNP-aware L10 Min",
                    "Healthy Baseline Min","Auto Baseline Min","OUT Replacement Delta",
                    "Projected Min","Override Delta","Low Min","High Min","Source","Regime"
                ] if c in final_minutes.columns]
                st.dataframe(
                    final_minutes[minute_cols].round(2),
                    use_container_width=True, hide_index=True,
                )
                sums = final_minutes.groupby("Team")["Projected Min"].sum()
                st.write({
                    team: round(float(total), 2)
                    for team,total in sums.items()
                })

                st.markdown("#### Confirmed OUT minute replacement")
                out_impacts = []
                for team_frame in [away_min, home_min]:
                    imp = team_frame.attrs.get("out_redistribution_impact")
                    if isinstance(imp, pd.DataFrame) and not imp.empty:
                        out_impacts.append(imp)
                if out_impacts:
                    st.dataframe(pd.concat(out_impacts, ignore_index=True).round(2), use_container_width=True, hide_index=True)
                else:
                    st.caption("No confirmed OUT minute redistribution in this scenario.")

                st.markdown("#### Role-aware override impact")
                impacts = []
                for team_frame in [away_min, home_min]:
                    imp = team_frame.attrs.get("override_impact")
                    if isinstance(imp, pd.DataFrame) and not imp.empty:
                        impacts.append(imp)
                if impacts:
                    st.dataframe(
                        concat_without_attrs(impacts, ignore_index=True).round(3),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("No explicit minute override is active.")

                st.markdown("#### Learned replacement weights")
                selected_focus = st.selectbox(
                    "Show replacement matrix for",
                    selected_names,
                    key="replacement_focus",
                )
                matrix_parts = []
                for team_frame in [away_min, home_min]:
                    aud = team_frame.attrs.get(
                        "redistribution_matrix_audit"
                    )
                    if isinstance(aud, pd.DataFrame) and not aud.empty:
                        part = aud[
                            aud["Focal"] == selected_focus
                        ].copy()
                        if not part.empty:
                            matrix_parts.append(part)
                if matrix_parts:
                    matrix_view = concat_without_attrs(
                        matrix_parts, ignore_index=True
                    ).sort_values("Weight", ascending=False)
                    st.dataframe(
                        matrix_view[[
                            "Focal","Replacement","Weight","games",
                            "neg_slope","onoff","confidence",
                            "role_prior","Focal Pos","Replacement Pos"
                        ]].round(3),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "Weight = historically learned inverse-minutes relation "
                        "+ on/off lift, shrunk toward a small positional prior "
                        "when the sample is thin."
                    )

            st.markdown("### 3. Shared pace")
            st.info(
                f"Game possessions: **{shared_pace:.2f}**. "
                "Each player's historical pace environment is calculated "
                "separately; only the relative pace difference is applied."
            )

            n = st.select_slider(
                "Simulations per selected player",
                [10_000,25_000,50_000,100_000,250_000],
                50_000,
                key="selected_sims",
            )
            same_role_enabled = st.toggle(
                "AUTO: exact + near confirmed-OUT state for player per-minute role",
                value=True,
                help=(
                    "Every historical game gets one stat-specific similarity score to today's OUT state. A 4/5 game is not "
                    "reused as 3/5 or 2/5. Position/role relevance changes by stat, while minutes remain separate in the 200-minute engine."
                ),
                key="same_role_absence_toggle",
            )

            if st.button(
                "Run selected players",
                type="primary",
                key="run_selected",
            ):
                board_rows = []
                detail_store = {}
                player_h2h_prior_minutes, player_h2h_calibration_audit = cached_player_h2h_prior_minutes(player_db)

                for pname in selected_names:
                    mr = final_minutes[
                        final_minutes["Player"] == pname
                    ].iloc[0]

                    if str(mr["Status"]).upper() == "OUT":
                        continue

                    pool_row = pool[
                        pool["PLAYER_NAME"] == pname
                    ]
                    if pool_row.empty:
                        continue
                    prow = pool_row.iloc[0]
                    pid = prow["PLAYER_ID"]
                    team_abbr = str(prow["TEAM_ABBR"])

                    if team_abbr == setup["away_abbr"]:
                        opp_abbr = setup["home_abbr"]
                        overall_profile = st.session_state["opp_profile_home"]
                    else:
                        opp_abbr = setup["away_abbr"]
                        overall_profile = st.session_state["opp_profile_away"]

                    plog = clean_player_log(
                        player_db[player_db["PLAYER_ID"] == pid].copy()
                    )

                    # Player injury logic has three deliberately separate layers:
                    #   1) 200-minute engine -> MINUTES after confirmed OUTs
                    #   2) exact-state inner weights -> PER-MINUTE role/opportunity rates
                    #   3) explicit trader role multipliers -> only if trader intentionally adds them
                    # The OUT state itself must not also trigger 35/20/45 automatically.
                    role = context_role(manual_context, pname)
                    regime = str(mr["Regime"])
                    out_teammates = current_out_teammates(
                        manual_context, pool, team_abbr, pname
                    )

                    if same_role_enabled and out_teammates:
                        player_state_stats = ["FGA", "3PA", "FTA", "REB", "AST"]
                        role_game_weights_by_stat, same_role_audit, role_state_scores = availability_similarity_weight_maps(
                            player_db, plog, team_abbr, out_teammates, player_state_stats,
                            current_pool=pool, focal_player=pname,
                            k=float(player_availability_k), maturity_games=5.0,
                        )
                        same_role_audit["Profile overlap guard"] = (
                            "Single score per historical game/stat; INNER weighting only inside Old/G6-10/L5. "
                            "No nested 4/5→3/5→2/5 samples."
                        )
                        # Same absence must not ALSO become the generic role-change outer profile.
                        cfg = WeightConfig.role_change() if role else WeightConfig.stable()
                    else:
                        role_game_weights_by_stat = {}
                        role_state_scores = {}
                        same_role_audit = pd.DataFrame([{
                            "Stat": "—",
                            "Confirmed OUT state": "—",
                            "Exact-state games": 0,
                            "Evidence mass": 0.0,
                            "State confidence": 0.0,
                            "Profile overlap guard": "Neutral: no confirmed OUT teammate state selected.",
                        }])
                        cfg = (
                            WeightConfig.role_change()
                            if regime == "role_change" or role
                            else WeightConfig.stable()
                        )

                    profile,audit = build_player_profile(
                        plog,cfg,game_weights_by_stat=role_game_weights_by_stat,
                        exclude_opponent_abbr=opp_abbr,
                    )

                    player_conf_by_stat = confidence_by_stat(same_role_audit)
                    fallback = {
                        "usage": 1.0, "three_role": 1.0, "fta_role": 1.0,
                        "creation": 1.0, "reb_role": 1.0,
                    }
                    fallback_audit = pd.DataFrame()
                    if isinstance(raw_role_mods, pd.DataFrame) and not raw_role_mods.empty:
                        hit = raw_role_mods[raw_role_mods["Player"].astype(str).eq(str(pname))]
                        if not hit.empty:
                            rr = hit.iloc[0]
                            raw_map = {
                                "usage": float(rr.get("Usage fallback", 1.0)),
                                "three_role": float(rr.get("Three-role fallback", 1.0)),
                                "fta_role": float(rr.get("FTA-role fallback", 1.0)),
                                "creation": float(rr.get("Creation fallback", 1.0)),
                                "reb_role": float(rr.get("Rebound-role fallback", 1.0)),
                            }
                            stat_for_role = {
                                "usage": "FGA", "three_role": "3PA", "fta_role": "FTA",
                                "creation": "AST", "reb_role": "REB",
                            }
                            remaining = {}
                            for k, rv in raw_map.items():
                                c = float(player_conf_by_stat.get(stat_for_role[k], 0.0))
                                # v2.16: sparse empirical evidence must not hand
                                # 100% control to the synthetic fallback.  The
                                # residual structural bridge is capped at 65%
                                # credibility until walk-forward calibration.
                                exponent = float(0.65 * np.clip(1.0 - c, 0.0, 1.0))
                                remaining[k] = exponent
                                fallback[k] = float(np.exp(exponent * np.log(max(rv, 1e-6))))
                            fallback_audit = pd.DataFrame([{
                                "Player": pname,
                                **{f"Confidence {st}": float(player_conf_by_stat.get(st, 0.0)) for st in ("FGA","3PA","FTA","AST","REB")},
                                **{f"Remaining {k}": v for k, v in remaining.items()},
                                **{f"Raw {k}": v for k, v in raw_map.items()},
                                **{f"Applied {k}": v for k, v in fallback.items()},
                            }])

                    pos_group = prow.get("POSITION_GROUP")
                    pvo,plg = {},{}
                    if pos_group:
                        pvo,plg = position_environment(
                            player_db,opp_abbr,pos_group
                        )
                    matchup_mods, matchup_audit = player_matchup_modifiers(
                        overall_profile,pvo,plg
                    )

                    historical_pace = player_historical_pace_environment(
                        plog, team_db, cfg
                    )
                    pace_mult = float(
                        np.clip(shared_pace / max(historical_pace, 1.0), .88, 1.12)
                    )

                    # v2.17.2: H2H opportunity evidence is disjoint from the
                    # profile above and pooled by a league-learned empirical-Bayes
                    # prior-minute mass.  No universal 5% cap is used.
                    _ph2h = plog[
                        plog["OPP_ABBR"].astype(str).str.upper().eq(str(opp_abbr).upper())
                    ].copy()
                    _ph2h_ids = _ph2h["GAME_ID"].astype(str).tolist() if "GAME_ID" in _ph2h.columns else []
                    _ph2h_rot = h2h_rotation_similarity(
                        player_db, pool, team_abbr, manual_context, _ph2h_ids
                    ) if _ph2h_ids else 0.0
                    h2h_mods, player_h2h_audit = player_h2h_modifiers(
                        plog, opp_abbr, profile, float(mr["Projected Min"]),
                        rotation_similarity=float(_ph2h_rot),
                        prior_minutes_by_stat=player_h2h_prior_minutes,
                    )

                    ctx = PlayerContext(
                        projected_minutes=float(mr["Projected Min"]),
                        minutes_sd=float(mr["Minutes SD"]),
                        pace_multiplier=pace_mult,
                        opp_pts=float(matchup_mods["PTS"]),
                        opp_reb=float(matchup_mods["REB"]),
                        opp_ast=float(matchup_mods["AST"]),
                        opp_3pa=float(matchup_mods["3PA"]),
                        opp_2pa=float(matchup_mods.get("2PA",1.0)),
                        opp_fta=float(matchup_mods["FTA"]),
                        opp_three_pct=float(matchup_mods.get("3P_PCT",1.0)),
                        opp_two_pct=float(matchup_mods.get("2P_PCT",1.0)),
                        usage=float(role.get("usage",1.0)) * fallback["usage"],
                        creation=float(role.get("creation",1.0)) * fallback["creation"],
                        reb_role=float(
                            role.get("reb_role",role.get("reb",1.0))
                        ) * fallback["reb_role"],
                        three_role=float(
                            role.get("three_role",role.get("three_pa",1.0))
                        ) * fallback["three_role"],
                        fta_role=float(role.get("fta_role",1.0)) * fallback["fta_role"],
                        h2h_2pa=float(h2h_mods.get("2PA",1.0)),
                        h2h_3pa=float(h2h_mods.get("3PA",1.0)),
                        h2h_reb=float(h2h_mods.get("REB",1.0)),
                        h2h_ast=float(h2h_mods.get("AST",1.0)),
                    )

                    seed = (abs(hash(str(pid))) % 100000) + 1000
                    sim = simulate_player(
                        profile,ctx,int(n),seed=seed
                    )

                    board_rows.append({
                        "Team":team_abbr,
                        "Player":pname,
                        "Min":float(mr["Projected Min"]),
                        "Min Low":float(mr["Low Min"]),
                        "Min High":float(mr["High Min"]),
                        "Min Source":mr["Source"],
                        "Hist Pace":historical_pace,
                        "Game Pace":shared_pace,
                        "Pace Mult":pace_mult,
                        "PTS":float(sim["PTS"].mean()),
                        "REB":float(sim["REB"].mean()),
                        "AST":float(sim["AST"].mean()),
                        "3PM":float(sim["3PM"].mean()),
                        "3PA":float(sim["3PA"].mean()),
                        "FTA":float(sim["FTA"].mean()),
                        "PRA":float(sim["PRA"].mean()),
                        "PR":float(sim["PR"].mean()),
                        "PA":float(sim["PA"].mean()),
                        "AR":float(sim["AR"].mean()),
                    })

                    detail_store[pname] = {
                        "sim":sim,
                        "profile_audit":audit,
                        "role_state_audit":profile.get("_role_state_audit"),
                        "matchup_audit":matchup_audit,
                        "same_role_audit":same_role_audit,
                        "availability_fallback_audit":fallback_audit,
                        "h2h_audit":player_h2h_audit,
                        "h2h_calibration_audit":player_h2h_calibration_audit,
                        "ctx":ctx,
                        "plog":plog,
                        "opp_abbr":opp_abbr,
                        "auto_market_table": auto_market_table(
                            sim, ["PTS","REB","AST","3PM","3PA","FTA","PRA","PR","PA","AR"],
                            target_ev=target_ev, reference_odds=reference_odds,
                        ),
                    }

                st.session_state["selected_player_board"] = pd.DataFrame(
                    board_rows
                )
                st.session_state["selected_player_details"] = detail_store

            board = st.session_state.get("selected_player_board")
            if isinstance(board,pd.DataFrame) and not board.empty:
                st.markdown("### Selected-player projections")
                st.dataframe(
                    board.round({
                        "Min":1,"Min Low":1,"Min High":1,
                        "Hist Pace":2,"Game Pace":2,"Pace Mult":3,
                        "PTS":2,"REB":2,"AST":2,"3PM":2,"3PA":2,"FTA":2,
                        "PRA":2,"PR":2,"PA":2,"AR":2,
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                _render_player_deep_analysis(
                    board, st.session_state["selected_player_details"],
                    target_ev, reference_odds,
                )
        else:
            st.info("Select at least one player.")


# =====================================================================
# DATA AUDIT
# =====================================================================
with tab_audit:
    st.subheader("Data Audit")

    st.markdown("""
**v2.15 overlap / state protocol**
- Old season / Games 6–10 / L5 stay non-overlapping.
- Stable = 55/20/25; role-change = 35/20/45.
- Confirmed OUT is explicit trader input only; QUESTIONABLE/GTD is not probability-modeled.
- Each historical game receives ONE stat-specific exact/near-state similarity score INSIDE the existing bucket; it is never reused in nested 4/5→3/5→2/5 samples.
- Five fully comparable games is a maturity point, not a hard cutoff: 1–4 games contribute with strong shrinkage.
- A 200-minute roster counterfactual supplies only residual fallback, fading separately by stat as near-state confidence rises.
- Shared projected-minute restrictions/returns are current-state information and affect BOTH team markets and player props.
- Residual Jaccard removes those selected OUT names first and is only 0.85–1.00, so the same absence is not counted twice.
- Same-season H2H rows are removed from baseline buckets, opponent-allowed profiles and location splits before H2H is added back once with a small rotation-aware weight.
- Team shot generation is conditional: possessions → TOV → FGA → 3P share → 3PA/2PA. Independent 3PA/2PA Poisson draws are gone.
- 3P share uses offense style plus the opponent's deviation from league in logit space; extreme shot-profile defenses can now move share materially without replacing offensive identity.
- FTA/poss uses offense/defense log-rate blending, so foul-suppressing defenses can meaningfully pull down a high-FTA offense.
- Team OREB matchup uses OREB per miss; AST matchup uses AST per made FG; TOV remains per possession.
- Shooting percentages remain heavily regressed and are not set by L5 hot/cold results.
- Team BLK = own ability × opponent blocks-suffered × disjoint H2H × optional relative positional susceptibility × tiny 2PA opportunity nudge (max ±2%).
- BLK conservation is against total missed FGA, not only missed 2PA, because three-point attempts can also be blocked.
- Positional BLK is applied only when a real player-level BLKA/blocked-attempt field exists; otherwise it is exactly neutral rather than inferred from 2PA.
- REB still uses available misses; no extra 3PA→REB multiplier is added, avoiding overlap. A 2P/3P rebound split waits for real play-by-play attribution.
- Fair price is break-even only; Play-from price adds the selected target-EV buffer.
- Team-with-most calibration remains downstream of the joint simulation and does not use bookmaker prices.

**Minutes / pace**
- Full team rotation is constrained to 200 regulation minutes.
- Explicit minute overrides use the historically learned replacement matrix.
- One shared Confirmed OUT selector in Game Setup feeds BOTH Team Markets and Player Props.
- Player Props use stat-specific exact/near absence-state partial pooling for per-minute rates. Pre-roster non-appearances are mismatches, never fake OUT games.
- Vacated opportunities are routed through the learned teammate replacement matrix, so guard creation does not become a generic frontcourt boost; residual fallback shrinks separately for FGA/3PA/FTA/AST/REB.
- Pace control is fitted from completed WNBA games with a mild ridge prior.
- The exact same projected possessions feed Team Markets and Player Props.
- Market total/handicap remain audit-only.
    """)

    setup = st.session_state.get("game_setup")
    pace = st.session_state.get("pace_projection")
    if setup:
        st.markdown("### Matchup")
        st.json(setup)

    if pace:
        st.markdown("### Pace fit")
        st.json({
            "home_pace":pace.home_pace,
            "away_pace":pace.away_pace,
            "league_pace":pace.league_pace,
            "fast_weight":pace.fast_weight,
            "slow_weight":pace.slow_weight,
            "auto_central":pace.central,
            "shared_used":current_game_pace(),
            "sd":pace.sd,
            "calibration_games":pace.calibration_games,
        })

    st.markdown("### Trader context")
    st.json(
        st.session_state.get("game_context", {})
        or {"message":"No trader context"}
    )

    if setup:
        hp = st.session_state.get("opp_profile_home")
        ap = st.session_state.get("opp_profile_away")
        if hp:
            st.markdown(f"### {setup['home_abbr']} opponent allowed")
            st.dataframe(hp["audit"].round(4),use_container_width=True)
        if ap:
            st.markdown(f"### {setup['away_abbr']} opponent allowed")
            st.dataframe(ap["audit"].round(4),use_container_width=True)
        thp = st.session_state.get("team_opp_profile_home")
        tap = st.session_state.get("team_opp_profile_away")
        if thp:
            st.markdown(f"### {setup['home_abbr']} TEAM-MARKET opponent allowed (current H2H excluded)")
            st.caption("Raw opponent rates are shown here. Team Markets recompute the final offense-vs-defense interaction for the structural rates using league-calibrated opponent elasticities; the generic modifier column is audit/backward-compatibility only. See the opponent-elasticity audit for the beta actually used.")
            st.dataframe(thp["audit"].round(4), use_container_width=True)
        if tap:
            st.markdown(f"### {setup['away_abbr']} TEAM-MARKET opponent allowed (current H2H excluded)")
            st.caption("Raw opponent rates are shown here. Team Markets recompute the final offense-vs-defense interaction for the structural rates using league-calibrated opponent elasticities; the generic modifier column is audit/backward-compatibility only. See the opponent-elasticity audit for the beta actually used.")
            st.dataframe(tap["audit"].round(4), use_container_width=True)

    st.write({
        "historical_provider":"SportsDataverse",
        "stats_nba_used":False,
        "injuries_guessed_by_model":False,
        "market_total_used_as_model_input":False,
        "shared_pace_engine":True,
        "team_minutes_constraint":200,
    })


st.caption(
    "Model-implied fair odds are not yet historically calibrated true odds. "
    "v2.16.0 adds relevant-OUT player states, stat-specific event routing, an individual defensive-absence bridge with <10 MIN exclusion, and a possession-consistent FGA chain. The v2.15.1 stale-state fix remains. Re-backtest before treating fair odds as calibrated true probabilities."
)
