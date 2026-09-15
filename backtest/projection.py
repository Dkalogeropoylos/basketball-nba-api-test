from __future__ import annotations

import numpy as np
import pandas as pd

from core.buckets import WeightConfig
from core.cleaning import clean_team_log
from core.exposure import regulation_equivalent_factor
from core.team_model import (
    TeamContext,
    build_team_profile,
    estimate_possessions,
    h2h_team_audit,
    h2h_profile_blend,
    team_location_modifiers,
    simulate_game,
)
from core.matchup import (
    opponent_allowed_profile,
    team_matchup_modifiers,
    block_position_susceptibility_modifier,
)
from core.minutes_engine import residual_rotation_similarity_weights, h2h_rotation_similarity
from core.shooting_efficiency import predict_shooting_efficiency_modifiers
from core.pace_engine import _weighted_team_pace
from providers.sportsdataverse_nba import SportsDataverseNBA

from backtest.calibration import LeagueCalibration, predict_structural_modifiers_frozen


MARKETS = [
    "PTS","FGA","FGM","3PA","3PM","2PA","2PM","FTA","FTM",
    "REB","OREB","DREB","AST","STL","BLK","TOV","PF",
]


def _frozen_pace(team_hist: pd.DataFrame, home_abbr: str, away_abbr: str, cal: LeagueCalibration, cfg: WeightConfig):
    home_log = team_hist[team_hist["TEAM_ABBR"].astype(str).str.upper().eq(str(home_abbr).upper())].copy()
    away_log = team_hist[team_hist["TEAM_ABBR"].astype(str).str.upper().eq(str(away_abbr).upper())].copy()
    home = _weighted_team_pace(home_log, cfg)
    away = _weighted_team_pace(away_log, cfg)
    poss = estimate_possessions(team_hist) * regulation_equivalent_factor(team_hist)
    league = float(poss.mean())
    fw = float(cal.pace.get("fast_weight", 0.5))
    sw = float(cal.pace.get("slow_weight", 0.5))
    fast, slow = max(home, away), min(home, away)
    central = league + fw * (fast - league) + sw * (slow - league)
    return {
        "home": float(home), "away": float(away), "league": league,
        "fast_weight": fw, "slow_weight": sw,
        "central": float(np.clip(central, 75.0, 115.0)),
        "sd": float(np.clip(cal.pace.get("rmse", 3.2), 1.5, 6.5)),
    }


def _combined_mods(auto: dict, loc: dict):
    return {
        "FGA": float(np.clip(auto.get("FGA", 1.0) * loc.get("FGA", 1.0), 0.90, 1.10)),
        "3P_SHARE": float(auto.get("3P_SHARE", 1.0) * loc.get("3P_SHARE", 1.0)),
        "FTA": float(auto.get("FTA", 1.0) * loc.get("FTA", 1.0)),
        "TOV": float(auto.get("TOV", 1.0) * loc.get("TOV", 1.0)),
        "OREB": float(np.clip(auto.get("OREB", 1.0) * loc.get("OREB", 1.0), 0.82, 1.18)),
        "AST": float(auto.get("AST", 1.0) * loc.get("AST", 1.0)),
        "PF": float(np.clip(auto.get("PF", 1.0) * loc.get("PF", 1.0), 0.84, 1.16)),
        "DREB": float(np.clip(auto.get("DREB", 1.0) * loc.get("DREB", 1.0), 0.88, 1.12)),
        "STL": float(np.clip(auto.get("STL", 1.0) * loc.get("STL", 1.0), 0.86, 1.14)),
        "BLK": float(np.clip(auto.get("BLK", 1.0) * loc.get("BLK", 1.0), 0.82, 1.18)),
        "3P_PCT": float(np.clip(auto.get("3P_PCT", 1.0) * loc.get("3P_PCT", 1.0), 0.92, 1.08)),
        "2P_PCT": float(np.clip(auto.get("2P_PCT", 1.0) * loc.get("2P_PCT", 1.0), 0.92, 1.08)),
    }


def project_game_pregame(
    team_hist: pd.DataFrame,
    player_hist: pd.DataFrame,
    home_abbr: str,
    away_abbr: str,
    calibration: LeagueCalibration,
    n_sims: int = 3000,
    seed: int = 100,
    use_rotation_similarity: bool = True,
):
    """Leakage-safe team-market projection from history available before tip.

    Test A intentionally applies no historical injury override. It does retain
    pregame-observable rotation similarity when enabled.
    """
    cfg = WeightConfig.stable()
    provider = SportsDataverseNBA()
    pool = provider.current_player_pool(player_hist) if player_hist is not None and not player_hist.empty else pd.DataFrame()

    home_log = clean_team_log(team_hist[team_hist["TEAM_ABBR"].astype(str).str.upper().eq(str(home_abbr).upper())].copy())
    away_log = clean_team_log(team_hist[team_hist["TEAM_ABBR"].astype(str).str.upper().eq(str(away_abbr).upper())].copy())
    if len(home_log) < 8 or len(away_log) < 8:
        raise ValueError("Insufficient team history")

    home_rot_w = {}
    away_rot_w = {}
    if use_rotation_similarity and player_hist is not None and not player_hist.empty and not pool.empty:
        home_rot_w = residual_rotation_similarity_weights(player_hist, pool, home_abbr, {}, out_players=[], residual_strength=0.15)
        away_rot_w = residual_rotation_similarity_weights(player_hist, pool, away_abbr, {}, out_players=[], residual_strength=0.15)

    home_profile, _ = build_team_profile(
        home_log, cfg, league_team_logs=team_hist, game_weights=home_rot_w,
        exclude_opponent_abbr=away_abbr,
    )
    away_profile, _ = build_team_profile(
        away_log, cfg, league_team_logs=team_hist, game_weights=away_rot_w,
        exclude_opponent_abbr=home_abbr,
    )

    h2h_all = h2h_team_audit(team_hist, home_abbr, away_abbr)
    home_ids = h2h_all[h2h_all["TEAM_ABBR"].astype(str).str.upper().eq(str(home_abbr).upper())]["GAME_ID"].astype(str).tolist() if not h2h_all.empty else []
    away_ids = h2h_all[h2h_all["TEAM_ABBR"].astype(str).str.upper().eq(str(away_abbr).upper())]["GAME_ID"].astype(str).tolist() if not h2h_all.empty else []
    home_h2h_sim = h2h_rotation_similarity(player_hist, pool, home_abbr, {}, home_ids) if use_rotation_similarity and home_ids and not pool.empty else (1.0 if home_ids else 0.0)
    away_h2h_sim = h2h_rotation_similarity(player_hist, pool, away_abbr, {}, away_ids) if use_rotation_similarity and away_ids and not pool.empty else (1.0 if away_ids else 0.0)

    skip = {"three_share", "fta_pp", "tov_pp", "assist_per_make"}
    home_profile, _ = h2h_profile_blend(team_hist, home_abbr, away_abbr, home_profile, rotation_similarity=home_h2h_sim, skip_features=skip)
    away_profile, _ = h2h_profile_blend(team_hist, away_abbr, home_abbr, away_profile, rotation_similarity=away_h2h_sim, skip_features=skip)

    home_opp = opponent_allowed_profile(team_hist, away_abbr, exclude_team_abbr=home_abbr)
    away_opp = opponent_allowed_profile(team_hist, home_abbr, exclude_team_abbr=away_abbr)
    home_auto = team_matchup_modifiers(home_opp, own_profile=home_profile, elasticities=calibration.opponent_elasticities)
    away_auto = team_matchup_modifiers(away_opp, own_profile=away_profile, elasticities=calibration.opponent_elasticities)

    home_struct, home_struct_audit = predict_structural_modifiers_frozen(
        team_hist, home_abbr, away_abbr, calibration.structural_models, cfg, home_h2h_sim
    )
    away_struct, away_struct_audit = predict_structural_modifiers_frozen(
        team_hist, away_abbr, home_abbr, calibration.structural_models, cfg, away_h2h_sim
    )
    # Match current v2.18.2 NBA sandbox production wiring exactly.
    for k in ("3P_SHARE", "FTA", "TOV", "AST"):
        home_auto[k] = float(home_struct.get(k, home_auto.get(k, 1.0)))
        away_auto[k] = float(away_struct.get(k, away_auto.get(k, 1.0)))

    home_shoot, _ = predict_shooting_efficiency_modifiers(
        team_hist, home_abbr, away_abbr, home_profile, calibration.shooting_models
    )
    away_shoot, _ = predict_shooting_efficiency_modifiers(
        team_hist, away_abbr, home_abbr, away_profile, calibration.shooting_models
    )
    for k in ("3P_PCT", "2P_PCT"):
        home_auto[k] = float(home_shoot.get(k, home_auto.get(k, 1.0)))
        away_auto[k] = float(away_shoot.get(k, away_auto.get(k, 1.0)))

    home_loc, _ = team_location_modifiers(home_log, True, league_team_logs=team_hist, exclude_opponent_abbr=away_abbr)
    away_loc, _ = team_location_modifiers(away_log, False, league_team_logs=team_hist, exclude_opponent_abbr=home_abbr)
    home_mod = _combined_mods(home_auto, home_loc)
    away_mod = _combined_mods(away_auto, away_loc)

    home_blk_pos = 1.0
    away_blk_pos = 1.0
    if player_hist is not None and not player_hist.empty and not pool.empty:
        home_blk_pos, _ = block_position_susceptibility_modifier(player_hist, away_abbr, current_pool=pool, out_players=[])
        away_blk_pos, _ = block_position_susceptibility_modifier(player_hist, home_abbr, current_pool=pool, out_players=[])

    pace = _frozen_pace(team_hist, home_abbr, away_abbr, calibration, cfg)

    def ctx(mod, blk_pos):
        return TeamContext(
            projected_possessions=pace["central"], possessions_sd=pace["sd"],
            fga=mod["FGA"], three_share=mod["3P_SHARE"],
            three_pct=mod["3P_PCT"], two_pct=mod["2P_PCT"],
            fta=mod["FTA"], tov=mod["TOV"], oreb=mod["OREB"],
            ast=mod["AST"], pf=mod["PF"], dreb=mod["DREB"],
            stl=mod["STL"], blk=mod["BLK"], blk_position=float(blk_pos), blk_h2h=1.0,
        )

    home_sim, away_sim = simulate_game(
        home_profile, away_profile, ctx(home_mod, home_blk_pos), ctx(away_mod, away_blk_pos),
        n=int(n_sims), seed=int(seed),
    )
    return {
        "home": home_sim,
        "away": away_sim,
        "pace": pace,
        "home_h2h_rotation_similarity": float(home_h2h_sim),
        "away_h2h_rotation_similarity": float(away_h2h_sim),
        "home_structural_audit": home_struct_audit,
        "away_structural_audit": away_struct_audit,
    }
