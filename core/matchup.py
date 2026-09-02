from __future__ import annotations

import math
import numpy as np
import pandas as pd


def estimate_possessions(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(df["FGA"], errors="coerce").fillna(0)
        - pd.to_numeric(df["OREB"], errors="coerce").fillna(0)
        + pd.to_numeric(df["TOV"], errors="coerce").fillna(0)
        + 0.44 * pd.to_numeric(df["FTA"], errors="coerce").fillna(0)
    )


def _safe_div(a, b, default=np.nan):
    return float(a) / float(b) if np.isfinite(b) and float(b) > 0 else default


def _rates(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    x = df.copy()
    for c in [
        "PTS", "REB", "AST", "FG3M", "FG3A", "FGM", "FGA", "FTA",
        "TOV", "OREB", "PF", "BLK"
    ]:
        if c in x:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0)
    poss = float(estimate_possessions(x).sum())
    if poss <= 0:
        return {}

    a3 = float(x["FG3A"].sum())
    m3 = float(x["FG3M"].sum())
    fga = float(x["FGA"].sum())
    fgm = float(x["FGM"].sum())
    a2 = max(fga - a3, 0.0)
    m2 = max(fgm - m3, 0.0)
    tov = float(x["TOV"].sum())
    live = max(poss - tov, 1e-9)
    misses = max(fga - fgm, 0.0)

    return {
        "games": int(len(x)),
        "poss": poss,
        # Existing per-possession fields retained for Player Props/backward compatibility.
        "PTS": float(x["PTS"].sum()) / poss,
        "REB": float(x["REB"].sum()) / poss,
        "AST": float(x["AST"].sum()) / poss,
        "3PA": a3 / poss,
        "2PA": a2 / poss,
        "FTA": float(x["FTA"].sum()) / poss,
        "TOV": tov / poss,
        "OREB": float(x["OREB"].sum()) / poss,
        "PF": float(x["PF"].sum()) / poss,
        # For rows representing offenses playing AGAINST an opponent, BLK is the
        # number of blocks made by those offenses against the opponent's shots.
        # When opponent_allowed_profile filters OPP_ABBR==opponent, this therefore
        # measures how many blocks the opponent OFFENSE suffers from its opponents.
        "BLK": float(x["BLK"].sum()) / poss if "BLK" in x.columns else np.nan,
        "3P_PCT": (m3 / a3) if a3 > 0 else np.nan,
        "2P_PCT": (m2 / a2) if a2 > 0 else np.nan,
        "3P_ATT": a3,
        "2P_ATT": a2,
        # Team-market structural rates. These remove avoidable overlap:
        # FGA_LIVE is conditional on no turnover; 3P_SHARE allocates FGA between
        # 3PA/2PA; OREB_PER_MISS is conditional on a rebound opportunity;
        # AST_PER_MAKE is conditional on a made FG.
        "FGA_LIVE": fga / live,
        "3P_SHARE": a3 / fga if fga > 0 else np.nan,
        "OREB_PER_MISS": float(x["OREB"].sum()) / misses if misses > 0 else np.nan,
        "AST_PER_MAKE": float(x["AST"].sum()) / fgm if fgm > 0 else np.nan,
    }


def _modifier_from_ratio(stat: str, ratio: float) -> float:
    """Stat-specific shrinkage: opponent context is a correction, not a second sample."""
    if stat in {"3P_PCT", "2P_PCT"}:
        ratio = float(np.clip(ratio, 0.88, 1.12))
        return float(np.clip(ratio ** 0.18, 0.96, 1.04))
    if stat in {"3PA", "2PA"}:
        # Player-prop compatibility only; Team Markets use 3P_SHARE instead.
        ratio = float(np.clip(ratio, 0.78, 1.22))
        return float(np.clip(ratio ** 0.30, 0.91, 1.09))
    if stat == "FGA_LIVE":
        ratio = float(np.clip(ratio, 0.82, 1.18))
        return float(np.clip(ratio ** 0.20, 0.95, 1.05))
    if stat == "3P_SHARE":
        ratio = float(np.clip(ratio, 0.82, 1.18))
        return float(np.clip(ratio ** 0.35, 0.94, 1.06))
    if stat == "OREB_PER_MISS":
        ratio = float(np.clip(ratio, 0.78, 1.22))
        return float(np.clip(ratio ** 0.35, 0.90, 1.10))
    if stat == "AST_PER_MAKE":
        ratio = float(np.clip(ratio, 0.80, 1.20))
        return float(np.clip(ratio ** 0.30, 0.92, 1.08))
    if stat == "BLK":
        # BLK is opponent offensive susceptibility: how often this offense is
        # blocked by its opponents. This is deliberately more meaningful than
        # a 2PA proxy, but still only a correction to the defense's own BLK rate.
        ratio = float(np.clip(ratio, 0.72, 1.28))
        return float(np.clip(ratio ** 0.40, 0.90, 1.10))
    ratio = float(np.clip(ratio, 0.75, 1.25))
    return float(np.clip(ratio ** 0.35, 0.88, 1.12))


def opponent_allowed_profile(
    league_team_logs: pd.DataFrame,
    opponent_abbr: str,
    exclude_team_abbr: str | None = None,
):
    """Opponent-allowed profile, optionally excluding the current H2H opponent.

    exclude_team_abbr is used by Team Markets to keep opponent-allowed evidence
    DISJOINT from the explicit H2H sample. Player Props can continue using the
    default full opponent history.
    """
    all_rows = league_team_logs.copy()
    opp_rows = all_rows[
        all_rows["OPP_ABBR"].astype(str).str.upper() == str(opponent_abbr).upper()
    ].copy()
    if exclude_team_abbr:
        opp_rows = opp_rows[
            ~opp_rows["TEAM_ABBR"].astype(str).str.upper().eq(str(exclude_team_abbr).upper())
        ].copy()

    lg = _rates(all_rows)
    opp = _rates(opp_rows)

    keys = [
        "PTS", "REB", "AST", "3PA", "2PA", "FTA", "TOV", "OREB", "PF", "BLK",
        "3P_PCT", "2P_PCT", "FGA_LIVE", "3P_SHARE", "OREB_PER_MISS", "AST_PER_MAKE",
    ]
    rows, ratios, auto = [], {}, {}
    for k in keys:
        l = lg.get(k, np.nan)
        o = opp.get(k, np.nan)
        ratio = (o / l) if np.isfinite(l) and l > 0 and np.isfinite(o) else 1.0
        mod = _modifier_from_ratio(k, ratio)
        ratios[k] = float(ratio)
        auto[k] = mod
        rows.append({
            "Stat": k,
            "Opponent allowed": o,
            "League avg": l,
            "Raw ratio": ratio,
            "Applied overall modifier": mod,
            "Opponent games": opp.get("games", 0),
            "H2H excluded team": str(exclude_team_abbr or "—"),
        })
    return {
        "ratios": ratios,
        "modifiers": auto,
        "rates": {"league": lg, "opponent": opp},
        "audit": pd.DataFrame(rows),
    }


def position_environment(
    player_df: pd.DataFrame,
    opponent_abbr: str,
    position_group: str,
    exclude_team_abbr: str | None = None,
):
    """Same-position opponent environment for Player Props.

    ``exclude_team_abbr`` makes the opponent-by-position sample leave-pair-out:
    rows from the focal player's current team against this opponent are removed
    from the opponent-specific sample. The league position baseline remains
    untouched. This keeps the generic opponent layer disjoint from the explicit
    player/opponent H2H residual.
    """
    x = player_df[
        player_df["POSITION_GROUP"].astype(str) == str(position_group)
    ].copy()
    vs = x[
        x["OPP_ABBR"].astype(str).str.upper() == str(opponent_abbr).upper()
    ].copy()
    if exclude_team_abbr and "TEAM_ABBR" in vs.columns:
        vs = vs[
            ~vs["TEAM_ABBR"].astype(str).str.upper().eq(str(exclude_team_abbr).upper())
        ].copy()
    return _position_summary(vs), _position_summary(x)


def _position_summary(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    mins = float(pd.to_numeric(df["MIN"], errors="coerce").fillna(0).sum())
    if mins <= 0:
        return {}
    out = {"sample_min": mins, "player_game_rows": int(len(df))}
    for source, key in [
        ("PTS", "PTS"), ("REB", "REB"), ("AST", "AST"),
        ("FG3A", "3PA"), ("FTA", "FTA"), ("FGA", "FGA"),
        ("OREB", "OREB"), ("TOV", "TOV")
    ]:
        out[key] = float(pd.to_numeric(df[source], errors="coerce").fillna(0).sum()) / mins * 36.0

    a3 = float(pd.to_numeric(df["FG3A"], errors="coerce").fillna(0).sum())
    m3 = float(pd.to_numeric(df["FG3M"], errors="coerce").fillna(0).sum())
    fga = float(pd.to_numeric(df["FGA"], errors="coerce").fillna(0).sum())
    fgm = float(pd.to_numeric(df["FGM"], errors="coerce").fillna(0).sum())
    a2 = max(fga - a3, 0.0)
    m2 = max(fgm - m3, 0.0)
    out["2PA"] = a2 / mins * 36.0
    out["3P_PCT"] = m3 / a3 if a3 > 0 else np.nan
    out["2P_PCT"] = m2 / a2 if a2 > 0 else np.nan
    out["3P_ATT"] = a3
    out["2P_ATT"] = a2
    return out


def combine_overall_and_position(overall_ratio, position_ratio, position_sample_min=0.0):
    """Overall defense is base; position changes only the relative deviation."""
    overall_ratio = float(np.clip(overall_ratio or 1.0, 0.75, 1.25))
    if position_ratio is None or not np.isfinite(position_ratio):
        return float(np.clip(overall_ratio ** 0.30, 0.90, 1.10)), 0.0

    position_ratio = float(np.clip(position_ratio, 0.78, 1.22))
    conf = float(np.clip(position_sample_min / (position_sample_min + 900.0), 0.0, 1.0))
    relative_position = float(np.clip(position_ratio / overall_ratio, 0.84, 1.16))
    mod = (overall_ratio ** 0.30) * (relative_position ** (0.18 * conf))
    return float(np.clip(mod, 0.88, 1.12)), conf


def combine_efficiency_overall_and_position(
    overall_ratio,
    position_ratio,
    position_attempts=0.0,
):
    """Very conservative shooting-efficiency opponent adjustment."""
    overall_ratio = float(np.clip(overall_ratio or 1.0, 0.88, 1.12))
    if position_ratio is None or not np.isfinite(position_ratio):
        return float(np.clip(overall_ratio ** 0.16, 0.97, 1.03)), 0.0
    position_ratio = float(np.clip(position_ratio, 0.88, 1.12))
    conf = float(np.clip(position_attempts / (position_attempts + 180.0), 0.0, 1.0))
    relative = float(np.clip(position_ratio / overall_ratio, 0.92, 1.08))
    mod = (overall_ratio ** 0.16) * (relative ** (0.10 * conf))
    return float(np.clip(mod, 0.965, 1.035)), conf


def player_matchup_modifiers(overall_profile, position_vs_opp=None, position_league=None):
    mapping = {"PTS": "PTS", "REB": "REB", "AST": "AST", "3PA": "3PA", "2PA": "2PA", "FTA": "FTA"}
    rows, final = [], {}
    for stat, poskey in mapping.items():
        overall_ratio = overall_profile.get("ratios", {}).get(stat, 1.0)
        pv = (position_vs_opp or {}).get(poskey, np.nan)
        pl = (position_league or {}).get(poskey, np.nan)
        pos_ratio = (pv / pl) if np.isfinite(pv) and np.isfinite(pl) and pl > 0 else None
        sample_min = float((position_vs_opp or {}).get("sample_min", 0.0))
        mod, conf = combine_overall_and_position(overall_ratio, pos_ratio, sample_min)
        final[stat] = mod
        rows.append({
            "Stat": stat,
            "Overall raw ratio": overall_ratio,
            "Position vs opp": pv,
            "Position league": pl,
            "Position raw ratio": pos_ratio,
            "Position sample MIN": sample_min,
            "Position confidence": conf,
            "Final auto modifier": mod,
        })

    for stat, att_key in [("3P_PCT", "3P_ATT"), ("2P_PCT", "2P_ATT")]:
        overall_ratio = overall_profile.get("ratios", {}).get(stat, 1.0)
        pv = (position_vs_opp or {}).get(stat, np.nan)
        pl = (position_league or {}).get(stat, np.nan)
        pos_ratio = (pv / pl) if np.isfinite(pv) and np.isfinite(pl) and pl > 0 else None
        attempts = float((position_vs_opp or {}).get(att_key, 0.0))
        mod, conf = combine_efficiency_overall_and_position(overall_ratio, pos_ratio, attempts)
        final[stat] = mod
        rows.append({
            "Stat": stat,
            "Overall raw ratio": overall_ratio,
            "Position vs opp": pv,
            "Position league": pl,
            "Position raw ratio": pos_ratio,
            "Position attempts": attempts,
            "Position confidence": conf,
            "Final auto modifier": mod,
        })
    return final, pd.DataFrame(rows)



def _player_game_counts(row: pd.Series) -> dict:
    fga = float(pd.to_numeric(pd.Series([row.get("FGA", 0)]), errors="coerce").fillna(0.0).iloc[0])
    a3 = float(pd.to_numeric(pd.Series([row.get("FG3A", 0)]), errors="coerce").fillna(0.0).iloc[0])
    return {
        "2PA": max(fga - a3, 0.0),
        "3PA": max(a3, 0.0),
        "FTA": float(pd.to_numeric(pd.Series([row.get("FTA", 0)]), errors="coerce").fillna(0.0).iloc[0]),
        "REB": float(pd.to_numeric(pd.Series([row.get("REB", 0)]), errors="coerce").fillna(0.0).iloc[0]),
        "AST": float(pd.to_numeric(pd.Series([row.get("AST", 0)]), errors="coerce").fillna(0.0).iloc[0]),
    }



_RESIDUAL_H2H_STATS = ("2PA", "3PA", "FTA", "REB", "AST")


def _historical_generic_matchup_modifiers(
    player_logs: pd.DataFrame,
    team_logs: pd.DataFrame,
    cutoff_date,
    team_abbr: str,
    opponent_abbr: str,
    position_group: str | None,
):
    """Pregame generic opponent modifiers using only information before cutoff.

    The current team is removed from the opponent-specific overall and
    position samples, so this is a leave-pair-out generic environment.  It is
    deliberately the same opponent architecture used by live Player Props:
    overall opponent allowance + relative opponent-by-position deviation.
    """
    cutoff = pd.Timestamp(cutoff_date)
    p = player_logs.copy()
    t = team_logs.copy()
    if "GAME_DATE" not in p.columns or "GAME_DATE" not in t.columns:
        return {k: 1.0 for k in ("PTS", "REB", "AST", "3PA", "2PA", "FTA", "3P_PCT", "2P_PCT")}

    p_dates = pd.to_datetime(p["GAME_DATE"], errors="coerce")
    t_dates = pd.to_datetime(t["GAME_DATE"], errors="coerce")
    p_hist = p[p_dates < cutoff].copy()
    t_hist = t[t_dates < cutoff].copy()

    if t_hist.empty:
        overall = {"ratios": {}, "modifiers": {}, "rates": {}, "audit": pd.DataFrame()}
    else:
        overall = opponent_allowed_profile(
            t_hist,
            opponent_abbr,
            exclude_team_abbr=team_abbr,
        )

    pvo, plg = {}, {}
    if position_group is not None and str(position_group).strip() and not p_hist.empty:
        pvo, plg = position_environment(
            p_hist,
            opponent_abbr,
            str(position_group),
            exclude_team_abbr=team_abbr,
        )

    mods, _ = player_matchup_modifiers(overall, pvo, plg)
    return mods


def _simple_nonpair_player_rates(
    player_history: pd.DataFrame,
    opponent_abbr: str,
) -> dict:
    """Player per-minute opportunity rates before a historical H2H game.

    Pair rows are excluded.  This is used only to construct the historical
    no-H2H expectation against which matchup residuals are measured; the live
    player baseline still comes from the full v2.17.1 adaptive role-state.
    """
    if player_history is None or player_history.empty:
        return {k: np.nan for k in _RESIDUAL_H2H_STATS}
    x = player_history[
        ~player_history["OPP_ABBR"].astype(str).str.upper().eq(str(opponent_abbr).upper())
    ].copy()
    mins = pd.to_numeric(x.get("MIN", 0), errors="coerce").fillna(0.0)
    x = x[mins > 0].copy()
    if x.empty:
        return {k: np.nan for k in _RESIDUAL_H2H_STATS}
    mins = float(pd.to_numeric(x["MIN"], errors="coerce").fillna(0.0).sum())
    if mins <= 0:
        return {k: np.nan for k in _RESIDUAL_H2H_STATS}

    fga = float(pd.to_numeric(x.get("FGA", 0), errors="coerce").fillna(0.0).sum())
    a3 = float(pd.to_numeric(x.get("FG3A", 0), errors="coerce").fillna(0.0).sum())
    fta = float(pd.to_numeric(x["FTA"], errors="coerce").fillna(0.0).sum()) if "FTA" in x.columns else 0.0
    return {
        "2PA": max(fga - a3, 0.0) / mins,
        "3PA": a3 / mins,
        "FTA": fta / mins,
        "REB": float(pd.to_numeric(x.get("REB", 0), errors="coerce").fillna(0.0).sum()) / mins,
        "AST": float(pd.to_numeric(x.get("AST", 0), errors="coerce").fillna(0.0).sum()) / mins,
    }


def player_h2h_residual_history(
    league_player_logs: pd.DataFrame,
    league_team_logs: pd.DataFrame,
    player_id,
    team_abbr: str,
    opponent_abbr: str,
    position_group: str | None,
) -> pd.DataFrame:
    """Build historical *expected* H2H rates before each H2H occurred.

    For every same-season player/opponent game, the no-H2H expectation is:

        historical non-pair player rate
        × historical leave-pair-out generic opponent modifier.

    Both components use only rows strictly before that H2H date.  This avoids
    using today's opponent profile to explain an old game and avoids allowing
    the focal pair's own H2H rows to define the generic opponent environment.
    """
    if league_player_logs is None or league_player_logs.empty:
        return pd.DataFrame()
    p = league_player_logs.copy()
    if "GAME_DATE" not in p.columns:
        return pd.DataFrame()
    p["GAME_DATE"] = pd.to_datetime(p["GAME_DATE"], errors="coerce")
    focal = p[p["PLAYER_ID"].astype(str).eq(str(player_id))].copy()
    focal = focal.dropna(subset=["GAME_DATE"]).sort_values(["GAME_DATE", "GAME_ID"] if "GAME_ID" in focal.columns else ["GAME_DATE"])
    h = focal[
        focal["OPP_ABBR"].astype(str).str.upper().eq(str(opponent_abbr).upper())
    ].copy()
    if h.empty:
        return pd.DataFrame()

    rows = []
    for _, row in h.iterrows():
        cutoff = pd.Timestamp(row["GAME_DATE"])
        prior = focal[focal["GAME_DATE"] < cutoff].copy()
        base = _simple_nonpair_player_rates(prior, opponent_abbr)
        hist_team = str(row.get("TEAM_ABBR", team_abbr) or team_abbr).upper()
        mods = _historical_generic_matchup_modifiers(
            p,
            league_team_logs,
            cutoff,
            hist_team,
            opponent_abbr,
            position_group,
        )
        counts = _player_game_counts(row)
        mins = float(pd.to_numeric(pd.Series([row.get("MIN", 0)]), errors="coerce").fillna(0.0).iloc[0])
        rec = {
            "GAME_ID": str(row.get("GAME_ID", "")),
            "GAME_DATE": cutoff,
            "TEAM_ABBR": hist_team,
            "MIN": mins,
        }
        for st in _RESIDUAL_H2H_STATS:
            b = float(base.get(st, np.nan))
            om = float(mods.get(st, 1.0))
            exp_rate = b * om if np.isfinite(b) and b > 0 and np.isfinite(om) and om > 0 else np.nan
            rec[f"{st}_observed"] = float(counts[st])
            rec[f"{st}_historical_base_pm"] = b
            rec[f"{st}_historical_opp_mod"] = om
            rec[f"{st}_expected_pm"] = exp_rate
            rec[f"{st}_expected_events"] = exp_rate * mins if np.isfinite(exp_rate) and mins > 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def fit_player_h2h_residual_calibration(
    player_logs: pd.DataFrame,
    team_logs: pd.DataFrame,
):
    """Walk-forward calibration for residualized player H2H.

    Model:
        count ~ Poisson(minutes × baseline × generic_opponent × residual_pair)

    Pair-specific residuals receive a Gamma prior with mean 1. The prior mass
    ``K`` is measured in expected events, not minutes, because baseline rates
    differ by player/stat/opponent. Minute-relevance decay ``tau`` is learned
    jointly from next-H2H predictive likelihood; ``tau=inf`` means historical
    H2H minutes are not downweighted merely because they differ from today's
    minutes.

    v2.18.2 removes the old binary held-out activation gate. A finite
    Gamma-Poisson H2H candidate is always retained when calibration evidence is
    sufficient, then continuously averaged toward the no-H2H model using the
    later blocked holdout predictive likelihood. This preserves matchup
    information without allowing a weak pair sample to dominate.

    The walk-forward generic opponent context is computed from cumulative
    sufficient statistics, so calibration is chronological without the very
    expensive dataframe re-filtering that would otherwise occur for every
    player-game.
    """
    neutral = {
        st: {
            "active": False,
            "prior_events_k": np.inf,
            "minute_tau": np.inf,
            "predictive_model_weight": 0.0,
        }
        for st in _RESIDUAL_H2H_STATS
    }
    if player_logs is None or player_logs.empty or team_logs is None or team_logs.empty:
        return neutral, pd.DataFrame()

    required = {"PLAYER_ID", "TEAM_ABBR", "OPP_ABBR", "GAME_DATE", "MIN"}
    if not required.issubset(set(player_logs.columns)):
        return neutral, pd.DataFrame([{
            "Active": False,
            "Reason": "missing player/team/opponent/minute columns",
        }])

    x = player_logs.copy()
    x["GAME_DATE"] = pd.to_datetime(x["GAME_DATE"], errors="coerce")
    if "GAME_ID" not in x.columns:
        x["GAME_ID"] = np.arange(len(x)).astype(str)
    x = x.dropna(subset=["GAME_DATE"]).sort_values(
        ["GAME_DATE", "GAME_ID", "PLAYER_ID"]
    ).reset_index(drop=True)

    t = team_logs.copy()
    t["GAME_DATE"] = pd.to_datetime(t["GAME_DATE"], errors="coerce")
    t = t.dropna(subset=["GAME_DATE"]).sort_values(
        ["GAME_DATE", "GAME_ID"] if "GAME_ID" in t.columns else ["GAME_DATE"]
    ).reset_index(drop=True)

    # Player prior state used for the no-H2H baseline.
    total_exp, total_counts = {}, {}
    pair_exp, pair_counts = {}, {}
    pair_history = {}
    events = {st: [] for st in _RESIDUAL_H2H_STATS}

    # Cumulative TEAM sufficient statistics for overall opponent allowance.
    # Values are [poss, 2PA, 3PA, FTA, REB, AST].
    team_league = np.zeros(6, dtype=float)
    team_opp = {}
    team_pair = {}

    # Cumulative PLAYER sufficient statistics for opponent-by-position.
    # Values are [minutes, 2PA, 3PA, FTA, REB, AST].
    pos_league = {}
    pos_opp = {}
    pos_pair = {}

    idx = {"2PA": 1, "3PA": 2, "FTA": 3, "REB": 4, "AST": 5}

    def _arr_get(d, key):
        v = d.get(key)
        return np.zeros(6, dtype=float) if v is None else v

    def _team_row_vec(row):
        fga = float(pd.to_numeric(pd.Series([row.get("FGA", 0)]), errors="coerce").fillna(0.0).iloc[0])
        a3 = float(pd.to_numeric(pd.Series([row.get("FG3A", 0)]), errors="coerce").fillna(0.0).iloc[0])
        oreb = float(pd.to_numeric(pd.Series([row.get("OREB", 0)]), errors="coerce").fillna(0.0).iloc[0])
        tov = float(pd.to_numeric(pd.Series([row.get("TOV", 0)]), errors="coerce").fillna(0.0).iloc[0])
        fta = float(pd.to_numeric(pd.Series([row.get("FTA", 0)]), errors="coerce").fillna(0.0).iloc[0])
        poss = max(fga - oreb + tov + 0.44 * fta, 0.0)
        reb = float(pd.to_numeric(pd.Series([row.get("REB", 0)]), errors="coerce").fillna(0.0).iloc[0])
        ast = float(pd.to_numeric(pd.Series([row.get("AST", 0)]), errors="coerce").fillna(0.0).iloc[0])
        return np.asarray([poss, max(fga - a3, 0.0), max(a3, 0.0), max(fta, 0.0), reb, ast], float)

    def _player_row_vec(row):
        mins = float(pd.to_numeric(pd.Series([row.get("MIN", 0)]), errors="coerce").fillna(0.0).iloc[0])
        c = _player_game_counts(row)
        return np.asarray([mins, c["2PA"], c["3PA"], c["FTA"], c["REB"], c["AST"]], float)

    def _generic_modifiers(team, opp, pos):
        out = {}
        opp_vec = _arr_get(team_opp, opp) - _arr_get(team_pair, (team, opp))
        for st in _RESIDUAL_H2H_STATS:
            j = idx[st]
            lg_rate = (
                team_league[j] / team_league[0]
                if team_league[0] > 0 else np.nan
            )
            opp_rate = (
                opp_vec[j] / opp_vec[0]
                if opp_vec[0] > 0 else np.nan
            )
            overall_ratio = (
                opp_rate / lg_rate
                if np.isfinite(opp_rate) and np.isfinite(lg_rate) and lg_rate > 0
                else 1.0
            )

            position_ratio = None
            sample_min = 0.0
            if pos:
                lgp = _arr_get(pos_league, pos)
                oppp = (
                    _arr_get(pos_opp, (pos, opp))
                    - _arr_get(pos_pair, (pos, team, opp))
                )
                sample_min = max(float(oppp[0]), 0.0)
                lgp_rate = lgp[j] / lgp[0] if lgp[0] > 0 else np.nan
                oppp_rate = oppp[j] / oppp[0] if oppp[0] > 0 else np.nan
                if np.isfinite(lgp_rate) and lgp_rate > 0 and np.isfinite(oppp_rate):
                    position_ratio = oppp_rate / lgp_rate

            mod, _ = combine_overall_and_position(
                overall_ratio,
                position_ratio,
                position_sample_min=sample_min,
            )
            out[st] = float(mod)
        return out

    all_dates = sorted(
        set(x["GAME_DATE"].dropna().tolist()) | set(t["GAME_DATE"].dropna().tolist())
    )

    for date in all_dates:
        px = x[x["GAME_DATE"].eq(date)]
        pending = []

        # Predict every player row on this date using only earlier dates.
        for _, row in px.iterrows():
            pid = str(row.get("PLAYER_ID"))
            team = str(row.get("TEAM_ABBR", "")).upper()
            opp = str(row.get("OPP_ABBR", "")).upper()
            pos = str(row.get("POSITION_GROUP", "") or "")
            mins = float(pd.to_numeric(
                pd.Series([row.get("MIN", 0)]), errors="coerce"
            ).fillna(0.0).iloc[0])
            if not np.isfinite(mins) or mins < 4.0 or not team or not opp:
                continue

            counts = _player_game_counts(row)
            pk = (pid, opp)
            texp = float(total_exp.get(pid, 0.0))
            hexp = float(pair_exp.get(pk, 0.0))
            nonexp = max(texp - hexp, 0.0)
            prior_pair = pair_history.get(pk, [])
            mods = _generic_modifiers(team, opp, pos)

            current_expected_rates = {}
            for st in _RESIDUAL_H2H_STATS:
                tc = float(total_counts.get((pid, st), 0.0))
                hc = float(pair_counts.get((pid, opp, st), 0.0))
                nonc = max(tc - hc, 0.0)
                b = nonc / nonexp if nonexp > 0 else np.nan
                om = float(mods.get(st, 1.0))
                current_expected_rates[st] = (
                    b * om
                    if np.isfinite(b) and b > 1e-9 and np.isfinite(om) and om > 0
                    else np.nan
                )

            if prior_pair and nonexp > 0:
                for st in _RESIDUAL_H2H_STATS:
                    er = current_expected_rates[st]
                    if not (np.isfinite(er) and er > 0):
                        continue
                    prior_entries = [
                        (
                            float(z["MIN"]),
                            float(z[f"{st}_count"]),
                            float(z[f"{st}_expected_events"]),
                        )
                        for z in prior_pair
                        if np.isfinite(z.get(f"{st}_expected_events", np.nan))
                        and float(z.get(f"{st}_expected_events", 0.0)) > 0
                    ]
                    if not prior_entries:
                        continue
                    events[st].append({
                        "current_minutes": mins,
                        "current_expected_rate": er,
                        "actual": float(counts[st]),
                        "prior": prior_entries,
                    })

            pending.append({
                "row": row,
                "pid": pid,
                "team": team,
                "opp": opp,
                "pos": pos,
                "mins": mins,
                "counts": counts,
                "expected_rates": current_expected_rates,
            })

        # Only after every prediction for the date is frozen do today's rows
        # become historical information.
        for item in pending:
            row = item["row"]
            pid, team, opp, pos = item["pid"], item["team"], item["opp"], item["pos"]
            mins, counts = item["mins"], item["counts"]
            erates = item["expected_rates"]
            pk = (pid, opp)

            hist_entry = {"MIN": mins}
            for st in _RESIDUAL_H2H_STATS:
                er = erates[st]
                hist_entry[f"{st}_count"] = float(counts[st])
                hist_entry[f"{st}_expected_events"] = (
                    er * mins if np.isfinite(er) and er > 0 else np.nan
                )
            pair_history.setdefault(pk, []).append(hist_entry)

            total_exp[pid] = float(total_exp.get(pid, 0.0)) + mins
            pair_exp[pk] = float(pair_exp.get(pk, 0.0)) + mins
            for st in _RESIDUAL_H2H_STATS:
                c = float(counts[st])
                total_counts[(pid, st)] = float(
                    total_counts.get((pid, st), 0.0)
                ) + c
                pair_counts[(pid, opp, st)] = float(
                    pair_counts.get((pid, opp, st), 0.0)
                ) + c

            pv = _player_row_vec(row)
            if pos:
                pos_league[pos] = _arr_get(pos_league, pos) + pv
                pos_opp[(pos, opp)] = _arr_get(pos_opp, (pos, opp)) + pv
                pos_pair[(pos, team, opp)] = (
                    _arr_get(pos_pair, (pos, team, opp)) + pv
                )

        # Update team overall opponent allowance after all player predictions
        # for the date, preserving the same strict pregame cutoff.
        tx = t[t["GAME_DATE"].eq(date)]
        for _, row in tx.iterrows():
            team = str(row.get("TEAM_ABBR", "")).upper()
            opp = str(row.get("OPP_ABBR", "")).upper()
            if not team or not opp:
                continue
            tv = _team_row_vec(row)
            team_league += tv
            team_opp[opp] = _arr_get(team_opp, opp) + tv
            team_pair[(team, opp)] = _arr_get(team_pair, (team, opp)) + tv

    result, audit_rows = {}, []
    tau_grid = list(np.logspace(np.log10(4.0), np.log10(40.0), 12)) + [np.inf]

    for st in _RESIDUAL_H2H_STATS:
        ev = events[st]
        if len(ev) < 40:
            result[st] = dict(neutral[st])
            audit_rows.append({
                "Stat": st,
                "Active": False,
                "Calibration events": len(ev),
                "Prior residual events K": np.inf,
                "Minute relevance tau": np.inf,
                "No-H2H NLL": np.nan,
                "Best residual H2H NLL": np.nan,
                "NLL gain": 0.0,
                "Reason": "insufficient repeat-matchup residual events",
            })
            continue

        # Chronological blocked validation: select the finite K/tau candidate
        # on the earlier 70% of repeat-matchup prediction events, then use the
        # later 30% only to determine a continuous predictive model weight.
        # The later block therefore shrinks weak H2H evidence without a binary
        # delete/keep decision.
        split = int(np.floor(0.70 * len(ev)))
        split = min(max(split, 30), len(ev) - 10)
        train_ev = ev[:split]
        hold_ev = ev[split:]

        expected_masses = [
            sum(z[2] for z in e["prior"] if np.isfinite(z[2]) and z[2] > 0)
            for e in train_ev
        ]
        pos_mass = np.asarray([v for v in expected_masses if v > 0], float)
        med_e = max(
            float(np.median(pos_mass)) if len(pos_mass) else 4.0,
            0.5,
        )
        finite_k = np.asarray(
            list(med_e * np.logspace(-1.0, 1.8, 30)),
            dtype=float,
        )

        def _arrays(es):
            cm = np.asarray([float(e["current_minutes"]) for e in es], dtype=float)
            er = np.asarray([float(e["current_expected_rate"]) for e in es], dtype=float)
            yy = np.asarray([float(e["actual"]) for e in es], dtype=float)
            lf = np.asarray([math.lgamma(y + 1.0) for y in yy], dtype=float)
            base = np.clip(cm * er, 1e-9, None)
            return cm, yy, lf, base

        def _base_nll(es):
            _, yy, lf, base = _arrays(es)
            return float(np.sum(base - yy * np.log(base) + lf))

        def _score(es, k, tau):
            cm, yy, lf, base = _arrays(es)
            obs_eff = np.zeros(len(es), dtype=float)
            exp_eff = np.zeros(len(es), dtype=float)
            for i, e in enumerate(es):
                prior = e["prior"]
                if np.isinf(tau):
                    obs_eff[i] = sum(float(z[1]) for z in prior)
                    exp_eff[i] = sum(float(z[2]) for z in prior)
                else:
                    tt = max(float(tau), 1e-6)
                    for pm, pc, pe in prior:
                        w = float(np.exp(-abs(float(pm) - cm[i]) / tt))
                        obs_eff[i] += float(pc) * w
                        exp_eff[i] += float(pe) * w
            if np.isinf(k):
                residual = np.ones(len(es), dtype=float)
            else:
                residual = (obs_eff + float(k)) / np.maximum(
                    exp_eff + float(k), 1e-9
                )
            lam = np.clip(base * residual, 1e-9, None)
            return float(np.sum(lam - yy * np.log(lam) + lf))

        train_base_nll = _base_nll(train_ev)
        # Select the best *finite* Gamma-Poisson pair model on the earlier
        # block. The no-H2H model is evaluated separately rather than acting as
        # a hard switch that can delete matchup information.
        best_train = (np.inf, np.nan, np.nan)

        # Precompute weighted residual evidence once per tau for the training
        # block, then evaluate the K grid vectorially.
        cm_tr, y_tr, lf_tr, base_tr = _arrays(train_ev)
        for tau in tau_grid:
            obs_eff = np.zeros(len(train_ev), dtype=float)
            exp_eff = np.zeros(len(train_ev), dtype=float)
            for i, e in enumerate(train_ev):
                prior = e["prior"]
                if np.isinf(tau):
                    obs_eff[i] = sum(float(z[1]) for z in prior)
                    exp_eff[i] = sum(float(z[2]) for z in prior)
                else:
                    tt = max(float(tau), 1e-6)
                    for pm, pc, pe in prior:
                        w = float(np.exp(-abs(float(pm) - cm_tr[i]) / tt))
                        obs_eff[i] += float(pc) * w
                        exp_eff[i] += float(pe) * w

            for k in finite_k:
                residual = (obs_eff + k) / np.maximum(exp_eff + k, 1e-9)
                lam = np.clip(base_tr * residual, 1e-9, None)
                score = float(np.sum(lam - y_tr * np.log(lam) + lf_tr))
                if score < best_train[0]:
                    best_train = (score, float(k), float(tau))

        train_best_nll, candidate_k, candidate_tau = best_train
        hold_base_nll = _base_nll(hold_ev)
        hold_candidate_nll = _score(hold_ev, candidate_k, candidate_tau)

        # Continuous predictive model averaging. With equal prior model odds,
        # the later-block predictive likelihood gives the H2H model weight:
        #   w = L_h2h / (L_h2h + L_noh2h)
        #     = sigmoid(NLL_noh2h - NLL_h2h).
        # This is deliberately continuous: a slightly worse holdout shrinks the
        # H2H layer strongly but does not hard-delete it.
        heldout_gain = float(hold_base_nll - hold_candidate_nll)
        z = float(np.clip(heldout_gain, -60.0, 60.0))
        predictive_weight = float(1.0 / (1.0 + np.exp(-z)))
        active = bool(np.isfinite(candidate_k) and predictive_weight > 0.0)

        result[st] = {
            "active": active,
            "prior_events_k": float(candidate_k),
            "minute_tau": float(candidate_tau),
            "predictive_model_weight": predictive_weight,
        }
        audit_rows.append({
            "Stat": st,
            "Active": active,
            "Calibration events": len(ev),
            "Train events": len(train_ev),
            "Holdout events": len(hold_ev),
            "Prior residual events K": float(candidate_k),
            "Minute relevance tau": float(candidate_tau),
            "Predictive H2H model weight": predictive_weight,
            "Train no-H2H NLL": float(train_base_nll),
            "Train best residual H2H NLL": float(train_best_nll),
            "Held-out no-H2H NLL": float(hold_base_nll),
            "Held-out residual H2H NLL": float(hold_candidate_nll),
            "Held-out NLL gain": heldout_gain,
            # Compatibility summary aliases now report the actual finite H2H
            # candidate even when the holdout prefers the no-H2H model.
            "No-H2H NLL": float(hold_base_nll),
            "Best residual H2H NLL": float(hold_candidate_nll),
            "NLL gain": heldout_gain,
            "Reason": "continuous predictive-likelihood shrinkage; no hard H2H drop",
        })

    return result, pd.DataFrame(audit_rows)


def fit_player_h2h_prior_minutes(player_logs: pd.DataFrame):
    """Learn stat-specific H2H pooling strength from repeat matchups.

    A Gamma-Poisson empirical-Bayes view is used.  For a live player/opponent
    pairing, the non-H2H player rate is the prior mean and previous H2H minutes
    are additional exposure.  The prior-equivalent minutes K are selected from
    historical *next-game* predictive likelihood across repeated player/opponent
    matchups.  K=inf (ignore H2H) is always a candidate, so the data can decide
    that a stat has no repeat-matchup value.

    This replaces the old universal 5% cap.  Rotation/minute similarity is NOT
    fitted here; it is factual live relevance and reduces effective H2H exposure
    downstream.
    """
    neutral = {k: np.inf for k in ("2PA", "3PA", "REB", "AST")}
    if player_logs is None or player_logs.empty:
        return neutral, pd.DataFrame()
    required = {"PLAYER_ID", "OPP_ABBR", "GAME_DATE", "MIN"}
    if not required.issubset(set(player_logs.columns)):
        return neutral, pd.DataFrame([{"Active": False, "Reason": "missing player/opponent/minute columns"}])

    x = player_logs.copy()
    x["GAME_DATE"] = pd.to_datetime(x["GAME_DATE"], errors="coerce")
    if "GAME_ID" not in x.columns:
        x["GAME_ID"] = np.arange(len(x)).astype(str)
    x = x.dropna(subset=["GAME_DATE"]).sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).reset_index(drop=True)

    stats = ("2PA", "3PA", "REB", "AST")
    events = {st: [] for st in stats}
    total_exp = {}
    total_counts = {}
    pair_exp = {}
    pair_counts = {}

    for _, row in x.iterrows():
        pid = str(row.get("PLAYER_ID"))
        opp = str(row.get("OPP_ABBR", "")).upper()
        mins = float(pd.to_numeric(pd.Series([row.get("MIN", 0)]), errors="coerce").fillna(0.0).iloc[0])
        if not np.isfinite(mins) or mins < 4.0 or not opp:
            continue
        counts = _player_game_counts(row)
        pk = (pid, opp)
        texp = float(total_exp.get(pid, 0.0))
        hexp = float(pair_exp.get(pk, 0.0))
        nonexp = max(texp - hexp, 0.0)

        # Only prior H2H can predict the current H2H game; no future leakage.
        if hexp > 0 and nonexp > 0:
            for st in stats:
                tc = float(total_counts.get((pid, st), 0.0))
                hc = float(pair_counts.get((pid, opp, st), 0.0))
                nonc = max(tc - hc, 0.0)
                b = nonc / nonexp if nonexp > 0 else np.nan
                c = float(counts[st])
                if np.isfinite(b) and b > 1e-6 and np.isfinite(c):
                    events[st].append((b, hc, hexp, c, mins))

        total_exp[pid] = texp + mins
        pair_exp[pk] = hexp + mins
        for st in stats:
            c = float(counts[st])
            total_counts[(pid, st)] = float(total_counts.get((pid, st), 0.0)) + c
            pair_counts[(pid, opp, st)] = float(pair_counts.get((pid, opp, st), 0.0)) + c

    result, rows = {}, []
    for st in stats:
        ev = events[st]
        if len(ev) < 40:
            result[st] = np.inf
            rows.append({"Stat": st, "Active": False, "Calibration events": len(ev), "Prior minutes K": np.inf, "Reason": "insufficient repeat-matchup events"})
            continue

        b = np.asarray([z[0] for z in ev], float)
        hc = np.asarray([z[1] for z in ev], float)
        he = np.asarray([z[2] for z in ev], float)
        actual = np.asarray([z[3] for z in ev], float)
        mins = np.asarray([z[4] for z in ev], float)
        logfact = np.asarray([math.lgamma(float(c) + 1.0) for c in actual], float)

        # Data-scaled grid in equivalent player-minutes; inf = no H2H layer.
        med_h = max(float(np.median(he[he > 0])) if np.any(he > 0) else 30.0, 10.0)
        finite_grid = med_h * np.logspace(-1.0, 1.6, 28)
        grid = list(finite_grid) + [np.inf]

        def nll(k):
            if np.isinf(k):
                rate = b
            else:
                rate = (hc + float(k) * b) / np.maximum(he + float(k), 1e-9)
            lam = np.clip(mins * rate, 1e-9, None)
            return float(np.sum(lam - actual * np.log(lam) + logfact))

        scores = np.asarray([nll(k) for k in grid], float)
        j = int(np.argmin(scores))
        best_k = float(grid[j])
        base_nll = float(scores[-1])
        best_nll = float(scores[j])
        active = bool(np.isfinite(best_k) and best_nll < base_nll)
        if not active:
            best_k = np.inf
            best_nll = base_nll
        result[st] = best_k
        rows.append({
            "Stat": st, "Active": active, "Calibration events": len(ev),
            "Prior minutes K": best_k, "No-H2H NLL": base_nll,
            "Best H2H NLL": best_nll,
            "NLL gain": (base_nll - best_nll),
            "Reason": "repeat-matchup predictive gain" if active else "H2H did not improve next-game likelihood",
        })
    return result, pd.DataFrame(rows)


def player_h2h_modifiers(
    player_log: pd.DataFrame,
    opponent_abbr: str,
    profile: dict,
    projected_minutes: float,
    rotation_similarity: float = 1.0,
    prior_minutes_by_stat: dict | None = None,
    max_weight: float | None = None,
    residual_history: pd.DataFrame | None = None,
    residual_calibration_by_stat: dict | None = None,
    current_opponent_modifiers: dict | None = None,
):
    """Same-season H2H opportunity correction.

    v2.17.3 production mode is *residualized*: historical H2H counts are
    compared with what the no-H2H model would have expected at the time
    (historical non-pair player rate × leave-pair-out generic opponent effect).
    Only the remaining player×opponent residual is shrunk and applied.

    The residual multiplier has a Gamma prior centered at 1. Its prior mass K
    (expected events) and minute-relevance decay tau are learned league-wide.
    v2.18.2 then continuously averages the finite H2H model toward neutral 1
    using later blocked predictive likelihood; a weak holdout therefore shrinks
    H2H rather than hard-disabling it. ``tau=inf`` means no minute-distance
    penalty.

    The older v2.17.2 prior-minute path is retained for backward compatibility
    when residual inputs are not supplied.
    """
    neutral = {"2PA": 1.0, "3PA": 1.0, "FTA": 1.0, "REB": 1.0, "AST": 1.0}
    if player_log is None or player_log.empty:
        return neutral, pd.DataFrame()

    h = player_log[
        player_log["OPP_ABBR"].astype(str).str.upper().eq(str(opponent_abbr).upper())
    ].copy()
    if h.empty:
        return neutral, pd.DataFrame([{"H2H games": 0, "Posterior H2H weight": 0.0}])

    mins_s = pd.to_numeric(h.get("MIN", 0), errors="coerce").fillna(0.0)
    h = h[mins_s > 0].copy()
    if h.empty:
        return neutral, pd.DataFrame([{"H2H games": 0, "Posterior H2H weight": 0.0}])

    rot = float(np.clip(rotation_similarity, 0.0, 1.0))
    base = {
        "2PA": float(profile.get("two_pa_pm", np.nan)),
        "3PA": float(profile.get("three_pa_pm", np.nan)),
        "FTA": float(profile.get("fta_pm", np.nan)),
        "REB": float(profile.get("reb_pm", np.nan)),
        "AST": float(profile.get("ast_pm", np.nan)),
    }

    # ------------------------------------------------------------------
    # v2.17.3 production path: residualized H2H.
    # ------------------------------------------------------------------
    if (
        residual_history is not None
        and isinstance(residual_history, pd.DataFrame)
        and residual_calibration_by_stat is not None
    ):
        rh = residual_history.copy()
        if rh.empty:
            return neutral, pd.DataFrame([{
                "H2H games": int(len(h)),
                "Residual H2H eligible games": 0,
                "Posterior H2H weight": 0.0,
                "Reason": "historical no-H2H expectation unavailable",
            }])

        out, rows = dict(neutral), []
        current_opp = current_opponent_modifiers or {}

        for st in _RESIDUAL_H2H_STATS:
            cal = (residual_calibration_by_stat or {}).get(st, {}) or {}
            active = bool(cal.get("active", False))
            k = float(cal.get("prior_events_k", np.inf))
            tau = float(cal.get("minute_tau", np.inf))
            predictive_model_weight = float(np.clip(
                cal.get("predictive_model_weight", 1.0 if active else 0.0),
                0.0, 1.0,
            ))

            obs_col = f"{st}_observed"
            exp_col = f"{st}_expected_events"
            if obs_col not in rh.columns or exp_col not in rh.columns:
                eligible = pd.DataFrame()
            else:
                eligible = rh.copy()
                eligible[obs_col] = pd.to_numeric(eligible[obs_col], errors="coerce")
                eligible[exp_col] = pd.to_numeric(eligible[exp_col], errors="coerce")
                eligible["MIN"] = pd.to_numeric(eligible.get("MIN", 0), errors="coerce")
                eligible = eligible[
                    eligible[obs_col].notna()
                    & eligible[exp_col].notna()
                    & (eligible[exp_col] > 0)
                    & (eligible["MIN"] > 0)
                ].copy()

            if eligible.empty or np.isinf(k) or not np.isfinite(k):
                mod = 1.0
                full_eb_mod = 1.0
                post_weight = 0.0
                applied_effect_weight = 0.0
                raw_residual = np.nan
                obs_eff = 0.0
                exp_eff = 0.0
                mean_min_rel = 0.0
            else:
                hm = eligible["MIN"].to_numpy(float)
                if np.isinf(tau):
                    minute_rel = np.ones(len(eligible), dtype=float)
                else:
                    minute_rel = np.exp(
                        -np.abs(hm - float(projected_minutes)) / max(float(tau), 1e-6)
                    )
                rel = rot * minute_rel
                obs = eligible[obs_col].to_numpy(float)
                exp_events = eligible[exp_col].to_numpy(float)
                obs_eff = float(np.sum(obs * rel))
                exp_eff = float(np.sum(exp_events * rel))
                raw_residual = (obs_eff / exp_eff) if exp_eff > 0 else np.nan
                full_eb_mod = (
                    float((obs_eff + k) / max(exp_eff + k, 1e-9))
                    if exp_eff > 0 else 1.0
                )
                # Multiplicative/geometric interpolation is neutral at 1 and
                # preserves symmetry on the log scale for positive/negative
                # matchup residuals.
                mod = float(np.exp(
                    predictive_model_weight * np.log(max(full_eb_mod, 1e-9))
                ))
                post_weight = float(exp_eff / (exp_eff + k)) if exp_eff > 0 else 0.0
                applied_effect_weight = float(post_weight * predictive_model_weight)
                mean_min_rel = float(np.mean(minute_rel)) if len(minute_rel) else 0.0

            out[st] = float(mod if np.isfinite(mod) and mod > 0 else 1.0)
            b = float(base.get(st, np.nan))
            om = float(current_opp.get(st, 1.0))
            current_no_h2h = (
                b * om if np.isfinite(b) and b > 0 and np.isfinite(om) and om > 0
                else np.nan
            )
            rows.append({
                "Stat": st,
                "H2H games": int(len(h)),
                "Residual H2H eligible games": int(len(eligible)),
                "Historical observed events (effective)": obs_eff,
                "Historical no-H2H expected events (effective)": exp_eff,
                "Raw H2H residual ratio": raw_residual,
                "Rotation similarity": rot,
                "Learned minute relevance tau": tau,
                "Mean minute relevance": mean_min_rel,
                "Prior residual events K": k,
                "Predictive H2H model weight": predictive_model_weight,
                "Posterior H2H evidence weight": post_weight,
                "Posterior H2H weight": post_weight,
                "Applied H2H weight": applied_effect_weight,
                "Full EB residual modifier before model averaging": full_eb_mod,
                "Current non-H2H baseline rate/min": b,
                "Current generic opponent modifier": om,
                "Current no-H2H expected rate/min": current_no_h2h,
                "Final H2H residual modifier": out[st],
                # compatibility/audit aliases
                "Final H2H modifier": out[st],
                "Posterior rate/min": (
                    current_no_h2h * out[st]
                    if np.isfinite(current_no_h2h) else np.nan
                ),
            })
        return out, pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Legacy v2.17.2 path retained for old tests/callers.
    # ------------------------------------------------------------------
    mins = pd.to_numeric(h["MIN"], errors="coerce").fillna(0.0).to_numpy(float)
    if float(np.sum(mins)) <= 0:
        return neutral, pd.DataFrame([{"H2H games": 0, "Posterior H2H weight": 0.0}])

    minute_rel = np.exp(-np.abs(mins - float(projected_minutes)) / 10.0)
    rel = rot * minute_rel

    fga = pd.to_numeric(h.get("FGA", 0), errors="coerce").fillna(0.0).to_numpy(float)
    a3 = pd.to_numeric(h.get("FG3A", 0), errors="coerce").fillna(0.0).to_numpy(float)
    stat_counts = {
        "2PA": np.maximum(fga - a3, 0.0),
        "3PA": np.maximum(a3, 0.0),
        "FTA": (pd.to_numeric(h["FTA"], errors="coerce").fillna(0.0).to_numpy(float)
                if "FTA" in h.columns else np.zeros(len(h), dtype=float)),
        "REB": pd.to_numeric(h.get("REB", 0), errors="coerce").fillna(0.0).to_numpy(float),
        "AST": pd.to_numeric(h.get("AST", 0), errors="coerce").fillna(0.0).to_numpy(float),
    }

    legacy_mode = prior_minutes_by_stat is None and max_weight is not None
    kmap = prior_minutes_by_stat or {k: np.inf for k in neutral}

    out, rows = dict(neutral), []
    for st in _RESIDUAL_H2H_STATS:
        b = base[st]
        cw = float(np.sum(stat_counts[st] * rel))
        mw = float(np.sum(mins * rel))
        raw_rate = (cw / mw) if mw > 0 else np.nan
        k = float(kmap.get(st, np.inf))
        if legacy_mode:
            raw_ratio = (raw_rate / b) if np.isfinite(raw_rate) and np.isfinite(b) and b > 0 else 1.0
            raw_ratio = float(np.clip(raw_ratio, 0.70, 1.30))
            maturity = float(len(h) / (len(h) + 2.0))
            mean_min_rel = float(np.mean(minute_rel)) if len(minute_rel) else 0.0
            w = float(min(float(max_weight), float(max_weight) * maturity * rot * mean_min_rel))
            mod = float(np.clip(np.exp(w * np.log(max(raw_ratio, 1e-9))), 0.97, 1.03))
            post = b * mod if np.isfinite(b) else b
        elif not (np.isfinite(b) and b > 0 and mw > 0) or np.isinf(k):
            post = b
            w = 0.0
            mod = 1.0
        else:
            post = (cw + k * b) / (mw + k)
            w = mw / (mw + k)
            mod = float(post / b) if b > 0 else 1.0
        out[st] = float(mod if np.isfinite(mod) and mod > 0 else 1.0)
        rows.append({
            "Stat": st, "H2H games": int(len(h)),
            "H2H raw rate/min": raw_rate, "Non-H2H baseline rate/min": b,
            "Rotation similarity": rot, "Effective H2H minutes": mw,
            "Learned prior minutes K": (np.nan if legacy_mode else k),
            "Posterior H2H weight": w, "Applied H2H weight": w,
            "Posterior rate/min": post, "Final H2H modifier": out[st],
        })
    return out, pd.DataFrame(rows)


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-5, 1 - 1e-5))
    return float(np.log(p / (1.0 - p)))


def _inv_logit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


def _shrink_to_league(obs: float, league: float, games: float, k: float = 10.0) -> float:
    if not (np.isfinite(obs) and np.isfinite(league)):
        return float(league if np.isfinite(league) else obs)
    c = float(np.clip(float(games) / (float(games) + float(k)), 0.0, 1.0))
    return float(c * obs + (1.0 - c) * league)


def team_matchup_modifiers(overall_profile, own_profile=None, elasticities=None):
    """Opponent adjustment for Team Markets using league-calibrated elasticity.

    Conceptually:
        current offense identity
        + beta_stat * (opponent allowed - league average)

    where beta_stat is learned from historical pregame team-games by
    ``fit_opponent_elasticities``. Shares/percentages are adjusted in logit
    space; positive rates are adjusted in log space. This keeps opponent
    information meaningful without treating the opponent's raw allowed average
    as a second full sample or inventing a different hand-tuned exponent for
    every matchup.
    """
    mods = overall_profile.get("modifiers", {})
    out = {
        "FGA": mods.get("FGA_LIVE", 1.0),
        "3P_SHARE": mods.get("3P_SHARE", 1.0),
        "FTA": mods.get("FTA", 1.0),
        "TOV": mods.get("TOV", 1.0),
        "OREB": mods.get("OREB_PER_MISS", 1.0),
        "AST": mods.get("AST_PER_MAKE", 1.0),
        "PF": mods.get("PF", 1.0),
        "BLK": mods.get("BLK", 1.0),
        "3P_PCT": mods.get("3P_PCT", 1.0),
        "2P_PCT": mods.get("2P_PCT", 1.0),
    }
    if not own_profile:
        return out

    elasticities = elasticities or {}
    # If the empirical calibrator is unavailable, use only the same small
    # structural floors used by fit_opponent_elasticities -- never the old
    # hand-tuned 30-55% response coefficients.
    defaults = {
        "FGA_LIVE": 0.04,
        "3P_SHARE": 0.08,
        "FTA": 0.08,
        "TOV": 0.08,
        "OREB_PER_MISS": 0.08,
        "AST_PER_MAKE": 0.06,
        "PF": 0.06,
        "3P_PCT": 0.03,
        "2P_PCT": 0.03,
    }
    rates = overall_profile.get("rates", {}) or {}
    lg = rates.get("league", {}) or {}
    opp = rates.get("opponent", {}) or {}
    games = float(opp.get("games", 0) or 0)

    def beta(name):
        return float(elasticities.get(name, defaults[name]))

    def rate_modifier(feature, own_key, lg_key=None, opp_key=None, bounds=(0.85, 1.15), k=10.0):
        lk = lg_key or feature
        ok = opp_key or feature
        own = float(own_profile.get(own_key, np.nan))
        lv = float(lg.get(lk, np.nan))
        ov = float(opp.get(ok, np.nan))
        if not (np.isfinite(own) and own > 0 and np.isfinite(lv) and lv > 0 and np.isfinite(ov) and ov > 0):
            return None
        ovs = _shrink_to_league(ov, lv, games, k=k)
        target = float(np.exp(np.log(own) + beta(feature) * (np.log(ovs) - np.log(lv))))
        return float(np.clip(target / own, bounds[0], bounds[1]))

    def prob_modifier(feature, own_key, lg_key, opp_key, bounds=(0.94, 1.06), k=14.0):
        own = float(own_profile.get(own_key, np.nan))
        lv = float(lg.get(lg_key, np.nan))
        ov = float(opp.get(opp_key, np.nan))
        if not (np.isfinite(own) and 0 < own < 1 and np.isfinite(lv) and 0 < lv < 1 and np.isfinite(ov) and 0 < ov < 1):
            return None
        ovs = _shrink_to_league(ov, lv, games, k=k)
        target = _inv_logit(_logit(own) + beta(feature) * (_logit(ovs) - _logit(lv)))
        return float(np.clip(target / own, bounds[0], bounds[1]))

    replacements = {
        "FGA": rate_modifier("FGA_LIVE", "fga_live", bounds=(0.94, 1.06)),
        "FTA": rate_modifier("FTA", "fta_pp", bounds=(0.84, 1.16)),
        "TOV": rate_modifier("TOV", "tov_pp", bounds=(0.86, 1.14)),
        "OREB": rate_modifier("OREB_PER_MISS", "oreb_per_miss", bounds=(0.86, 1.14)),
        "AST": rate_modifier("AST_PER_MAKE", "assist_per_make", bounds=(0.88, 1.12)),
        "PF": rate_modifier("PF", "pf_pp", bounds=(0.86, 1.14)),
        "3P_SHARE": prob_modifier("3P_SHARE", "three_share", "3P_SHARE", "3P_SHARE", bounds=(0.84, 1.16), k=10.0),
        # Shooting efficiency remains the most strongly shrunk opponent layer.
        "3P_PCT": prob_modifier("3P_PCT", "three_pct", "3P_PCT", "3P_PCT", bounds=(0.94, 1.06), k=18.0),
        "2P_PCT": prob_modifier("2P_PCT", "two_pct", "2P_PCT", "2P_PCT", bounds=(0.94, 1.06), k=18.0),
    }
    for key, value in replacements.items():
        if value is not None and np.isfinite(value):
            out[key] = float(value)

    # BLK keeps the v3 own-ability/opponent-suffered logic that has already
    # calibrated well in audits; do not overwrite it with the generic learner.
    return out


def block_position_susceptibility_modifier(
    player_df: pd.DataFrame,
    offense_abbr: str,
    current_pool: pd.DataFrame | None = None,
    out_players: list[str] | None = None,
):
    """Optional small positional block-susceptibility correction.

    Requires a player-level blocked-attempt column (BLKA / BA /
    BLOCKS_AGAINST). If the loaded dataset does not contain one, returns a
    neutral modifier rather than fabricating position information from 2PA.

    The modifier is RELATIVE to the offense's own overall blocked-attempt rate,
    so it does not double count the overall opponent-BLK-suffered factor.
    """
    if player_df is None or player_df.empty or "POSITION_GROUP" not in player_df.columns:
        return 1.0, pd.DataFrame([{"Available": False, "Reason": "No player/position data"}])

    ba_col = next((c for c in ["BLKA", "BA", "BLOCKS_AGAINST", "BLOCKED_ATTEMPTS"] if c in player_df.columns), None)
    if ba_col is None:
        return 1.0, pd.DataFrame([{
            "Available": False,
            "Reason": "No player-level blocked-attempt field in loaded data; positional BLK modifier kept neutral",
        }])

    x = player_df.copy()
    x[ba_col] = pd.to_numeric(x[ba_col], errors="coerce").fillna(0.0)
    x["FGA"] = pd.to_numeric(x["FGA"], errors="coerce").fillna(0.0)
    league = x[x["FGA"] > 0].copy()
    team = league[league["TEAM_ABBR"].astype(str).str.upper().eq(str(offense_abbr).upper())].copy()
    if team.empty:
        return 1.0, pd.DataFrame([{"Available": False, "Reason": "No offense rows for positional BLK"}])

    out_norm = {str(v).casefold() for v in (out_players or [])}
    # Current shot mix: recent FGA of active current players by broad position.
    current_mix = team.copy()
    if current_pool is not None and not current_pool.empty:
        active_names = set(
            current_pool[
                current_pool["TEAM_ABBR"].astype(str).str.upper().eq(str(offense_abbr).upper())
            ]["PLAYER_NAME"].astype(str).str.casefold().tolist()
        ) - out_norm
        current_mix = current_mix[current_mix["PLAYER_NAME"].astype(str).str.casefold().isin(active_names)].copy()
    current_mix = current_mix.sort_values("GAME_DATE").groupby("PLAYER_ID", group_keys=False).tail(5)

    rows = []
    weighted_ratio = 0.0
    weight_total = 0.0
    for pos in ["G", "F", "C"]:
        lp = league[league["POSITION_GROUP"].astype(str).eq(pos)]
        tp = team[team["POSITION_GROUP"].astype(str).eq(pos)]
        mp = current_mix[current_mix["POSITION_GROUP"].astype(str).eq(pos)]
        l_rate = _safe_div(lp[ba_col].sum(), lp["FGA"].sum(), np.nan)
        t_rate = _safe_div(tp[ba_col].sum(), tp["FGA"].sum(), np.nan)
        shot_weight = float(mp["FGA"].sum())
        ratio = (t_rate / l_rate) if np.isfinite(t_rate) and np.isfinite(l_rate) and l_rate > 0 else np.nan
        if np.isfinite(ratio) and shot_weight > 0:
            weighted_ratio += shot_weight * ratio
            weight_total += shot_weight
        rows.append({
            "Position": pos,
            "Team BLKA/FGA": t_rate,
            "League BLKA/FGA": l_rate,
            "Raw position ratio": ratio,
            "Current active recent FGA weight": shot_weight,
        })

    if weight_total <= 0:
        return 1.0, pd.DataFrame(rows)

    pos_ratio = weighted_ratio / weight_total
    overall_team = _safe_div(team[ba_col].sum(), team["FGA"].sum(), np.nan)
    overall_lg = _safe_div(league[ba_col].sum(), league["FGA"].sum(), np.nan)
    overall_ratio = (overall_team / overall_lg) if np.isfinite(overall_team) and np.isfinite(overall_lg) and overall_lg > 0 else 1.0
    relative = float(np.clip(pos_ratio / max(overall_ratio, 1e-9), 0.85, 1.15))
    mod = float(np.clip(relative ** 0.20, 0.97, 1.03))
    audit = pd.DataFrame(rows)
    audit.attrs["summary"] = {
        "Available": True,
        "Weighted position ratio": pos_ratio,
        "Overall offense ratio": overall_ratio,
        "Relative position ratio": relative,
        "Applied modifier": mod,
    }
    return mod, audit


# ---------------------------------------------------------------------
# v2.13 empirical opponent-elasticity calibration
# ---------------------------------------------------------------------
def _single_game_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one-row-per-team-game structural features used by Team Markets."""
    x = df.copy()
    for c in ["FGA","FGM","FG3A","FG3M","FTA","TOV","OREB","AST","PF"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    x["POSS_CAL"] = estimate_possessions(x).clip(lower=1e-6)
    live = (x["POSS_CAL"] - x["TOV"]).clip(lower=1e-6)
    misses = (x["FGA"] - x["FGM"]).clip(lower=1e-6)
    a2 = (x["FGA"] - x["FG3A"]).clip(lower=1e-6)
    m2 = (x["FGM"] - x["FG3M"]).clip(lower=0.0)
    out = pd.DataFrame({
        "GAME_DATE": pd.to_datetime(x["GAME_DATE"], errors="coerce"),
        "GAME_ID": x["GAME_ID"].astype(str),
        "TEAM_ABBR": x["TEAM_ABBR"].astype(str),
        "OPP_ABBR": x["OPP_ABBR"].astype(str),
        "FGA_LIVE": x["FGA"] / live,
        "3P_SHARE": x["FG3A"] / x["FGA"].replace(0, np.nan),
        "FTA": x["FTA"] / x["POSS_CAL"],
        "TOV": x["TOV"] / x["POSS_CAL"],
        "OREB_PER_MISS": x["OREB"] / misses,
        "AST_PER_MAKE": x["AST"] / x["FGM"].replace(0, np.nan),
        "PF": x["PF"] / x["POSS_CAL"],
        "3P_PCT": x["FG3M"] / x["FG3A"].replace(0, np.nan),
        "2P_PCT": m2 / a2,
    })
    return out.replace([np.inf, -np.inf], np.nan)


def _transform_feature(name: str, s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").astype(float)
    if name in {"3P_SHARE", "3P_PCT", "2P_PCT"}:
        v = v.clip(1e-4, 1 - 1e-4)
        return np.log(v / (1.0 - v))
    return np.log(v.clip(lower=1e-4))


def fit_opponent_elasticities(league_team_logs: pd.DataFrame):
    """Estimate stat-specific opponent response from historical WNBA team-games.

    There is deliberately NO universal "opponent weight".  For every derivative
    feature we build a pregame own baseline, a pregame opponent-allowed baseline
    and a pregame league baseline.  The coefficient is the predictive slope of
    the realized deviation from the team's own identity on the opponent's
    deviation from league average.

    v2.14 changes two things versus v2.13:
      * strong hand-picked priors (e.g. 0.55 for 3P share) are removed;
      * the learned slope is shrunk by both sample size and its statistical
        signal, with only a SMALL non-zero structural floor.  Thus opponent
        context always matters a little, but it only matters a lot when WNBA
        history actually supports it.

    Shares / shooting percentages are learned in logit space. Positive rates are
    learned in log space. The model remains league-level rather than team-specific
    because ~30-40 games are far too noisy for a separate elasticity per team.
    """
    if league_team_logs is None or league_team_logs.empty:
        return {}, pd.DataFrame()

    x = _single_game_structural_features(league_team_logs)
    x = x.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBR"]).reset_index(drop=True)
    features = [
        "FGA_LIVE", "3P_SHARE", "FTA", "TOV", "OREB_PER_MISS",
        "AST_PER_MAKE", "PF", "3P_PCT", "2P_PCT",
    ]

    # Small structural floors only. These are NOT target weights; they simply
    # prevent a noisy early-season fit from pretending the opponent is irrelevant.
    floors = {
        "FGA_LIVE": 0.04,
        "3P_SHARE": 0.08,
        "FTA": 0.08,
        "TOV": 0.08,
        "OREB_PER_MISS": 0.08,
        "AST_PER_MAKE": 0.06,
        "PF": 0.06,
        "3P_PCT": 0.03,
        "2P_PCT": 0.03,
    }
    caps = {
        "FGA_LIVE": 0.55,
        "3P_SHARE": 0.95,
        "FTA": 0.85,
        "TOV": 0.75,
        "OREB_PER_MISS": 0.75,
        "AST_PER_MAKE": 0.65,
        "PF": 0.70,
        # Opponent FG% is real, but raw opponent shooting percentage is noisy.
        "3P_PCT": 0.35,
        "2P_PCT": 0.35,
    }

    result, audit = {}, []
    for feat in features:
        tmp = x[["GAME_DATE", "GAME_ID", "TEAM_ABBR", "OPP_ABBR", feat]].copy()

        # Pregame own-offense mean.
        tmp["OWN_PRIOR"] = (
            tmp.groupby("TEAM_ABBR")[feat]
            .transform(lambda z: z.expanding(min_periods=5).mean().shift(1))
        )
        tmp["OWN_N"] = tmp.groupby("TEAM_ABBR").cumcount()

        # Pregame defensive allowed mean. Rows with OPP_ABBR=D are exactly the
        # offensive outcomes previously allowed by defense D.
        tmp["DEF_PRIOR"] = (
            tmp.groupby("OPP_ABBR")[feat]
            .transform(lambda z: z.expanding(min_periods=5).mean().shift(1))
        )
        tmp["DEF_N"] = tmp.groupby("OPP_ABBR").cumcount()

        # Pregame league baseline by GAME, preventing the other row of the same
        # game from leaking into the prior.
        game_means = (
            tmp.groupby(["GAME_DATE", "GAME_ID"], sort=True)[feat]
            .mean().reset_index().sort_values(["GAME_DATE", "GAME_ID"])
        )
        game_means["LG_PRIOR"] = game_means[feat].expanding(min_periods=10).mean().shift(1)
        tmp = tmp.merge(
            game_means[["GAME_DATE", "GAME_ID", "LG_PRIOR"]],
            on=["GAME_DATE", "GAME_ID"], how="left",
        )
        tmp = tmp.dropna(subset=[feat, "OWN_PRIOR", "DEF_PRIOR", "LG_PRIOR"])
        tmp = tmp[(tmp["OWN_N"] >= 5) & (tmp["DEF_N"] >= 5)].copy()

        floor = floors[feat]
        cap = caps[feat]
        raw = np.nan
        se_beta = np.nan
        base_rmse = adj_rmse = np.nan
        gain = 0.0
        signal_conf = 0.0
        sample_conf = 0.0

        if len(tmp) < 60:
            beta = floor
        else:
            y_actual = _transform_feature(feat, tmp[feat])
            own_t = _transform_feature(feat, tmp["OWN_PRIOR"])
            def_t = _transform_feature(feat, tmp["DEF_PRIOR"])
            lg_t = _transform_feature(feat, tmp["LG_PRIOR"])

            xv = (def_t - lg_t).clip(-0.75, 0.75).to_numpy(dtype=float)
            yv = (y_actual - own_t).clip(-0.90, 0.90).to_numpy(dtype=float)
            good = np.isfinite(xv) & np.isfinite(yv)
            xv, yv = xv[good], yv[good]

            denom = float(np.sum(xv * xv))
            if denom <= 1e-9 or len(xv) < 40:
                beta = floor
            else:
                raw_unclipped = float(np.sum(xv * yv) / denom)
                raw = float(np.clip(raw_unclipped, 0.0, 1.25))

                residual = yv - raw * xv
                dof = max(len(xv) - 1, 1)
                sigma2 = float(np.sum(residual * residual) / dof)
                se_beta = float(np.sqrt(max(sigma2 / denom, 0.0)))

                # How believable is the sign/magnitude of the slope?
                signal_conf = float(
                    (raw * raw) / (raw * raw + se_beta * se_beta + 1e-9)
                )
                sample_conf = float(len(xv) / (len(xv) + 120.0))

                beta = floor + (raw - floor) * signal_conf * sample_conf
                beta = float(np.clip(beta, floor, cap))

                base_rmse = float(np.sqrt(np.mean(yv ** 2))) if len(yv) else np.nan
                adj_rmse = float(np.sqrt(np.mean((yv - beta * xv) ** 2))) if len(yv) else np.nan
                if np.isfinite(base_rmse) and base_rmse > 0 and np.isfinite(adj_rmse):
                    gain = float((base_rmse - adj_rmse) / base_rmse)
                    # If the opponent layer does not improve historical prediction,
                    # do not let noisy correlation create a large adjustment.
                    if gain <= 0:
                        beta = floor
                        adj_rmse = float(np.sqrt(np.mean((yv - beta * xv) ** 2)))
                        gain = float((base_rmse - adj_rmse) / base_rmse)

        result[feat] = float(beta)
        audit.append({
            "Feature": feat,
            "Learned elasticity": float(beta),
            "Raw slope": raw,
            "Slope SE": se_beta,
            "Signal confidence": signal_conf,
            "Sample confidence": sample_conf,
            "Structural floor": floor,
            "Calibration rows": int(len(tmp)),
            "Base transformed RMSE": base_rmse,
            "Opponent-adjusted RMSE": adj_rmse,
            "Relative RMSE gain": gain,
        })

    return result, pd.DataFrame(audit)
