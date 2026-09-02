from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from core.buckets import WeightConfig, split_non_overlapping, active_weights, weighted_average_feature
from core.player_role_state import adaptive_role_state, role_state_table


@dataclass
class PlayerContext:
    projected_minutes: float
    minutes_sd: float = 2.0

    # Central matchup pace relative to historical pace already embedded in rates.
    pace_multiplier: float = 1.00

    # Opponent environment, already shrinked; 1.00 = neutral.
    opp_pts: float = 1.00
    opp_reb: float = 1.00
    opp_ast: float = 1.00
    opp_3pa: float = 1.00
    opp_2pa: float = 1.00
    opp_fta: float = 1.00
    # Shooting-efficiency context is deliberately much weaker than volume context.
    opp_three_pct: float = 1.00
    opp_two_pct: float = 1.00

    # Role redistribution / trader override.
    usage: float = 1.00
    creation: float = 1.00
    reb_role: float = 1.00
    three_role: float = 1.00
    fta_role: float = 1.00

    # H2H context only: recommended range 0.90 - 1.10.
    h2h_pts: float = 1.00  # legacy; kept for backward compatibility
    h2h_2pa: float = 1.00
    h2h_reb: float = 1.00
    h2h_ast: float = 1.00
    h2h_3pa: float = 1.00
    h2h_fta: float = 1.00


def _safe_div(a, b, default=0.0):
    return float(a) / float(b) if b and np.isfinite(b) and b > 0 else default


def _row_weights(df: pd.DataFrame, game_weights: Optional[Dict[str, float]]) -> np.ndarray:
    if not game_weights or "GAME_ID" not in df.columns:
        return np.ones(len(df), dtype=float)
    return np.asarray(
        [float(game_weights.get(str(gid), 1.0)) for gid in df["GAME_ID"]],
        dtype=float,
    )


def _weighted_sum(series: pd.Series, weights: np.ndarray) -> float:
    vals = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return float(np.sum(vals * weights))


def _features(df: pd.DataFrame, game_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    if df.empty:
        return {}
    w = _row_weights(df, game_weights)
    mins_arr = pd.to_numeric(df["MIN"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mins = float(np.sum(mins_arr * w))
    fga = _weighted_sum(df["FGA"], w)
    a3 = _weighted_sum(df["FG3A"], w)
    m3 = _weighted_sum(df["FG3M"], w)
    fgm = _weighted_sum(df["FGM"], w)
    fta = _weighted_sum(df["FTA"], w)
    ftm = _weighted_sum(df["FTM"], w)
    a2 = max(fga - a3, 0.0)
    m2 = max(fgm - m3, 0.0)
    return {
        "games": len(df),
        "effective_games": float(np.sum(w)),
        "min_pg": float(np.average(mins_arr, weights=w)) if len(w) and np.sum(w) > 0 else np.nan,
        "two_pa_pm": _safe_div(a2, mins),
        "three_pa_pm": _safe_div(a3, mins),
        "fta_pm": _safe_div(fta, mins),
        "reb_pm": _safe_div(_weighted_sum(df["REB"], w), mins),
        "ast_pm": _safe_div(_weighted_sum(df["AST"], w), mins),
        "pts_pm": _safe_div(_weighted_sum(df["PTS"], w), mins),
        "three_pct": _safe_div(m3, a3, np.nan),
        "two_pct": _safe_div(m2, a2, np.nan),
        "ft_pct": _safe_div(ftm, fta, np.nan),
        "three_att": a3,
        "two_att": a2,
        "ft_att": fta,
    }


def _shrink(obs, attempts, prior, prior_attempts):
    if not np.isfinite(obs):
        obs, attempts = prior, 0.0
    return float((obs * attempts + prior * prior_attempts) / (attempts + prior_attempts))


def build_player_profile(
    df: pd.DataFrame,
    cfg: WeightConfig,
    game_weights: Optional[Dict[str, float]] = None,
    game_weights_by_stat: Optional[Dict[str, Dict[str, float]]] = None,
    exclude_opponent_abbr: Optional[str] = None,
) -> Tuple[dict, pd.DataFrame]:
    """
    Outer Old/G6-10/L5 weights remain non-overlapping. Availability similarity
    is an INNER weight only. v2.15 optionally uses a different single-score map
    for each opportunity family (FGA/3PA/FTA/REB/AST), so a guard absence cannot
    mechanically reweight a forward's rebound history as strongly as creation.

    ``game_weights`` is retained for backward compatibility. If
    ``game_weights_by_stat`` is supplied, it wins for the mapped feature only.
    """
    x = df.sort_values("GAME_DATE").copy()

    # v2.17.2 disjoint player H2H: split the real timeline FIRST, then remove
    # current-opponent rows from opportunity buckets.  Those rows are reserved
    # for the explicit empirical-Bayes H2H layer later in the pipeline, so one
    # Phoenix game cannot both move the adaptive role-state and then be counted
    # again as matchup-specific H2H evidence.  Shooting skill still uses the
    # full sample because H2H never changes 3P%/2P%/FT% directly.
    raw_buckets = split_non_overlapping(x)
    if exclude_opponent_abbr and "OPP_ABBR" in x.columns:
        opp = str(exclude_opponent_abbr).upper()
        buckets = {
            k: v[~v["OPP_ABBR"].astype(str).str.upper().eq(opp)].copy()
            for k, v in raw_buckets.items()
        }
        role_x = x[~x["OPP_ABBR"].astype(str).str.upper().eq(opp)].copy()
    else:
        buckets = raw_buckets
        role_x = x
    weights = active_weights(buckets, cfg)
    by = game_weights_by_stat or {}

    feature_stat = {
        "two_pa_pm": "FGA",
        "three_pa_pm": "3PA",
        "fta_pm": "FTA",
        "reb_pm": "REB",
        "ast_pm": "AST",
        "pts_pm": "FGA",
    }

    # Calculate only the small set of distinct inner-weight views required by
    # this player. Minutes are intentionally left neutral because the 200-minute
    # engine already models them directly.
    cache = {}
    def bucket_features(bucket_name: str, stat_key: str | None):
        key = (bucket_name, stat_key or "NEUTRAL")
        if key not in cache:
            wm = by.get(stat_key, game_weights) if stat_key else None
            cache[key] = _features(buckets[bucket_name], game_weights=wm)
        return cache[key]

    profile = {}
    neutral_feats = {k: bucket_features(k, None) for k in buckets}
    profile["min_pg"] = weighted_average_feature(neutral_feats, weights, "min_pg")
    audit_feats = neutral_feats

    for key, stat_key in feature_stat.items():
        feats = {k: bucket_features(k, stat_key) for k in buckets}
        profile[key] = weighted_average_feature(feats, weights, key)

    # v2.17.1 adaptive player-role state.  The existing non-overlapping bucket
    # profile remains the prior/base.  A local-level state-space model estimates
    # whether the player's own per-minute opportunity state has moved through
    # time.  The process variance is selected by walk-forward predictive
    # likelihood and model-averaged against q=0, so there is no fixed L5 boost
    # and no automatic role-change switch.  Availability-similarity weights, if
    # present, are reused as information weights so the same OUT state is not
    # counted twice.
    role_state_results = []
    for key in ("two_pa_pm", "three_pa_pm", "fta_pm", "reb_pm", "ast_pm"):
        stat_key = feature_stat[key]
        wm = by.get(stat_key, game_weights)
        rs = adaptive_role_state(role_x, key, profile[key], game_weights=wm)
        if np.isfinite(rs.applied_rate) and rs.applied_rate > 0:
            profile[key] = rs.applied_rate
        role_state_results.append(rs)
    profile["_role_state_audit"] = role_state_table(role_state_results)

    # Shooting ability intentionally uses the larger unweighted sample. A same-role
    # split changes opportunities/role first; it does not declare a hot L5 as true skill.
    full = _features(x, game_weights=None)
    # Slightly lighter 3P prior than before: elite/high-volume shooters are no longer
    # pulled toward 34% as aggressively, while small samples remain regularized.
    profile["three_pct"] = _shrink(full.get("three_pct", np.nan), full.get("three_att", 0), 0.340, 32)
    profile["two_pct"] = _shrink(full.get("two_pct", np.nan), full.get("two_att", 0), 0.510, 70)
    profile["ft_pct"] = _shrink(full.get("ft_pct", np.nan), full.get("ft_att", 0), 0.785, 40)

    audit = []
    for k in ("old", "mid", "l5"):
        row = {"bucket": k, "weight": weights[k], **audit_feats.get(k, {"games": 0})}
        row["raw_bucket_games"] = int(len(raw_buckets.get(k, [])))
        row["H2H_excluded"] = int(len(raw_buckets.get(k, [])) - len(buckets.get(k, [])))
        # Transparent effective sample sizes under each stat-specific map.
        for stat_key in ("FGA", "3PA", "FTA", "REB", "AST"):
            sf = bucket_features(k, stat_key)
            row[f"effective_games_{stat_key}"] = sf.get("effective_games", len(buckets[k]))
        audit.append(row)
    return profile, pd.DataFrame(audit)


def simulate_player(profile: dict, ctx: PlayerContext, n=50_000, seed=1, opportunity_mult=1.0):
    rng = np.random.default_rng(seed)

    z_min = rng.normal(size=n)
    z_pace = rng.normal(size=n)
    z_role = rng.normal(size=n)
    z_perim = rng.normal(size=n)
    z_foul = rng.normal(size=n)
    z_reb = rng.normal(size=n)
    z_ast = rng.normal(size=n)
    z_shoot = rng.normal(size=n)

    mins = np.clip(
        ctx.projected_minutes + ctx.minutes_sd * z_min,
        max(4.0, ctx.projected_minutes - 8.0),
        ctx.projected_minutes + 8.0,
    )
    pace = ctx.pace_multiplier * np.exp(0.035 * z_pace - 0.5 * 0.035**2)
    role = np.exp(0.05 * z_role - 0.5 * 0.05**2)
    perim = np.exp(0.07 * z_perim - 0.5 * 0.07**2)
    foul = np.exp(0.10 * z_foul - 0.5 * 0.10**2)
    reb_env = np.exp(0.10 * z_reb - 0.5 * 0.10**2)
    ast_env = np.exp(0.10 * z_ast - 0.5 * 0.10**2)

    base = pace * role * opportunity_mult

    lam3 = mins * profile["three_pa_pm"] * base * perim * ctx.usage * ctx.three_role * ctx.opp_3pa * ctx.h2h_3pa
    # 2PA volume must use opponent 2PA allowance, not opponent PTS allowance.
    # PTS is an outcome; 2PA is an opportunity channel.
    lam2 = mins * profile["two_pa_pm"] * base * ctx.usage * ctx.opp_2pa * ctx.h2h_2pa
    lamft = mins * profile["fta_pm"] * base * foul * ctx.usage * ctx.fta_role * ctx.opp_fta * ctx.h2h_fta

    a3 = rng.poisson(np.clip(lam3, 0.001, None))
    a2 = rng.poisson(np.clip(lam2, 0.001, None))
    fta = rng.poisson(np.clip(lamft, 0.001, None))

    p3_central = np.clip(profile["three_pct"] * ctx.opp_three_pct, 0.05, 0.70)
    p2_central = np.clip(profile["two_pct"] * ctx.opp_two_pct, 0.15, 0.80)
    p3 = np.clip(p3_central + 0.035 * z_shoot, 0.02, 0.98)
    p2 = np.clip(p2_central + 0.026 * z_shoot, 0.02, 0.98)
    pft = np.clip(profile["ft_pct"] + 0.010 * z_shoot, 0.02, 0.98)

    m3 = rng.binomial(a3, p3)
    m2 = rng.binomial(a2, p2)
    ftm = rng.binomial(fta, pft)
    pts = 3*m3 + 2*m2 + ftm

    # Makes support assists; misses support rebound opportunity.
    ast_shot = np.clip(1 + 0.18*z_shoot, 0.70, 1.35)
    reb_shot = np.clip(1 - 0.10*z_shoot, 0.75, 1.30)

    lam_ast = mins * profile["ast_pm"] * pace * role * ast_env * ast_shot * opportunity_mult * ctx.creation * ctx.opp_ast * ctx.h2h_ast
    lam_reb = mins * profile["reb_pm"] * pace * reb_env * reb_shot * opportunity_mult * ctx.reb_role * ctx.opp_reb * ctx.h2h_reb

    ast = rng.poisson(np.clip(lam_ast, 0.001, None))
    reb = rng.poisson(np.clip(lam_reb, 0.001, None))

    out = pd.DataFrame({
        "MIN": mins, "PTS": pts, "REB": reb, "AST": ast,
        "3PM": m3, "3PA": a3, "FTA": fta,
    })
    out["PRA"] = out.PTS + out.REB + out.AST
    out["PR"] = out.PTS + out.REB
    out["PA"] = out.PTS + out.AST
    out["AR"] = out.AST + out.REB
    return out
