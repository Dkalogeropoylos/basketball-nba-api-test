from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.exposure import regulation_equivalent_factor

from core.buckets import (
    WeightConfig,
    split_non_overlapping,
    active_weights,
    weighted_average_feature,
)


@dataclass
class TeamContext:
    projected_possessions: float
    possessions_sd: float = 3.0

    # Offense-vs-opponent interaction multipliers.
    # 1.00 = neutral. These are already shrinked contextual adjustments,
    # not extra samples.
    # v2.11 shot architecture: first generate total FGA, then allocate the
    # shot mix through 3P_SHARE. Legacy three_pa/two_pa fields are retained
    # for backward compatibility but are no longer used by simulate_game().
    fga: float = 1.0
    three_share: float = 1.0
    three_pa: float = 1.0
    three_pct: float = 1.0
    two_pa: float = 1.0
    two_pct: float = 1.0
    fta: float = 1.0
    tov: float = 1.0
    oreb: float = 1.0
    ast: float = 1.0
    pf: float = 1.0

    # Defensive opportunity conversion. In the coupled game simulator these
    # are applied to the OPPONENT'S turnovers / misses / 2P misses.
    dreb: float = 1.0
    stl: float = 1.0
    blk: float = 1.0
    # Optional RELATIVE positional susceptibility correction. Neutral unless
    # a player-level blocked-attempt field is actually available.
    blk_position: float = 1.0
    # Legacy field retained; v2.11 folds H2H into the disjoint team profile.
    blk_h2h: float = 1.0


def estimate_possessions(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(df["FGA"], errors="coerce").fillna(0)
        - pd.to_numeric(df["OREB"], errors="coerce").fillna(0)
        + pd.to_numeric(df["TOV"], errors="coerce").fillna(0)
        + 0.44 * pd.to_numeric(df["FTA"], errors="coerce").fillna(0)
    )


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a) / float(b) if np.isfinite(b) and b > 0 else default


def _weighted_sum(series: pd.Series, weights: np.ndarray) -> float:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return float(np.sum(values * weights))


def _row_weights(df: pd.DataFrame, game_weights: Optional[Dict[str, float]]) -> np.ndarray:
    if not game_weights or "GAME_ID" not in df.columns:
        return np.ones(len(df), dtype=float)
    return np.asarray(
        [float(game_weights.get(str(gid), 1.0)) for gid in df["GAME_ID"]],
        dtype=float,
    )


def _paired_opponent_rows(
    own_df: pd.DataFrame,
    league_team_logs: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Attach the opponent box-score row for each own-team game."""
    if league_team_logs is None or own_df.empty or "GAME_ID" not in own_df.columns:
        return pd.DataFrame()

    need = [
        "GAME_ID", "TEAM_ABBR", "FGM", "FGA", "FG3M", "FG3A",
        "OREB", "DREB", "REB", "TOV", "FTA", "PF",
    ]
    available = [c for c in need if c in league_team_logs.columns]
    if "GAME_ID" not in available or "TEAM_ABBR" not in available:
        return pd.DataFrame()

    opp = league_team_logs[available].copy()
    rename = {c: f"OPP_{c}" for c in available if c != "GAME_ID"}
    opp = opp.rename(columns=rename)

    own_cols = [c for c in ["GAME_ID", "TEAM_ABBR", "OPP_ABBR", "DREB", "STL", "BLK"] if c in own_df.columns]
    paired = own_df[own_cols].copy().merge(opp, on="GAME_ID", how="left")

    if "OPP_ABBR" in paired.columns and "OPP_TEAM_ABBR" in paired.columns:
        paired = paired[
            paired["OPP_ABBR"].astype(str).str.upper()
            == paired["OPP_TEAM_ABBR"].astype(str).str.upper()
        ].copy()
    elif "TEAM_ABBR" in paired.columns and "OPP_TEAM_ABBR" in paired.columns:
        paired = paired[
            paired["TEAM_ABBR"].astype(str).str.upper()
            != paired["OPP_TEAM_ABBR"].astype(str).str.upper()
        ].copy()

    return paired.drop_duplicates(subset=["GAME_ID"]).reset_index(drop=True)


def _feat(
    df: pd.DataFrame,
    league_team_logs: Optional[pd.DataFrame] = None,
    game_weights: Optional[Dict[str, float]] = None,
) -> dict:
    if df.empty:
        return {}

    x = df.copy()
    w = _row_weights(x, game_weights)
    poss_rows = estimate_possessions(x).to_numpy(dtype=float)
    reg_factor = regulation_equivalent_factor(x).to_numpy(dtype=float)
    poss_rows_reg = poss_rows * reg_factor
    poss = float(np.sum(poss_rows * w))

    fga = _weighted_sum(x["FGA"], w)
    fgm = _weighted_sum(x["FGM"], w)
    a3 = _weighted_sum(x["FG3A"], w)
    m3 = _weighted_sum(x["FG3M"], w)
    a2 = max(fga - a3, 0.0)
    m2 = max(fgm - m3, 0.0)
    fta = _weighted_sum(x["FTA"], w)
    ftm = _weighted_sum(x["FTM"], w)
    tov = _weighted_sum(x["TOV"], w)
    oreb = _weighted_sum(x["OREB"], w)
    dreb = _weighted_sum(x["DREB"], w) if "DREB" in x.columns else 0.0
    misses = max(fga - fgm, 0.0)
    live_poss = max(poss - tov, 1e-9)

    out = {
        "games": int(len(x)),
        "effective_games": float(np.sum(w)),
        # Pace/count exposure is regulation-equivalent; event rates below keep
        # their natural raw opportunity denominators.
        "poss_pg": float(np.average(poss_rows_reg, weights=w)) if len(w) else np.nan,
        "poss_pg_raw": float(np.average(poss_rows, weights=w)) if len(w) else np.nan,
        "ot_games": int(np.sum(reg_factor < 0.999999)),
        "reg_eq_factor_mean": float(np.average(reg_factor, weights=w)) if len(w) else 1.0,
        "three_pa_pp": _safe_div(a3, poss),
        "two_pa_pp": _safe_div(a2, poss),
        "three_pa_live": _safe_div(a3, live_poss),
        "two_pa_live": _safe_div(a2, live_poss),
        "fga_live": _safe_div(fga, live_poss),
        "three_share": _safe_div(a3, fga, 0.35),
        "fta_pp": _safe_div(fta, poss),
        "tov_pp": _safe_div(tov, poss),
        "oreb_pp": _safe_div(oreb, poss),
        "dreb_pp": _safe_div(dreb, poss),
        "oreb_per_miss": _safe_div(oreb, misses),
        "ast_pp": _safe_div(_weighted_sum(x["AST"], w), poss),
        "stl_pp": _safe_div(_weighted_sum(x["STL"], w), poss),
        "blk_pp": _safe_div(_weighted_sum(x["BLK"], w), poss),
        "pf_pp": _safe_div(_weighted_sum(x["PF"], w), poss),
        "three_pct": _safe_div(m3, a3, np.nan),
        "two_pct": _safe_div(m2, a2, np.nan),
        "ft_pct": _safe_div(ftm, fta, np.nan),
        "three_att": a3,
        "two_att": a2,
        "ft_att": fta,
    }

    # Defensive stats must be driven by OPPONENT opportunities, never the
    # team's own TOV/2PA/misses. Attach the paired opponent row for each game.
    paired = _paired_opponent_rows(x, league_team_logs)
    if not paired.empty:
        # Recreate the same inner rotation-similarity weights on the paired rows.
        pw = _row_weights(paired, game_weights)
        opp_fga = _weighted_sum(paired["OPP_FGA"], pw)
        opp_fgm = _weighted_sum(paired["OPP_FGM"], pw)
        opp_a3 = _weighted_sum(paired["OPP_FG3A"], pw)
        opp_m3 = _weighted_sum(paired["OPP_FG3M"], pw)
        opp_a2 = max(opp_fga - opp_a3, 0.0)
        opp_m2 = max(opp_fgm - opp_m3, 0.0)
        opp_misses = max(opp_fga - opp_fgm, 0.0)
        opp_oreb = _weighted_sum(paired["OPP_OREB"], pw)
        opp_tov = _weighted_sum(paired["OPP_TOV"], pw)
        opp_2miss = max(opp_a2 - opp_m2, 0.0)

        own_dreb = _weighted_sum(paired["DREB"], pw)
        own_stl = _weighted_sum(paired["STL"], pw)
        own_blk = _weighted_sum(paired["BLK"], pw)

        # DREB is conditional on an opponent miss that was NOT recovered by
        # the offense. This prevents OREB + DREB double counting.
        dreb_chances = max(opp_misses - opp_oreb, 0.0)
        out["dreb_capture"] = _safe_div(own_dreb, dreb_chances, 0.94)

        # A steal is a subset of opponent turnovers.
        out["stl_per_opp_tov"] = _safe_div(own_stl, opp_tov, 0.55)

        # Learn block ability per opponent 2PA. Using opponent misses in the
        # denominator confounds rim protection with opponent shooting luck.
        out["blk_per_opp_2pa"] = _safe_div(own_blk, opp_a2, 0.075)
        out["opp_2pa_att"] = opp_a2
    else:
        # Conservative fallbacks only if paired game rows are unavailable.
        out["dreb_capture"] = 0.94
        out["stl_per_opp_tov"] = 0.55
        out["blk_per_opp_2pa"] = _safe_div(out["blk_pp"], max(out["two_pa_pp"], 0.05), 0.075)
        out["opp_2pa_att"] = out["two_att"]

    # Useful structural conversion rate for AST simulation.
    made_fg_pp = out["three_pa_pp"] * (out["three_pct"] if np.isfinite(out["three_pct"]) else 0.34) \
        + out["two_pa_pp"] * (out["two_pct"] if np.isfinite(out["two_pct"]) else 0.51)
    out["assist_per_make"] = _safe_div(out["ast_pp"], max(made_fg_pp, 0.05), 0.62)

    return out


def _shrink(obs: float, attempts: float, prior: float, prior_attempts: float) -> float:
    if not np.isfinite(obs):
        obs, attempts = prior, 0.0
    return float((obs * attempts + prior * prior_attempts) / (attempts + prior_attempts))


def build_team_profile(
    df: pd.DataFrame,
    cfg: WeightConfig,
    league_team_logs: Optional[pd.DataFrame] = None,
    game_weights: Optional[Dict[str, float]] = None,
    game_weights_by_stat: Optional[Dict[str, Dict[str, float]]] = None,
    exclude_opponent_abbr: Optional[str] = None,
) -> Tuple[dict, pd.DataFrame]:
    """
    Team profile with:
      - non-overlapping Old / G6-10 / L5 outer weights
      - optional current-rotation similarity INSIDE each bucket
      - opportunity rates from the weighted buckets
      - shooting ability from a larger-sample shrinkage model
      - defensive conversion rates tied to opponent opportunities
    """
    x = df.sort_values("GAME_DATE").copy()
    # Split the ACTUAL timeline first, then remove current-opponent H2H rows
    # inside each bucket. This preserves the meaning of L5/G6-10 while keeping
    # explicit H2H evidence disjoint from the baseline.
    raw_buckets = split_non_overlapping(x)
    if exclude_opponent_abbr and "OPP_ABBR" in x.columns:
        opp = str(exclude_opponent_abbr).upper()
        buckets = {
            k: v[~v["OPP_ABBR"].astype(str).str.upper().eq(opp)].copy()
            for k, v in raw_buckets.items()
        }
    else:
        buckets = raw_buckets
    weights = active_weights(buckets, cfg)
    by = game_weights_by_stat or {}

    feature_stat = {
        # Pace/possessions remain neutral here because the shared pace engine is
        # already the dedicated possession model. Availability changes style,
        # not a second copy of pace.
        "poss_pg": None,
        "three_pa_pp": "3PA", "three_pa_live": "3PA", "three_share": "3PA",
        "two_pa_pp": "FGA", "two_pa_live": "FGA", "fga_live": "FGA",
        "fta_pp": "FTA", "tov_pp": "TOV",
        "oreb_pp": "OREB", "oreb_per_miss": "OREB",
        "dreb_pp": "DREB", "dreb_capture": "DREB",
        "ast_pp": "AST", "assist_per_make": "AST",
        "stl_pp": "STL", "stl_per_opp_tov": "STL",
        "blk_pp": "BLK", "blk_per_opp_2pa": "BLK",
        "pf_pp": "PF",
    }

    cache = {}
    def bucket_feat(bucket_name: str, stat_key: str | None):
        key = (bucket_name, stat_key or "NEUTRAL")
        if key not in cache:
            wm = by.get(stat_key, game_weights) if stat_key else None
            cache[key] = _feat(
                buckets[bucket_name], league_team_logs=league_team_logs, game_weights=wm
            )
        return cache[key]

    p = {}
    neutral_feats = {k: bucket_feat(k, None) for k in buckets}
    for key, stat_key in feature_stat.items():
        feats_for_key = {k: bucket_feat(k, stat_key) for k in buckets}
        p[key] = weighted_average_feature(feats_for_key, weights, key)

    # Keep a neutral audit frame, then expose stat-specific effective sample
    # sizes below. The outer Old/G6-10/L5 weights never change.
    feats = neutral_feats

    # Shooting ability is NOT H2H-blended in v2.11, so keep the full season
    # (including H2H) for the larger-sample shooting-percentage shrinkage.
    # Removing H2H here would throw away information without avoiding any
    # actual double count.
    full = _feat(x, league_team_logs=league_team_logs, game_weights=None)

    # Same philosophy as the player engine: recent shooting is NOT accepted as
    # raw future truth. Use the larger sample and regress by actual attempts.
    # Use the CURRENT loaded league as the shooting prior instead of fixed
    # constants. This avoids baking an NBA/WNBA-era assumption into 3PM/2PM.
    # The team still owns most of its shooting ability; the league prior only
    # regularizes finite-attempt noise.
    if league_team_logs is not None and not league_team_logs.empty:
        lg_full = _feat(league_team_logs, league_team_logs=league_team_logs, game_weights=None)
        lg_three = float(lg_full.get("three_pct", 0.340))
        lg_two = float(lg_full.get("two_pct", 0.510))
        lg_ft = float(lg_full.get("ft_pct", 0.785))
    else:
        lg_three, lg_two, lg_ft = 0.340, 0.510, 0.785

    p["three_pct"] = _shrink(
        full.get("three_pct", np.nan), full.get("three_att", 0.0), lg_three, 45.0
    )
    p["two_pct"] = _shrink(
        full.get("two_pct", np.nan), full.get("two_att", 0.0), lg_two, 70.0
    )
    p["ft_pct"] = _shrink(
        full.get("ft_pct", np.nan), full.get("ft_att", 0.0), lg_ft, 40.0
    )
    p["league_three_pct"] = lg_three
    p["league_two_pct"] = lg_two
    p["league_ft_pct"] = lg_ft

    # Defensive rates need sensible bounds after weighted blending.
    p["dreb_capture"] = float(np.clip(p.get("dreb_capture", 0.94), 0.78, 0.995))
    p["stl_per_opp_tov"] = float(np.clip(p.get("stl_per_opp_tov", 0.55), 0.20, 0.90))
    # Blocks: keep the transparent team-level quantity the user actually wants:
    # how many blocks this team makes per possession. Old/G6-10/L5 + current
    # rotation similarity already live inside p["blk_pp"]. Because BLK is noisy,
    # add only a light 15% league prior; do NOT divide by opponent misses/2PA.
    if league_team_logs is not None and not league_team_logs.empty:
        lg_poss = float(estimate_possessions(league_team_logs).sum())
        lg_blk = float(pd.to_numeric(league_team_logs.get("BLK", 0), errors="coerce").fillna(0).sum())
        league_blk_pp = _safe_div(lg_blk, lg_poss, 0.050)
        lg_fga = float(pd.to_numeric(league_team_logs.get("FGA", 0), errors="coerce").fillna(0).sum())
        lg_3pa = float(pd.to_numeric(league_team_logs.get("FG3A", 0), errors="coerce").fillna(0).sum())
        league_two_pa_pp = _safe_div(max(lg_fga - lg_3pa, 0.0), lg_poss, 0.54)
    else:
        league_blk_pp = 0.050
        league_two_pa_pp = 0.54

    weighted_blk_pp = float(p.get("blk_pp", league_blk_pp))
    p["blk_rate_pp"] = float(np.clip(
        0.85 * weighted_blk_pp + 0.15 * league_blk_pp, 0.015, 0.100
    ))
    p["league_blk_pp"] = float(league_blk_pp)
    p["league_two_pa_pp"] = float(league_two_pa_pp)
    p["assist_per_make"] = float(np.clip(p.get("assist_per_make", 0.62), 0.25, 0.92))

    audit = []
    for k in ("old", "mid", "l5"):
        row = {"bucket": k, "weight": weights[k], **feats.get(k, {"games": 0})}
        row["raw_bucket_games"] = int(len(raw_buckets.get(k, [])))
        row["H2H_excluded"] = int(len(raw_buckets.get(k, [])) - len(buckets.get(k, [])))
        for stat_key in ("FGA", "3PA", "FTA", "TOV", "OREB", "DREB", "AST", "STL", "BLK", "PF"):
            sf = bucket_feat(k, stat_key)
            row[f"effective_games_{stat_key}"] = sf.get("effective_games", len(buckets[k]))
        audit.append(row)
    return p, pd.DataFrame(audit)


def _home_mask(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None

    if "IS_HOME" in df.columns:
        s = df["IS_HOME"]
        if s.dtype == bool:
            return s
        raw = s.astype(str).str.strip().str.lower()
        return raw.isin(["1", "true", "yes", "home", "h"])

    for col in ["HOME_AWAY", "LOCATION", "VENUE"]:
        if col in df.columns:
            raw = df[col].astype(str).str.strip().str.lower()
            return raw.str.startswith("h") | raw.eq("home")

    if "MATCHUP" in df.columns:
        raw = df["MATCHUP"].astype(str)
        return raw.str.contains(r"\bvs\.?\b", case=False, regex=True)

    return None


def team_location_modifiers(
    team_log: pd.DataFrame,
    is_home: bool,
    league_team_logs: Optional[pd.DataFrame] = None,
    min_games: int = 5,
    exclude_opponent_abbr: Optional[str] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Small, data-driven home/away correction -- never a generic away penalty.

    Research on basketball home advantage is much more consistent for scoring
    efficiency than for tactical opportunity counts such as possessions and
    2PA/3PA attempts.  v2.14 therefore does NOT assume "away teams shoot worse".

    For each stat we estimate the team's location-vs-overall deviation and shrink
    it heavily toward the WNBA-wide location-vs-overall deviation in the loaded
    data.  If the data show almost no effect, the modifier is almost exactly 1.
    Tactical opportunity variables are capped more tightly than shooting
    efficiency / foul-related variables.
    """
    neutral = {
        "FGA": 1.0, "3P_SHARE": 1.0,
        "3PA": 1.0, "2PA": 1.0, "FTA": 1.0, "TOV": 1.0,
        "OREB": 1.0, "AST": 1.0, "PF": 1.0,
        "DREB": 1.0, "STL": 1.0, "BLK": 1.0,
        "3P_PCT": 1.0, "2P_PCT": 1.0,
    }
    mask = _home_mask(team_log)
    if mask is None:
        return neutral, pd.DataFrame([{"Location": "unavailable", "Games": 0}])

    base_log = team_log.copy()
    split = team_log[mask if is_home else ~mask].copy()
    if exclude_opponent_abbr and "OPP_ABBR" in team_log.columns:
        opp = str(exclude_opponent_abbr).upper()
        base_log = base_log[~base_log["OPP_ABBR"].astype(str).str.upper().eq(opp)].copy()
        split = split[~split["OPP_ABBR"].astype(str).str.upper().eq(opp)].copy()
    if len(split) < min_games:
        return neutral, pd.DataFrame([{
            "Location": "home" if is_home else "away",
            "Games": int(len(split)),
            "Note": f"< {min_games} games; neutral location modifier",
        }])

    base = _feat(base_log, league_team_logs=league_team_logs)
    loc = _feat(split, league_team_logs=league_team_logs)

    # League-wide empirical location prior.  This is the only direction prior;
    # there is no hard-coded home boost or away shooting penalty.
    lg_base = lg_loc = {}
    if league_team_logs is not None and not league_team_logs.empty:
        lgmask = _home_mask(league_team_logs)
        if lgmask is not None:
            lg_base = _feat(league_team_logs, league_team_logs=league_team_logs)
            lg_split = league_team_logs[lgmask if is_home else ~lgmask].copy()
            if len(lg_split) >= 20:
                lg_loc = _feat(lg_split, league_team_logs=league_team_logs)

    mapping = {
        "FGA": "fga_live",
        "3P_SHARE": "three_share",
        "3PA": "three_pa_live",   # audit compatibility only
        "2PA": "two_pa_live",     # audit compatibility only
        "FTA": "fta_pp",
        "TOV": "tov_pp",
        "OREB": "oreb_per_miss",
        "AST": "assist_per_make",
        "PF": "pf_pp",
        "DREB": "dreb_capture",
        "STL": "stl_per_opp_tov",
        "BLK": "blk_pp",
        "3P_PCT": "three_pct",
        "2P_PCT": "two_pct",
    }
    prob_keys = {"3P_SHARE", "3P_PCT", "2P_PCT", "DREB", "STL"}
    # Tactical choices receive minimal location weight; effectiveness can move a
    # little more if the actual WNBA/team split supports it.
    caps = {
        "FGA": 0.015, "3P_SHARE": 0.015, "3PA": 0.015, "2PA": 0.015,
        "FTA": 0.025, "TOV": 0.020, "OREB": 0.020, "AST": 0.020,
        "PF": 0.025, "DREB": 0.020, "STL": 0.020, "BLK": 0.025,
        "3P_PCT": 0.030, "2P_PCT": 0.030,
    }

    def tlog(v, prob=False):
        if prob:
            v = float(np.clip(v, 1e-5, 1 - 1e-5))
            return float(np.log(v / (1 - v)))
        return float(np.log(max(float(v), 1e-5)))

    rows = []
    out = dict(neutral)
    n = int(len(split))
    team_conf = float(np.clip(n / (n + 24.0), 0.0, 1.0))

    for market, key in mapping.items():
        b = float(base.get(key, np.nan))
        l = float(loc.get(key, np.nan))
        lb = float(lg_base.get(key, np.nan)) if lg_base else np.nan
        ll = float(lg_loc.get(key, np.nan)) if lg_loc else np.nan
        prob = market in prob_keys

        if not (np.isfinite(b) and b > 0 and np.isfinite(l) and l > 0):
            out[market] = 1.0
            continue

        team_effect = tlog(l, prob) - tlog(b, prob)
        if np.isfinite(lb) and lb > 0 and np.isfinite(ll) and ll > 0:
            league_effect = tlog(ll, prob) - tlog(lb, prob)
        else:
            league_effect = 0.0

        # Empirical Bayes: small team samples are pulled strongly to the WNBA
        # location effect; a large stable team split can express more of itself.
        blended_effect = team_conf * team_effect + (1.0 - team_conf) * league_effect

        if prob:
            target = 1.0 / (1.0 + np.exp(-(tlog(b, True) + blended_effect)))
            mod = target / b
        else:
            mod = float(np.exp(blended_effect))

        cap = float(caps[market])
        mod = float(np.clip(mod, 1.0 - cap, 1.0 + cap))
        out[market] = mod
        rows.append({
            "Stat": market,
            "Overall": b,
            "Location": l,
            "Team raw ratio": (l / b) if b > 0 else np.nan,
            "League location ratio": (ll / lb) if np.isfinite(lb) and lb > 0 and np.isfinite(ll) else np.nan,
            "Team location confidence": team_conf,
            "Applied modifier": mod,
            "Games": n,
            "Rule": "data-driven; no hard-coded home/away direction",
        })

    return out, pd.DataFrame(rows)

def h2h_team_audit(
    league_team_logs: pd.DataFrame,
    home_abbr: str,
    away_abbr: str,
) -> pd.DataFrame:
    """Same-season H2H audit only. No extra numerical weight is applied."""
    x = league_team_logs.copy()
    mask = (
        x["TEAM_ABBR"].astype(str).str.upper().isin([str(home_abbr).upper(), str(away_abbr).upper()])
        & x["OPP_ABBR"].astype(str).str.upper().isin([str(home_abbr).upper(), str(away_abbr).upper()])
    )
    h = x[mask].copy()
    if h.empty:
        return pd.DataFrame()
    cols = [
        c for c in [
            "GAME_DATE", "GAME_ID", "TEAM_ABBR", "OPP_ABBR", "OT_FLAG", "OT_COUNT", "GAME_LENGTH_MIN", "PTS", "FGA",
            "FG3A", "FG3M", "FTA", "FTM", "OREB", "DREB", "REB", "AST",
            "STL", "BLK", "TOV", "PF",
        ] if c in h.columns
    ]
    return h[cols].sort_values(["GAME_DATE", "TEAM_ABBR"], ascending=[False, True]).reset_index(drop=True)


def h2h_profile_blend(
    league_team_logs: pd.DataFrame,
    team_abbr: str,
    opponent_abbr: str,
    base_profile: dict,
    rotation_similarity: float = 1.0,
    max_weight: float = 0.10,
    skip_features: Optional[set[str]] = None,
) -> Tuple[dict, pd.DataFrame]:
    """Legacy disjoint H2H blend retained for compatibility/audit.

    v2.18 calls this with every structural feature in ``skip_features`` so no
    fixed percentage H2H weight reaches production Team Markets. Supported
    rates receive H2H only through the chronologically validated residual model
    in ``core.structural_calibration``; unsupported rates keep H2H audit-only.

    The old 0.20*N/(N+2), capped at 10%, code path is kept only so older tests
    or external callers do not break. It is not used by the v2.18 Streamlit
    production path.
    """
    out = dict(base_profile)
    if league_team_logs is None or league_team_logs.empty:
        return out, pd.DataFrame()

    x = league_team_logs.copy()
    mask = (
        x["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())
        & x["OPP_ABBR"].astype(str).str.upper().eq(str(opponent_abbr).upper())
    )
    h = x[mask].copy()
    if h.empty:
        return out, pd.DataFrame([{
            "H2H games": 0, "Rotation similarity": float(rotation_similarity),
            "Applied H2H weight": 0.0,
        }])

    hfeat = _feat(h, league_team_logs=league_team_logs, game_weights=None)
    n = int(len(h))
    sim = float(np.clip(rotation_similarity, 0.0, 1.0))
    w = min(float(max_weight), 0.20 * (n / (n + 2.0)) * sim)

    mapping = [
        ("fga_live", "fga_live"),
        ("three_share", "three_share"),
        ("fta_pp", "fta_pp"),
        ("tov_pp", "tov_pp"),
        ("oreb_per_miss", "oreb_per_miss"),
        ("assist_per_make", "assist_per_make"),
        ("pf_pp", "pf_pp"),
        ("dreb_capture", "dreb_capture"),
        ("stl_per_opp_tov", "stl_per_opp_tov"),
        ("blk_rate_pp", "blk_pp"),
    ]
    skip = set(skip_features or set())
    rows = []
    for target_key, h2h_key in mapping:
        if target_key in skip:
            rows.append({
                "Feature": target_key,
                "Base non-H2H": float(base_profile.get(target_key, np.nan)),
                "H2H value": float(hfeat.get(h2h_key, np.nan)),
                "H2H games": n,
                "Rotation similarity": sim,
                "Applied H2H weight": 0.0,
                "Final": float(base_profile.get(target_key, np.nan)),
                "Reason": "v2.18 production: fixed H2H weight disabled; residual model or audit-only",
            })
            continue
        b = float(base_profile.get(target_key, np.nan))
        hv = float(hfeat.get(h2h_key, np.nan))
        if not (np.isfinite(b) and b > 0 and np.isfinite(hv) and hv > 0 and w > 0):
            applied = b
        elif target_key in {"dreb_capture", "stl_per_opp_tov"}:
            applied = (1.0 - w) * b + w * hv
        else:
            applied = float(np.exp((1.0 - w) * np.log(b) + w * np.log(hv)))
        if np.isfinite(applied):
            out[target_key] = float(applied)
        rows.append({
            "Feature": target_key,
            "Base non-H2H": b,
            "H2H value": hv,
            "H2H games": n,
            "Rotation similarity": sim,
            "Applied H2H weight": w,
            "Final": applied,
        })

    # Keep legacy derived shot fields coherent for audits/backward code paths.
    if np.isfinite(out.get("fga_live", np.nan)) and np.isfinite(out.get("three_share", np.nan)):
        out["three_pa_live"] = out["fga_live"] * out["three_share"]
        out["two_pa_live"] = out["fga_live"] * (1.0 - out["three_share"])
    return out, pd.DataFrame(rows)


def h2h_block_modifier(
    league_team_logs: pd.DataFrame,
    team_abbr: str,
    opponent_abbr: str,
    base_blk_pp: float,
) -> Tuple[float, pd.DataFrame]:
    """
    Tiny same-season H2H correction for blocks only.

    H2H is already contained in Old/G6-10/L5, so it is NOT another sample.
    We use the H2H/base rate ratio with very small exponent and a +/-3% cap.
    One H2H game therefore cannot meaningfully move the projection.
    """
    if league_team_logs is None or league_team_logs.empty:
        return 1.0, pd.DataFrame()

    x = league_team_logs.copy()
    mask = (
        x["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())
        & x["OPP_ABBR"].astype(str).str.upper().eq(str(opponent_abbr).upper())
    )
    h = x[mask].copy()
    if h.empty:
        return 1.0, pd.DataFrame([{"H2H games": 0, "Applied BLK H2H modifier": 1.0}])

    poss = float(estimate_possessions(h).sum())
    blk = float(pd.to_numeric(h["BLK"], errors="coerce").fillna(0).sum())
    h2h_rate = _safe_div(blk, poss, base_blk_pp)
    ratio = _safe_div(h2h_rate, max(base_blk_pp, 1e-9), 1.0)
    ratio = float(np.clip(ratio, 0.60, 1.40))
    n_games = int(len(h))
    confidence = float(n_games / (n_games + 3.0))
    # Max exponent 0.08, plus hard +/-3% cap. This is intentionally tiny.
    mod = float(np.clip(ratio ** (0.08 * confidence), 0.97, 1.03))
    audit = pd.DataFrame([{
        "H2H games": n_games,
        "Base BLK/poss": float(base_blk_pp),
        "H2H BLK/poss": float(h2h_rate),
        "Raw H2H/base ratio": ratio,
        "H2H confidence": confidence,
        "Applied BLK H2H modifier": mod,
    }])
    return mod, audit


def _simulate_offense(
    profile: dict,
    ctx: TeamContext,
    poss: np.ndarray,
    rng: np.random.Generator,
    z_style: np.ndarray,
    z_shoot: np.ndarray,
    z_foul: np.ndarray,
    z_tov: np.ndarray,
    z_reb: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Simulate one offense with an approximately possession-consistent event chain.

    v2.16 key change:
        possessions -> TOV + FT-possession share -> initial shot endings
        -> offensive-rebound recycling -> total FGA -> 3P share -> makes

    The old v2.15 chain used ``FGA_LIVE`` directly on (POSS-TOV) and then also
    generated FTA and OREB downstream.  Because the historical possession
    estimator already contains ``-OREB + 0.44*FTA``, that construction could
    systematically produce too few FGA for the stated pace.  The new chain
    uses the same possession identity in expectation:

        POSS ~= FGA - OREB + TOV + 0.44*FTA

    A small residual ``ctx.fga`` elasticity is retained, but strongly shrunk so
    matchup/roster evidence can nudge shot volume without becoming a second pace.
    """
    poss_i = np.maximum(poss.astype(int), 1)

    tov_rate = np.clip(
        profile["tov_pp"] * ctx.tov * np.exp(0.08 * z_tov - 0.5 * 0.08**2),
        0.03, 0.30,
    )
    tov = rng.binomial(poss_i, tov_rate)

    # Free throws consume possession mass in the same 0.44 convention used by
    # the historical possession estimator.  Draw them before FGA so the two
    # opportunity channels reconcile rather than overlap.
    # v2.17: structural FTA rate may move materially when the walk-forward
    # model finds a real matchup signal.  Bound the RATE by physical historical
    # plausibility here rather than clipping the learned matchup modifier upstream.
    fta_rate = np.clip(
        profile["fta_pp"] * ctx.fta * np.exp(0.12 * z_foul - 0.5 * 0.12**2),
        0.05, 0.55,
    )
    fta = rng.poisson(np.clip(poss * fta_rate, 0.001, None))

    base_share = float(profile.get(
        "three_share",
        profile.get("three_pa_live", 0.35) /
        max(profile.get("three_pa_live", 0.35) + profile.get("two_pa_live", 0.55), 1e-9),
    ))
    base_share = float(np.clip(base_share * ctx.three_share, 0.08, 0.72))
    logit = np.log(base_share / max(1.0 - base_share, 1e-9))
    p3_share = 1.0 / (1.0 + np.exp(-(logit + 0.18 * z_style)))
    p3_share = np.clip(p3_share, 0.06, 0.75)

    p3 = np.clip(profile["three_pct"] * ctx.three_pct + 0.03 * z_shoot, 0.10, 0.60)
    p2 = np.clip(profile["two_pct"] * ctx.two_pct + 0.025 * z_shoot, 0.25, 0.75)
    pft = np.clip(profile["ft_pct"] + 0.01 * z_shoot, 0.45, 0.98)

    oreb_share = np.clip(
        profile.get("oreb_per_miss", 0.25)
        * ctx.oreb
        * np.exp(0.10 * z_reb - 0.5 * 0.10**2),
        0.08, 0.45,
    )

    # Possessions that can terminate in an initial field-goal attempt after
    # turnovers and the free-throw possession component are removed.
    initial_shot_endings = np.maximum(
        poss.astype(float) - tov.astype(float) - 0.44 * fta.astype(float),
        0.25,
    )
    miss_rate = np.clip(p3_share * (1.0 - p3) + (1.0 - p3_share) * (1.0 - p2), 0.20, 0.80)
    recycle_prob = np.clip(miss_rate * oreb_share, 0.01, 0.32)
    recycle_factor = 1.0 / np.maximum(1.0 - recycle_prob, 0.68)

    # Residual FGA context only: FGA is now primarily an identity consequence of
    # pace/TOV/FTA/OREB, not an independent full-strength opportunity multiplier.
    fga_residual = float(np.clip(ctx.fga, 0.90, 1.10)) ** 0.35
    fga_mean = np.clip(initial_shot_endings * recycle_factor * fga_residual, 0.001, None)
    fga = rng.poisson(fga_mean)

    a3 = rng.binomial(fga, p3_share)
    a2 = fga - a3

    m3 = rng.binomial(a3, p3)
    m2 = rng.binomial(a2, p2)
    ftm = rng.binomial(fta, pft)
    fgm = m3 + m2
    pts = 3 * m3 + 2 * m2 + ftm

    misses3 = np.maximum(a3 - m3, 0)
    misses2 = np.maximum(a2 - m2, 0)
    misses = misses3 + misses2
    oreb = rng.binomial(misses, oreb_share)

    assist_per_make = float(np.clip(profile.get("assist_per_make", 0.62), 0.25, 0.92))
    ast_prob = np.clip(assist_per_make * ctx.ast * (1.0 + 0.07 * z_shoot), 0.20, 0.95)
    ast = rng.binomial(fgm, ast_prob)

    pf = rng.poisson(np.clip(
        poss * profile["pf_pp"] * ctx.pf * np.exp(0.10 * z_foul - 0.5 * 0.10**2),
        0.001, None,
    ))

    return {
        "POSS": poss,
        "TOV": tov,
        "3PA": a3,
        "3PM": m3,
        "2PA": a2,
        "2PM": m2,
        "FGA": fga,
        "FGM": fgm,
        "FTA": fta,
        "FTM": ftm,
        "OREB": oreb,
        "AST": ast,
        "PF": pf,
        "PTS": pts,
        "MISSES": misses,
        "3MISS": misses3,
        "2MISS": misses2,
    }

def simulate_game(
    home_profile: dict,
    away_profile: dict,
    home_ctx: TeamContext,
    away_ctx: TeamContext,
    n: int = 50_000,
    seed: int = 3,
    opportunity_mult: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Coupled two-team game simulation.

    Key conservation / dependency rules:
      - both teams share the same game-possession state
      - FGA = 2PA + 3PA
      - PTS = 2*2PM + 3*3PM + FTM
      - OREB is sampled from OWN misses
      - DREB is sampled from OPPONENT misses not already rebounded offensively
      - STL is a subset of OPPONENT turnovers
      - BLK starts from OWN non-H2H weighted BLK/poss, with disjoint H2H folded
        into the profile, opponent block-susceptibility, optional positional
        susceptibility, and only a tiny opponent-2PA opportunity nudge
      - BLK <= opponent total missed FGA (three-point shots can be blocked too)
      - REB = OREB + DREB

    This removes the old same-team STL/TOV and BLK/2PA direction errors and
    enables coherent totals / "team with most" markets.
    """
    rng = np.random.default_rng(seed)

    central = 0.5 * (float(home_ctx.projected_possessions) + float(away_ctx.projected_possessions))
    sd = 0.5 * (float(home_ctx.possessions_sd) + float(away_ctx.possessions_sd))
    poss = np.rint(np.clip(rng.normal(central, sd, n), 55, 115) * opportunity_mult).astype(int)
    poss = np.maximum(poss, 1)

    # Shared game environments plus team-specific deviations.
    z_game_style = rng.normal(size=n)
    z_game_shoot = rng.normal(size=n)
    z_game_foul = rng.normal(size=n)
    z_game_tov = rng.normal(size=n)
    z_game_reb = rng.normal(size=n)
    z_game_blk = rng.normal(size=n)

    def blend(shared, scale=0.70):
        return scale * shared + np.sqrt(max(1.0 - scale**2, 0.0)) * rng.normal(size=n)

    h = _simulate_offense(
        home_profile, home_ctx, poss, rng,
        blend(z_game_style), blend(z_game_shoot, 0.45), blend(z_game_foul),
        blend(z_game_tov, 0.55), blend(z_game_reb, 0.55),
    )
    a = _simulate_offense(
        away_profile, away_ctx, poss, rng,
        blend(z_game_style), blend(z_game_shoot, 0.45), blend(z_game_foul),
        blend(z_game_tov, 0.55), blend(z_game_reb, 0.55),
    )

    # Defensive rebound conversion from the OPPONENT'S remaining missed shots.
    h_dreb_chances = np.maximum(a["MISSES"] - a["OREB"], 0)
    a_dreb_chances = np.maximum(h["MISSES"] - h["OREB"], 0)

    h_dreb_p = np.clip(
        home_profile.get("dreb_capture", 0.94) * home_ctx.dreb
        * np.exp(0.035 * blend(z_game_reb, 0.50) - 0.5 * 0.035**2),
        0.72, 0.995,
    )
    a_dreb_p = np.clip(
        away_profile.get("dreb_capture", 0.94) * away_ctx.dreb
        * np.exp(0.035 * blend(z_game_reb, 0.50) - 0.5 * 0.035**2),
        0.72, 0.995,
    )
    h_dreb = rng.binomial(h_dreb_chances, h_dreb_p)
    a_dreb = rng.binomial(a_dreb_chances, a_dreb_p)

    # Steals are a subset of OPPONENT turnovers.
    h_stl_p = np.clip(
        home_profile.get("stl_per_opp_tov", 0.55) * home_ctx.stl
        * np.exp(0.05 * blend(z_game_tov, 0.45) - 0.5 * 0.05**2),
        0.10, 0.95,
    )
    a_stl_p = np.clip(
        away_profile.get("stl_per_opp_tov", 0.55) * away_ctx.stl
        * np.exp(0.05 * blend(z_game_tov, 0.45) - 0.5 * 0.05**2),
        0.10, 0.95,
    )
    h_stl = rng.binomial(a["TOV"], h_stl_p)
    a_stl = rng.binomial(h["TOV"], a_stl_p)

    # Blocks -- v2.11:
    #   own ability (base profile, with DISJOINT H2H already blended in)
    # x opponent's tendency to be blocked (ctx.blk)
    # x optional RELATIVE positional susceptibility (ctx.blk_position)
    # x tiny 2PA-opportunity adjustment (max +/-2%)
    # x pace automatically through possessions.
    # 2PA is deliberately NOT the main denominator.
    h_lg_2pa = max(float(home_profile.get("league_two_pa_pp", 0.54)), 0.10)
    a_lg_2pa = max(float(away_profile.get("league_two_pa_pp", 0.54)), 0.10)
    a_2pa_rate = a["2PA"] / np.maximum(poss, 1)
    h_2pa_rate = h["2PA"] / np.maximum(poss, 1)
    h_2pa_mod = np.clip((a_2pa_rate / h_lg_2pa) ** 0.07, 0.98, 1.02)
    a_2pa_mod = np.clip((h_2pa_rate / a_lg_2pa) ** 0.07, 0.98, 1.02)

    h_blk_noise = np.exp(0.10 * blend(z_game_blk, 0.55) - 0.5 * 0.10**2)
    a_blk_noise = np.exp(0.10 * blend(z_game_blk, 0.55) - 0.5 * 0.10**2)
    h_blk_mean = np.clip(
        poss * home_profile.get("blk_rate_pp", home_profile.get("blk_pp", 0.05))
        * home_ctx.blk * home_ctx.blk_position * h_2pa_mod * h_blk_noise,
        0.001, None,
    )
    a_blk_mean = np.clip(
        poss * away_profile.get("blk_rate_pp", away_profile.get("blk_pp", 0.05))
        * away_ctx.blk * away_ctx.blk_position * a_2pa_mod * a_blk_noise,
        0.001, None,
    )
    h_blk_candidate = rng.poisson(h_blk_mean)
    a_blk_candidate = rng.poisson(a_blk_mean)
    # Three-point attempts can also be blocked. Total missed FGA is the correct
    # box-score conservation cap; the old 2MISS-only cap was structurally wrong.
    h_blk = np.minimum(h_blk_candidate, a["MISSES"])
    a_blk = np.minimum(a_blk_candidate, h["MISSES"])

    h["DREB"] = h_dreb
    a["DREB"] = a_dreb
    h["REB"] = h["OREB"] + h_dreb
    a["REB"] = a["OREB"] + a_dreb
    h["STL"] = h_stl
    a["STL"] = a_stl
    h["BLK"] = h_blk
    a["BLK"] = a_blk

    keep = [
        "POSS", "PTS", "FGM", "FGA", "3PM", "3PA", "2PM", "2PA",
        "FTM", "FTA", "REB", "OREB", "DREB", "AST", "STL", "BLK",
        "TOV", "PF",
    ]
    home = pd.DataFrame({k: h[k] for k in keep})
    away = pd.DataFrame({k: a[k] for k in keep})
    return home, away


def simulate_team(
    profile: dict,
    ctx: TeamContext,
    n: int = 50_000,
    seed: int = 3,
    opportunity_mult: float = 1.0,
) -> pd.DataFrame:
    """
    Legacy single-team wrapper retained for backward compatibility.

    New Team Markets should use simulate_game(), because REB/DREB/STL/BLK and
    relative/total markets require the opponent side. Defensive outputs from
    this wrapper are intentionally omitted rather than fabricated from the
    team's own opportunities.
    """
    rng = np.random.default_rng(seed)
    poss = np.rint(
        np.clip(rng.normal(ctx.projected_possessions, ctx.possessions_sd, n), 55, 115)
        * opportunity_mult
    ).astype(int)
    z = [rng.normal(size=n) for _ in range(5)]
    o = _simulate_offense(profile, ctx, poss, rng, *z)
    keep = [
        "POSS", "PTS", "FGM", "FGA", "3PM", "3PA", "2PM", "2PA",
        "FTM", "FTA", "OREB", "AST", "TOV", "PF",
    ]
    return pd.DataFrame({k: o[k] for k in keep})
