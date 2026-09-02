from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from core.buckets import WeightConfig, split_non_overlapping, active_weights
from core.team_model import estimate_possessions
from core.exposure import regulation_equivalent_factor


@dataclass
class PaceProjection:
    home_pace: float
    away_pace: float
    league_pace: float
    fast_weight: float
    slow_weight: float
    central: float
    sd: float
    low: float
    high: float
    calibration_games: int
    rmse: float


def _weighted_team_pace(team_log: pd.DataFrame, cfg: WeightConfig) -> float:
    x = team_log.sort_values("GAME_DATE").copy()
    x["POSS_RAW"] = estimate_possessions(x)
    x["POSS"] = x["POSS_RAW"] * regulation_equivalent_factor(x)
    buckets = split_non_overlapping(x)
    outer = active_weights(buckets, cfg)

    vals, ws = [], []
    for key in ("old", "mid", "l5"):
        b = buckets[key]
        if not b.empty and outer.get(key, 0) > 0:
            vals.append(float(b["POSS"].mean()))
            ws.append(float(outer[key]))
    if not vals:
        return float(x["POSS"].mean())
    return float(np.average(vals, weights=ws))


def _build_pace_calibration(team_db: pd.DataFrame) -> pd.DataFrame:
    x = team_db.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBR"]).copy()
    x["POSS_RAW"] = estimate_possessions(x)
    x["POSS"] = x["POSS_RAW"] * regulation_equivalent_factor(x)

    prior_parts = []
    for team, g in x.groupby("TEAM_ABBR", sort=False):
        g = g.sort_values("GAME_DATE").copy()
        # Pregame pace estimate uses only games already completed.
        g["PRIOR_PACE"] = g["POSS"].expanding(min_periods=5).mean().shift(1)
        prior_parts.append(g)
    x = pd.concat(prior_parts, ignore_index=True)

    rows = []
    for gid, g in x.groupby("GAME_ID"):
        if len(g) < 2:
            continue
        g = g.dropna(subset=["PRIOR_PACE"])
        if len(g) < 2:
            continue
        p = sorted(g["PRIOR_PACE"].astype(float).tolist())
        actual = float(g["POSS"].astype(float).mean())
        rows.append({
            "GAME_ID": gid,
            "slow": p[0],
            "fast": p[-1],
            "actual": actual,
        })
    return pd.DataFrame(rows)


def fit_pace_control(team_db: pd.DataFrame):
    cal = _build_pace_calibration(team_db)
    all_poss = estimate_possessions(team_db) * regulation_equivalent_factor(team_db)
    league = float(all_poss.mean())

    default = {
        "league_pace": league,
        "fast_weight": 0.57,
        "slow_weight": 0.43,
        "rmse": 3.0,
        "n": 0,
    }
    if len(cal) < 30:
        return default

    # Center around league pace and ridge-shrink toward a mild fast-control
    # prior rather than hardcoding the observed intuition.
    X = np.column_stack([
        cal["fast"].to_numpy() - league,
        cal["slow"].to_numpy() - league,
    ])
    y = cal["actual"].to_numpy() - league

    prior = np.asarray([0.57, 0.43], dtype=float)
    lam = 45.0
    A = X.T @ X + lam * np.eye(2)
    b = X.T @ y + lam * prior
    coef = np.linalg.solve(A, b)

    # Interpret as relative control weights; preserve mild stability.
    coef = np.clip(coef, [0.30, 0.20], [0.80, 0.70])
    s = float(coef.sum())
    if s <= 0:
        coef = prior
    else:
        coef = coef / s

    pred = league + X @ coef
    rmse = float(np.sqrt(np.mean((cal["actual"].to_numpy() - pred) ** 2)))
    rmse = float(np.clip(rmse, 2.0, 5.5))

    return {
        "league_pace": league,
        "fast_weight": float(coef[0]),
        "slow_weight": float(coef[1]),
        "rmse": rmse,
        "n": int(len(cal)),
    }


def project_game_pace(
    team_db: pd.DataFrame,
    home_abbr: str,
    away_abbr: str,
    cfg: WeightConfig | None = None,
) -> PaceProjection:
    cfg = cfg or WeightConfig.stable()

    home_log = team_db[
        team_db["TEAM_ABBR"].astype(str).str.upper()
        == str(home_abbr).upper()
    ].copy()
    away_log = team_db[
        team_db["TEAM_ABBR"].astype(str).str.upper()
        == str(away_abbr).upper()
    ].copy()

    home = _weighted_team_pace(home_log, cfg)
    away = _weighted_team_pace(away_log, cfg)

    fit = fit_pace_control(team_db)
    league = fit["league_pace"]

    fast = max(home, away)
    slow = min(home, away)

    central = (
        league
        + fit["fast_weight"] * (fast - league)
        + fit["slow_weight"] * (slow - league)
    )
    central = float(np.clip(central, 75.0, 115.0))
    sd = fit["rmse"]

    return PaceProjection(
        home_pace=home,
        away_pace=away,
        league_pace=league,
        fast_weight=fit["fast_weight"],
        slow_weight=fit["slow_weight"],
        central=central,
        sd=sd,
        low=float(np.clip(central - sd, 70.0, 120.0)),
        high=float(np.clip(central + sd, 70.0, 120.0)),
        calibration_games=fit["n"],
        rmse=fit["rmse"],
    )


def player_historical_pace_environment(
    player_log: pd.DataFrame,
    team_db: pd.DataFrame,
    cfg: WeightConfig,
) -> float:
    """
    Player rates are per-minute, so convert today's projected possessions into
    a multiplier relative to the pace environment already embedded in the
    player's historical minutes. Uses the same non-overlapping bucket protocol.
    """
    if player_log.empty:
        return float((estimate_possessions(team_db) * regulation_equivalent_factor(team_db)).mean())

    team_rows = team_db[
        team_db["GAME_ID"].astype(str).isin(
            player_log["GAME_ID"].astype(str)
        )
    ].copy()

    # Prefer the player's own team row in each game.
    player_team_by_game = (
        player_log[["GAME_ID", "TEAM_ABBR"]]
        .drop_duplicates()
        .assign(GAME_ID=lambda d: d["GAME_ID"].astype(str))
    )
    team_rows["GAME_ID"] = team_rows["GAME_ID"].astype(str)
    merged = team_rows.merge(
        player_team_by_game,
        on="GAME_ID",
        suffixes=("", "_PLAYER")
    )
    merged = merged[
        merged["TEAM_ABBR"].astype(str).str.upper()
        == merged["TEAM_ABBR_PLAYER"].astype(str).str.upper()
    ].copy()

    if merged.empty:
        return float((estimate_possessions(team_db) * regulation_equivalent_factor(team_db)).mean())

    merged["POSS_RAW"] = estimate_possessions(merged)
    merged["POSS"] = merged["POSS_RAW"] * regulation_equivalent_factor(merged)
    dates = player_log[["GAME_ID", "GAME_DATE"]].drop_duplicates().copy()
    dates["GAME_ID"] = dates["GAME_ID"].astype(str)
    merged = merged.merge(dates, on="GAME_ID", how="left", suffixes=("", "_P"))
    date_col = "GAME_DATE_P" if "GAME_DATE_P" in merged.columns else "GAME_DATE"
    merged["GAME_DATE"] = pd.to_datetime(merged[date_col], errors="coerce")

    buckets = split_non_overlapping(merged[["GAME_ID", "GAME_DATE", "POSS"]])
    outer = active_weights(buckets, cfg)

    vals, ws = [], []
    for key in ("old", "mid", "l5"):
        b = buckets[key]
        if not b.empty and outer.get(key, 0) > 0:
            vals.append(float(b["POSS"].mean()))
            ws.append(float(outer[key]))

    if not vals:
        return float(merged["POSS"].mean())
    return float(np.average(vals, weights=ws))
