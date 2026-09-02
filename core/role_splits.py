from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def _ci_lookup_name(df: pd.DataFrame, name: str):
    if df is None or df.empty or "PLAYER_NAME" not in df.columns:
        return None
    target = str(name).strip().casefold()
    hit = df[df["PLAYER_NAME"].astype(str).str.strip().str.casefold() == target]
    return hit.iloc[0] if not hit.empty else None


def current_out_teammates(
    manual_context: dict,
    player_pool: pd.DataFrame,
    team_abbr: str,
    focal_player: str,
) -> List[str]:
    """Confirmed OUT teammates from trader context, resolved to the current team."""
    injuries = (manual_context or {}).get("injuries", {})
    if not isinstance(injuries, dict):
        return []

    out = []
    focal_cf = str(focal_player).strip().casefold()
    team_u = str(team_abbr).upper()

    for name, info in injuries.items():
        if str(name).strip().casefold() == focal_cf:
            continue
        if not isinstance(info, dict):
            continue
        if str(info.get("status", "")).strip().upper() != "OUT":
            continue

        info_team = str(info.get("team", "")).strip().upper()
        row = _ci_lookup_name(player_pool, str(name))
        pool_team = str(row.get("TEAM_ABBR", "")).upper() if row is not None else ""

        if (info_team and info_team == team_u) or (pool_team and pool_team == team_u):
            out.append(str(name))

    return sorted(set(out))


def same_role_game_weights(
    focal_log: pd.DataFrame,
    player_db: pd.DataFrame,
    team_abbr: str,
    absent_teammates: List[str],
    enabled: bool = True,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Reweight historical focal-player games toward the current teammate-absence state.

    IMPORTANT: this is an INNER weight inside Old/G6-10/L5. It is not a fourth
    sample and therefore does not double-count the recent buckets.

    Sparse same-role samples are regularized: with only a few matching games the
    weight shift is deliberately small; it grows gradually with sample size.
    """
    if focal_log is None or focal_log.empty or not enabled or not absent_teammates:
        audit = pd.DataFrame([{
            "Enabled": bool(enabled),
            "Absent teammates": ", ".join(absent_teammates or []),
            "Same-role games": 0,
            "Total focal games": int(len(focal_log)) if focal_log is not None else 0,
            "Confidence": 0.0,
            "Match weight": 1.0,
            "Mismatch weight": 1.0,
            "Note": "Neutral: no confirmed OUT teammate state to match.",
        }])
        return {}, audit

    team_u = str(team_abbr).upper()
    out_cf = {str(x).strip().casefold() for x in absent_teammates}

    pdb = player_db.copy()
    pdb = pdb[
        pdb["TEAM_ABBR"].astype(str).str.upper().eq(team_u)
        & pdb["PLAYER_NAME"].astype(str).str.strip().str.casefold().isin(out_cf)
    ].copy()
    if "MIN" in pdb.columns:
        pdb = pdb[pd.to_numeric(pdb["MIN"], errors="coerce").fillna(0) > 0]

    teammate_played = set(pdb["GAME_ID"].astype(str)) if not pdb.empty else set()

    x = focal_log.copy()
    gids = x["GAME_ID"].astype(str)
    same_role = ~gids.isin(teammate_played)
    n_total = int(len(x))
    n_match = int(same_role.sum())

    # Regularization inspired by lineup models: sparse splits get only a mild
    # adjustment. At 2 matching games conf=.25; 6 -> .50; 12 -> .67.
    confidence = float(n_match / (n_match + 6.0)) if n_match >= 2 else 0.0
    match_w = 1.0 + 0.40 * confidence
    mismatch_w = 1.0 - 0.18 * confidence

    raw = np.where(same_role.to_numpy(), match_w, mismatch_w).astype(float)
    if raw.size and np.mean(raw) > 0:
        raw = raw / np.mean(raw)  # preserve the outer bucket scale

    weights = {str(gid): float(w) for gid, w in zip(gids, raw)}

    def _mean(col, mask):
        if col not in x.columns or int(mask.sum()) == 0:
            return np.nan
        return float(pd.to_numeric(x.loc[mask, col], errors="coerce").mean())

    audit = pd.DataFrame([{
        "Enabled": True,
        "Absent teammates": ", ".join(absent_teammates),
        "Same-role games": n_match,
        "Total focal games": n_total,
        "Confidence": confidence,
        "Match weight": float(match_w),
        "Mismatch weight": float(mismatch_w),
        "Same-role MIN": _mean("MIN", same_role),
        "Same-role PTS": _mean("PTS", same_role),
        "Same-role REB": _mean("REB", same_role),
        "Same-role AST": _mean("AST", same_role),
        "Same-role 3PA": _mean("FG3A", same_role),
        "Same-role FTA": _mean("FTA", same_role),
        "Note": (
            "Applied inside Old/G6-10/L5 only."
            if confidence > 0 else
            "Fewer than 2 matching games; neutral weighting."
        ),
    }])
    return weights, audit
