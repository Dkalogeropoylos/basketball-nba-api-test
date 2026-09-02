from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from core.buckets import WeightConfig, active_weights, split_non_overlapping
from core.minutes_engine import project_team_minutes


STAT_COLS = [
    "PTS", "FGA", "FGM", "FG3A", "FG3M", "FTA", "FTM", "TOV",
    "OREB", "DREB", "REB", "AST", "PF", "STL", "BLK",
]


def _norm(v) -> str:
    return str(v).strip().casefold()


def _safe_div(a: float, b: float, default: float = np.nan) -> float:
    return float(a) / float(b) if np.isfinite(b) and float(b) > 0 else default


def _broad_position_from_row(row) -> str:
    raw = str(row.get("POSITION_GROUP", "") or "").strip().upper()
    if raw in {"G", "F", "C"}:
        return raw
    raw = str(row.get("POSITION_ABBR", raw) or "").upper().replace(" ", "")
    if "C" in raw and "F" not in raw and "G" not in raw:
        return "C"
    if "F" in raw:
        return "F"
    if "G" in raw:
        return "G"
    if "C" in raw:
        return "C"
    return ""


def _event_position_priority(stat: str, source_pos: str, target_pos: str) -> float:
    """Stat-specific role-family routing prior for vacated opportunities.

    This is deliberately separate from the generic minute-replacement matrix.
    Minutes answer *who is on the floor*; this prior answers *which replacement
    roles should inherit this particular event*.
    """
    s = str(stat).upper()
    a, b = str(source_pos or ""), str(target_pos or "")
    if not a or not b:
        return 0.35
    if a == b:
        return 1.0
    pair = {a, b}
    if s == "AST":
        if pair == {"G", "F"}: return 0.58
        if pair == {"F", "C"}: return 0.24
        return 0.08
    if s == "REB":
        if pair == {"F", "C"}: return 0.92
        if pair == {"G", "F"}: return 0.28
        return 0.10
    if s == "FG3A":
        if pair == {"G", "F"}: return 0.82
        if pair == {"F", "C"}: return 0.30
        return 0.08
    if s in {"FGA", "FTA"}:
        if pair == {"G", "F"}: return 0.76
        if pair == {"F", "C"}: return 0.72
        return 0.20
    return 0.35


def _estimate_possessions_local(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(df.get("FGA", 0), errors="coerce").fillna(0.0)
        - pd.to_numeric(df.get("OREB", 0), errors="coerce").fillna(0.0)
        + pd.to_numeric(df.get("TOV", 0), errors="coerce").fillna(0.0)
        + 0.44 * pd.to_numeric(df.get("FTA", 0), errors="coerce").fillna(0.0)
    )


def _opponent_structural_rates(rows: pd.DataFrame) -> dict:
    """Opponent offensive outcomes used for current defensive-roster bridge."""
    if rows is None or rows.empty:
        return {}
    x = rows.copy()
    for c in ["PTS","FGA","FGM","FG3A","FG3M","FTA","TOV","OREB","AST"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    poss = float(_estimate_possessions_local(x).sum())
    if poss <= 0:
        return {}
    fga = float(x["FGA"].sum())
    fgm = float(x["FGM"].sum())
    a3 = float(x["FG3A"].sum())
    m3 = float(x["FG3M"].sum())
    a2 = max(fga - a3, 0.0)
    m2 = max(fgm - m3, 0.0)
    misses = max(fga - fgm, 0.0)
    return {
        "games": int(len(x)),
        "poss": poss,
        "PTS100": 100.0 * float(x["PTS"].sum()) / poss,
        "3P_SHARE": a3 / fga if fga > 0 else np.nan,
        "FTA": float(x["FTA"].sum()) / poss,
        "TOV": float(x["TOV"].sum()) / poss,
        "OREB": float(x["OREB"].sum()) / misses if misses > 0 else np.nan,
        "AST": float(x["AST"].sum()) / fgm if fgm > 0 else np.nan,
        "3P_PCT": m3 / a3 if a3 > 0 else np.nan,
        "2P_PCT": m2 / a2 if a2 > 0 else np.nan,
    }


def defensive_absence_bridge(
    player_db: pd.DataFrame,
    team_db: pd.DataFrame,
    current_pool: pd.DataFrame,
    defense_abbr: str,
    out_players: Iterable[str],
    exclude_opponent_abbr: str | None = None,
    on_min_minutes: float = 10.0,
    k: float = 8.0,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Conservative current defensive-roster adjustment from individual OUT splits.

    For each confirmed OUT separately, compare what opponents produced when the
    player logged >= ``on_min_minutes`` to true OFF games after that player's
    first team appearance.  Games with 0 < MIN < threshold are EXCLUDED from
    both groups, so an early injury/limited stint is never treated as a normal ON.

    Individual effects are partial-pooled and overlap-protected before they are
    applied to the opponent offense.  This is intentionally a small correction,
    not a second opponent sample.
    """
    outs = [str(x) for x in out_players if str(x).strip()]
    neutral = {k: 1.0 for k in ["3P_SHARE","FTA","TOV","OREB","AST","3P_PCT","2P_PCT"]}
    if not outs or player_db is None or player_db.empty or team_db is None or team_db.empty:
        return neutral, pd.DataFrame()

    dg = team_db[team_db["TEAM_ABBR"].astype(str).str.upper().eq(str(defense_abbr).upper())].copy()
    dg["GAME_ID"] = dg["GAME_ID"].astype(str)
    dg["GAME_DATE"] = pd.to_datetime(dg["GAME_DATE"], errors="coerce")
    if exclude_opponent_abbr and "OPP_ABBR" in dg.columns:
        dg = dg[~dg["OPP_ABBR"].astype(str).str.upper().eq(str(exclude_opponent_abbr).upper())].copy()

    # Pair each defensive-team game with the opponent's actual box row.
    opp_rows = team_db.copy()
    opp_rows["GAME_ID"] = opp_rows["GAME_ID"].astype(str)
    paired = dg[["GAME_ID","GAME_DATE"]].merge(opp_rows, on="GAME_ID", how="left", suffixes=("_DEF", ""))
    paired = paired[~paired["TEAM_ABBR"].astype(str).str.upper().eq(str(defense_abbr).upper())].copy()
    if paired.empty:
        return neutral, pd.DataFrame()

    audit_rows = []
    effects = {k: [] for k in neutral}
    for name in outs:
        ph = player_db[
            player_db["TEAM_ABBR"].astype(str).str.upper().eq(str(defense_abbr).upper())
            & player_db["PLAYER_NAME"].astype(str).str.casefold().eq(str(name).strip().casefold())
        ].copy()
        if ph.empty:
            continue
        ph["GAME_ID"] = ph["GAME_ID"].astype(str)
        ph["GAME_DATE"] = pd.to_datetime(ph["GAME_DATE"], errors="coerce")
        first = ph["GAME_DATE"].dropna().min()
        if pd.isna(first):
            continue
        minute_map = pd.to_numeric(ph.set_index("GAME_ID")["MIN"], errors="coerce").fillna(0.0).to_dict()
        eligible = paired[paired["GAME_DATE_DEF"] >= first].copy()
        if eligible.empty:
            continue
        eligible["_MIN"] = eligible["GAME_ID"].map(lambda gid: float(minute_map.get(str(gid), 0.0)))
        on = eligible[eligible["_MIN"] >= float(on_min_minutes)].copy()
        off = eligible[eligible["_MIN"] <= 1e-9].copy()
        limited = eligible[(eligible["_MIN"] > 1e-9) & (eligible["_MIN"] < float(on_min_minutes))].copy()
        all_clean = pd.concat([on, off], ignore_index=True)
        if on.empty or off.empty or all_clean.empty:
            continue
        onr, offr, baser = _opponent_structural_rates(on), _opponent_structural_rates(off), _opponent_structural_rates(all_clean)
        n_on, n_off = len(on), len(off)
        harmonic = (2.0 * n_on * n_off / max(n_on + n_off, 1.0))
        maturity = float(np.clip(min(n_on, n_off) / 3.0, 0.0, 1.0))
        conf = float(np.clip((harmonic / (harmonic + float(k))) * maturity, 0.0, 1.0))
        audit_rows.append({
            "Player": name, "ON min threshold": float(on_min_minutes),
            "ON games": int(n_on), "OFF games": int(n_off), "Limited 0<MIN<threshold excluded": int(len(limited)),
            "Opp PTS/100 ON": onr.get("PTS100", np.nan), "Opp PTS/100 OFF": offr.get("PTS100", np.nan),
            "OFF-ON PTS/100": offr.get("PTS100", np.nan) - onr.get("PTS100", np.nan),
            "Split confidence": conf,
        })
        for stat in neutral:
            b = baser.get(stat, np.nan); o = offr.get(stat, np.nan); onv = onr.get(stat, np.nan)
            if not (np.isfinite(b) and b > 0 and np.isfinite(o)):
                continue
            ratio = float(np.clip(o / b, 0.82, 1.18))
            effects[stat].append((name, ratio, conf, onv, o, b))

    caps = {
        "3P_SHARE": (0.95,1.05), "FTA": (0.94,1.06), "TOV": (0.94,1.06),
        "OREB": (0.93,1.07), "AST": (0.95,1.05),
        "3P_PCT": (0.97,1.03), "2P_PCT": (0.97,1.03),
    }
    strength = {"3P_PCT":0.18, "2P_PCT":0.18, "3P_SHARE":0.30, "FTA":0.32, "TOV":0.36, "OREB":0.36, "AST":0.30}
    mods = dict(neutral)
    detail_rows = []
    for stat, vals in effects.items():
        vals = [v for v in vals if v[2] > 0]
        if not vals:
            continue
        total_c = sum(v[2] for v in vals)
        avg_log = sum(v[2] * np.log(max(v[1], 1e-6)) for v in vals) / max(total_c, 1e-9)
        agg_conf = 1.0 - float(np.prod([1.0 - min(v[2], 0.95) for v in vals]))
        severity = float(np.clip(np.sqrt(total_c / max(max(v[2] for v in vals), 1e-9)), 1.0, 1.35))
        exponent = float(strength[stat] * agg_conf * severity)
        lo, hi = caps[stat]
        mod = float(np.clip(np.exp(avg_log * exponent), lo, hi))
        mods[stat] = mod
        detail_rows.append({
            "Player": "COMBINED", "Stat": stat, "Current-OFF modifier": mod,
            "Aggregate confidence": agg_conf, "Overlap severity": severity, "Applied exponent": exponent,
            "Contributors": ", ".join(v[0] for v in vals),
        })
        for name, ratio, conf, onv, offv, basev in vals:
            detail_rows.append({
                "Player": name, "Stat": stat, "OFF / tenure-baseline ratio": ratio,
                "Split confidence": conf, "ON rate": onv, "OFF rate": offv, "Tenure clean baseline": basev,
            })

    audit = pd.concat([pd.DataFrame(audit_rows), pd.DataFrame(detail_rows)], ignore_index=True, sort=False)
    return mods, audit


def _weighted_bucket_rate(bucket: pd.DataFrame, stat: str) -> float:
    if bucket is None or bucket.empty or stat not in bucket.columns:
        return np.nan
    mins = pd.to_numeric(bucket.get("MIN", 0), errors="coerce").fillna(0.0)
    vals = pd.to_numeric(bucket.get(stat, 0), errors="coerce").fillna(0.0)
    den = float(mins.sum())
    return float(vals.sum() / den) if den > 0 else np.nan


def _player_rate_profile(log: pd.DataFrame) -> dict:
    """Stable non-overlapping per-minute player rates for roster synthesis.

    This is intentionally simpler than the Player Props engine: it is used only
    to estimate how a changed 200-minute rotation changes aggregate team style.
    It never becomes a fourth historical sample.
    """
    x = log.sort_values("GAME_DATE").copy()
    if x.empty:
        return {}
    buckets = split_non_overlapping(x)
    outer = active_weights(buckets, WeightConfig.stable())
    out = {}
    for stat in STAT_COLS:
        vals, ws = [], []
        for k in ("old", "mid", "l5"):
            r = _weighted_bucket_rate(buckets.get(k, pd.DataFrame()), stat)
            w = float(outer.get(k, 0.0))
            if w > 0 and np.isfinite(r):
                vals.append(r)
                ws.append(w)
        if vals:
            ws = np.asarray(ws, dtype=float)
            ws = ws / ws.sum()
            out[stat] = float(np.sum(np.asarray(vals) * ws))
        else:
            out[stat] = 0.0
    return out


def recent_team_player_names(
    player_db: pd.DataFrame,
    team_abbr: str,
    lookback_days: int = 70,
    min_recent_minutes: float = 1.0,
) -> list[str]:
    """Recent team names, including players no longer returned by current-pool APIs.

    This lets the trader explicitly mark a just-traded/bought-out player OUT so
    the historical team baseline can be adjusted away from that player.
    """
    if player_db is None or player_db.empty:
        return []
    x = player_db[
        player_db["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())
    ].copy()
    if x.empty:
        return []
    x["_DATE"] = pd.to_datetime(x["GAME_DATE"], errors="coerce")
    max_date = x["_DATE"].max()
    if pd.notna(max_date):
        x = x[x["_DATE"] >= max_date - pd.Timedelta(days=int(lookback_days))].copy()
    x["_MIN"] = pd.to_numeric(x.get("MIN", 0), errors="coerce").fillna(0.0)
    g = x.groupby("PLAYER_NAME", as_index=False)["_MIN"].sum()
    names = g[g["_MIN"] >= float(min_recent_minutes)]["PLAYER_NAME"].astype(str).tolist()
    return sorted(set(names), key=str.casefold)


def augment_current_pool(
    current_pool: pd.DataFrame,
    player_db: pd.DataFrame,
    team_abbr: str,
    extra_names: Iterable[str],
) -> pd.DataFrame:
    """Add explicitly referenced recent players to the provider's current pool.

    Needed for newly unavailable players (buyout/trade/season-ending removal)
    who may disappear from a current-roster endpoint before the historical model
    has any games without them.
    """
    pool = current_pool.copy() if current_pool is not None else pd.DataFrame()
    extras = [str(x) for x in extra_names if str(x).strip()]
    if not extras or player_db is None or player_db.empty:
        return pool

    existing = set(pool.get("PLAYER_NAME", pd.Series(dtype=str)).astype(str).map(_norm).tolist())
    add_rows = []
    for name in extras:
        if _norm(name) in existing:
            continue
        hist = player_db[
            player_db["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())
            & player_db["PLAYER_NAME"].astype(str).map(_norm).eq(_norm(name))
        ].copy()
        if hist.empty:
            continue
        latest = hist.sort_values("GAME_DATE").iloc[-1]
        row = {}
        # Preserve the provider-pool schema where possible.
        cols = list(pool.columns) if len(pool.columns) else [
            "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBR", "POSITION_ABBR", "POSITION_GROUP"
        ]
        for c in cols:
            if c == "PLAYER_NAME":
                row[c] = str(latest.get(c, name))
            elif c == "TEAM_ABBR":
                row[c] = str(team_abbr).upper()
            else:
                row[c] = latest.get(c, np.nan)
        if "PLAYER_ID" not in row:
            row["PLAYER_ID"] = latest.get("PLAYER_ID")
        if "PLAYER_NAME" not in row:
            row["PLAYER_NAME"] = name
        if "TEAM_ABBR" not in row:
            row["TEAM_ABBR"] = str(team_abbr).upper()
        add_rows.append(row)
        existing.add(_norm(name))
    if add_rows:
        pool = pd.concat([pool, pd.DataFrame(add_rows)], ignore_index=True, sort=False)
    return pool.drop_duplicates(subset=["PLAYER_ID"], keep="last").reset_index(drop=True)


def _set_selected_out_in(ctx: dict, out_players: Iterable[str]) -> dict:
    out = copy.deepcopy(ctx or {})
    injuries = out.setdefault("injuries", {})
    targets = {_norm(x) for x in out_players}
    for name, info in list(injuries.items()):
        if _norm(name) not in targets:
            continue
        if isinstance(info, dict):
            info = dict(info)
            info["status"] = "IN"
            injuries[name] = info
        else:
            injuries[name] = {"status": "IN"}
    return out


def _without_team_minute_overrides(ctx: dict, team_names: Iterable[str]) -> dict:
    out = copy.deepcopy(ctx or {})
    block = dict(out.get("projected_minutes", {}) or {})
    targets = {_norm(x) for x in team_names}
    block = {k: v for k, v in block.items() if _norm(k) not in targets}
    out["projected_minutes"] = block
    return out


def _profiles_for_team(player_db: pd.DataFrame, team_abbr: str, names: Iterable[str]) -> Dict[str, dict]:
    profiles = {}
    for name in names:
        log = player_db[
            player_db["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())
            & player_db["PLAYER_NAME"].astype(str).map(_norm).eq(_norm(name))
        ].copy()
        p = _player_rate_profile(log)
        if p:
            profiles[str(name)] = p
    return profiles


def _synthetic_from_minutes(board: pd.DataFrame, profiles: Dict[str, dict]) -> Tuple[dict, dict]:
    totals = {s: 0.0 for s in STAT_COLS}
    by_player = {}
    if board is None or board.empty:
        return totals, by_player
    for _, row in board.iterrows():
        name = str(row["Player"])
        mins = float(row.get("Projected Min", 0.0) or 0.0)
        p = profiles.get(name, {})
        pt = {}
        for stat in STAT_COLS:
            val = mins * float(p.get(stat, 0.0) or 0.0)
            totals[stat] += val
            pt[stat] = val
        pt["MIN"] = mins
        by_player[name] = pt
    fga, fgm = totals["FGA"], totals["FGM"]
    a3, m3 = totals["FG3A"], totals["FG3M"]
    a2, m2 = max(fga - a3, 0.0), max(fgm - m3, 0.0)
    misses = max(fga - fgm, 0.0)
    feats = {
        "FGA": fga,
        "3P_SHARE": _safe_div(a3, fga, 0.35),
        "FTA": totals["FTA"],
        "TOV": totals["TOV"],
        "OREB": _safe_div(totals["OREB"], misses, 0.25),
        "AST": _safe_div(totals["AST"], fgm, 0.62),
        "PF": totals["PF"],
        "DREB": totals["DREB"],
        "STL": totals["STL"],
        "BLK": totals["BLK"],
        "3P_PCT": _safe_div(m3, a3, np.nan),
        "2P_PCT": _safe_div(m2, a2, np.nan),
    }
    return feats, by_player


def _ratio(a, b, default=1.0):
    return _safe_div(a, b, default) if np.isfinite(a) and np.isfinite(b) else default


def _pow_clip(ratio: float, exponent: float, lo: float, hi: float) -> float:
    if not np.isfinite(ratio) or ratio <= 0 or exponent <= 0:
        return 1.0
    return float(np.clip(float(ratio) ** float(exponent), lo, hi))


TEAM_STAT_POWER = {
    # FGA count is mostly possession-driven, so roster identity only nudges it.
    "FGA": 0.25,
    # Shot selection / foul drawing / creation can move more when a role changes.
    "3P_SHARE": 0.65,
    "FTA": 0.60,
    "TOV": 0.45,
    "OREB": 0.45,
    "AST": 0.55,
    "PF": 0.40,
    "DREB": 0.35,
    "STL": 0.35,
    "BLK": 0.55,
    # Shooter-mix changes matter, but efficiency stays heavily regularized.
    "3P_PCT": 0.45,
    "2P_PCT": 0.40,
}

TEAM_CAPS = {
    "FGA": (0.96, 1.04),
    "3P_SHARE": (0.90, 1.10),
    "FTA": (0.88, 1.12),
    "TOV": (0.90, 1.10),
    "OREB": (0.90, 1.10),
    "AST": (0.90, 1.10),
    "PF": (0.92, 1.08),
    "DREB": (0.94, 1.06),
    "STL": (0.92, 1.08),
    "BLK": (0.88, 1.12),
    "3P_PCT": (0.96, 1.04),
    "2P_PCT": (0.96, 1.04),
}


@dataclass
class RotationStateImpact:
    modifiers: Dict[str, float]
    team_audit: pd.DataFrame
    current_minutes: pd.DataFrame
    out_only_minutes: pd.DataFrame
    healthy_minutes: pd.DataFrame
    raw_player_role_modifiers: pd.DataFrame


def build_rotation_state_impact(
    player_db: pd.DataFrame,
    team_db: pd.DataFrame,
    current_pool: pd.DataFrame,
    team_abbr: str,
    team_name: str,
    manual_context: dict,
    out_players: Iterable[str],
    exact_state_confidence: float = 0.0,
    state_confidence_by_stat: Dict[str, float] | None = None,
) -> RotationStateImpact:
    """Roster/minute-state bridge for Team Markets and Player Props.

    Three states are simulated with the SAME player per-minute priors:
      healthy  = selected OUT players restored; no team minute overrides
      out_only = confirmed OUT state; no team minute overrides
      current  = confirmed OUT + explicit projected-minute restrictions/returns

    Therefore:
      - exact historical OUT games remain the primary empirical layer;
      - the synthetic OUT modifier is strongest only when exact-state evidence
        is sparse, avoiding duplicate injury information;
      - minute restrictions are a separate trader-information layer and are not
        mistaken for another historical sample.
    """
    out_players = [str(x) for x in out_players if str(x).strip()]
    # Only explicitly selected unavailable players are reintroduced into the
    # calculation pool. Other recent ex-players must never receive today's minutes merely because they appear in historical logs.
    pool = augment_current_pool(current_pool, player_db, team_abbr, out_players)
    team_names = pool[
        pool["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())
    ]["PLAYER_NAME"].astype(str).tolist()

    current_ctx = copy.deepcopy(manual_context or {})
    out_only_ctx = _without_team_minute_overrides(current_ctx, team_names)
    healthy_ctx = _set_selected_out_in(out_only_ctx, out_players)

    current_board = project_team_minutes(
        player_db, team_db, pool, team_abbr, team_name, current_ctx
    )
    out_only_board = project_team_minutes(
        player_db, team_db, pool, team_abbr, team_name, out_only_ctx
    )
    healthy_board = project_team_minutes(
        player_db, team_db, pool, team_abbr, team_name, healthy_ctx
    )

    names = sorted(set(team_names), key=str.casefold)
    profiles = _profiles_for_team(player_db, team_abbr, names)
    current_feat, current_by = _synthetic_from_minutes(current_board, profiles)
    out_feat, out_by = _synthetic_from_minutes(out_only_board, profiles)
    healthy_feat, healthy_by = _synthetic_from_minutes(healthy_board, profiles)

    conf = float(np.clip(exact_state_confidence, 0.0, 1.0))
    conf_by = {
        str(k).upper(): float(np.clip(v, 0.0, 1.0))
        for k, v in (state_confidence_by_stat or {}).items()
    }
    def _state_conf(stat: str) -> float:
        return float(conf_by.get(str(stat).upper(), conf))

    # Explicit projected_minutes are trader information and are not represented
    # by exact historical OUT matching, so they receive a separate strong but
    # still shrinked bridge.
    def _shifted_minutes(a: pd.DataFrame, b: pd.DataFrame) -> float:
        if a is None or a.empty or b is None or b.empty:
            return 0.0
        aa = a.set_index("Player")["Projected Min"]
        bb = b.set_index("Player")["Projected Min"]
        idx = aa.index.union(bb.index)
        return 0.5 * float((aa.reindex(idx, fill_value=0.0) - bb.reindex(idx, fill_value=0.0)).abs().sum())

    out_shift = _shifted_minutes(out_only_board, healthy_board)
    restriction_shift = _shifted_minutes(current_board, out_only_board)
    restriction_evidence = 0.80 if restriction_shift >= 0.5 else 0.0

    # Some team stats are extremely concentrated in one or two players. A fixed
    # +/- cap can make a multi-absence scenario mathematically unable to move
    # enough (classic example: two rim protectors OUT but BLK can fall at most 12%).
    # Use the actual healthy contribution share of the confirmed OUT players only
    # to relax the LOWER cap for concentration-sensitive defensive stats. This is
    # a guard rail, not an extra boost, and it cannot move a stat unless the
    # synthetic healthy->OUT ratio already points in that direction.
    concentration_stat = {"BLK": "BLK", "DREB": "DREB", "OREB": "OREB"}
    def _out_healthy_share(team_stat: str) -> float:
        pstat = concentration_stat.get(team_stat)
        if not pstat:
            return 0.0
        total = sum(float(healthy_by.get(n, {}).get(pstat, 0.0) or 0.0) for n in names)
        lost = sum(
            max(float(healthy_by.get(n, {}).get(pstat, 0.0) or 0.0)
                - float(out_by.get(n, {}).get(pstat, 0.0) or 0.0), 0.0)
            for n in out_players
        )
        return float(np.clip(_safe_div(lost, total, 0.0), 0.0, 1.0))

    mods = {}
    rows = []
    for stat, power in TEAM_STAT_POWER.items():
        lo, hi = TEAM_CAPS[stat]
        lost_share = _out_healthy_share(stat)
        base_lo = lo
        if stat == "BLK" and lost_share > 0:
            lo = max(0.72, lo - 0.30 * lost_share)
        elif stat == "DREB" and lost_share > 0:
            lo = max(0.88, lo - 0.14 * lost_share)
        elif stat == "OREB" and lost_share > 0:
            lo = max(0.86, lo - 0.12 * lost_share)
        stat_conf = _state_conf(stat)
        # Historical exact/near-state evidence already lives inside the team
        # buckets. The synthetic bridge fills only the residual information gap.
        out_evidence = 0.65 * (1.0 - stat_conf)
        r_out = _ratio(out_feat.get(stat, np.nan), healthy_feat.get(stat, np.nan), 1.0)
        r_rest = _ratio(current_feat.get(stat, np.nan), out_feat.get(stat, np.nan), 1.0)
        m_out = _pow_clip(r_out, power * out_evidence, lo, hi)
        m_rest = _pow_clip(r_rest, power * restriction_evidence, lo, hi)
        final = float(np.clip(m_out * m_rest, lo, hi))
        mods[stat] = final
        rows.append({
            "Stat": stat,
            "Healthy synthetic": healthy_feat.get(stat, np.nan),
            "OUT-only synthetic": out_feat.get(stat, np.nan),
            "Current synthetic": current_feat.get(stat, np.nan),
            "OUT raw ratio": r_out,
            "State confidence": stat_conf,
            "OUT bridge strength": power * out_evidence,
            "Restriction raw ratio": r_rest,
            "Restriction bridge strength": power * restriction_evidence,
            "OUT healthy stat share": lost_share,
            "Base lower cap": base_lo,
            "Adaptive lower cap": lo,
            "Applied roster-state modifier": final,
        })

    # Player-role fallback: minutes are ALREADY modeled directly, so this layer
    # only allocates a fraction of vacated event volume that would otherwise
    # disappear because replacement players have lower historical per-minute rates.
    # It is later shrunk again by each focal player's exact-state confidence.
    player_rows = []
    events = {
        "FGA": ("usage", 0.90),
        "FG3A": ("three_rate", 0.85),
        "FTA": ("fta_rate", 0.75),
        "AST": ("creation", 0.80),
        "REB": ("reb_role", 0.95),
    }
    event_state_key = {"FGA":"FGA", "FG3A":"3PA", "FTA":"FTA", "AST":"AST", "REB":"REB"}
    replacement_matrix = out_only_board.attrs.get("redistribution_matrix", {}) or {}
    pos_map = {}
    if pool is not None and not pool.empty:
        for _, _r in pool.iterrows():
            pos_map[str(_r.get("PLAYER_NAME", ""))] = _broad_position_from_row(_r)

    def _out_stage_adjust(base_by, reference_by, stage_current_board):
        """Redistribute each OUT player's vacated events through the SAME
        learned teammate-replacement relationships used for minutes.

        This is the key v2.15 change: a guard's vacated AST/FGA no longer becomes
        a generic team-wide boost. Players who actually replace that guard in the
        historical/position-shrunk minute matrix receive most of the event volume.
        """
        adj = {name: {s: float(base_by.get(name, {}).get(s, 0.0)) for s in events} for name in names}
        active = {
            str(r["Player"]): float(r.get("Projected Min", 0.0) or 0.0)
            for _, r in stage_current_board.iterrows()
            if float(r.get("Projected Min", 0.0) or 0.0) > 0.1
        }
        out_set = {str(x) for x in out_players}
        for stat, (_, preserve) in events.items():
            stat_conf = _state_conf(event_state_key[stat])
            residual_scale = (1.0 - stat_conf)
            for focal in out_set:
                lost = max(
                    float(reference_by.get(focal, {}).get(stat, 0.0))
                    - float(base_by.get(focal, {}).get(stat, 0.0)),
                    0.0,
                )
                residual = lost * preserve * residual_scale
                if residual <= 0 or not active:
                    continue
                matrix_row = replacement_matrix.get(focal, {}) or {}
                raw_w = {}
                for n, mins in active.items():
                    if n in out_set or n == focal:
                        continue
                    replace_w = max(float(matrix_row.get(n, 0.0)), 0.0)
                    contrib = max(float(base_by.get(n, {}).get(stat, 0.0)), 0.0)
                    ability = max(contrib / max(mins, 1.0), 0.005)
                    # v2.16: minute replacement is no longer the only routing
                    # signal.  A stat-specific role-family priority prevents,
                    # for example, creator AST from flowing mechanically to a
                    # frontcourt minute replacement.
                    role_priority = _event_position_priority(
                        stat, pos_map.get(focal, ""), pos_map.get(n, "")
                    )
                    raw_w[n] = (
                        max(replace_w, 0.002) ** 0.55
                        * max(role_priority, 0.03) ** 0.30
                        * ability ** 0.15
                    )
                den = sum(raw_w.values())
                if den <= 0:
                    continue
                for n, w in raw_w.items():
                    adj.setdefault(n, {})[stat] = adj.get(n, {}).get(stat, 0.0) + residual * w / den
        return adj

    def _stage_adjust(
        base_by, reference_by, stage_current_board, preserve_scale=1.0,
        reference_board=None, exclude_minute_decliners=False,
    ):
        adj = {name: {s: float(base_by.get(name, {}).get(s, 0.0)) for s in events} for name in names}
        active = {
            str(r["Player"]): float(r.get("Projected Min", 0.0) or 0.0)
            for _, r in stage_current_board.iterrows()
            if float(r.get("Projected Min", 0.0) or 0.0) > 0.1
        }
        ref_min = {}
        if reference_board is not None and not reference_board.empty:
            ref_min = {
                str(r["Player"]): float(r.get("Projected Min", 0.0) or 0.0)
                for _, r in reference_board.iterrows()
            }
        for stat, (_, preserve) in events.items():
            base_total = sum(float(base_by.get(n, {}).get(stat, 0.0)) for n in names)
            ref_total = sum(float(reference_by.get(n, {}).get(stat, 0.0)) for n in names)
            lost = max(ref_total - base_total, 0.0)
            residual = lost * preserve * preserve_scale
            if residual <= 0 or not active:
                continue
            raw_w = {}
            for n, mins in active.items():
                if exclude_minute_decliners and mins < ref_min.get(n, mins) - 0.25:
                    raw_w[n] = 0.0
                    continue
                contrib = max(float(base_by.get(n, {}).get(stat, 0.0)), 0.0)
                raw_w[n] = max(contrib, 0.02 * mins) ** 1.10
            den = sum(raw_w.values())
            if den <= 0:
                continue
            for n, w in raw_w.items():
                adj.setdefault(n, {})[stat] = adj.get(n, {}).get(stat, 0.0) + residual * w / den
        return adj

    # OUT residual is reduced by exact-state evidence; restriction residual is
    # separate and full-strength because it is trader-specified current context.
    out_adjusted = _out_stage_adjust(
        out_by, healthy_by, out_only_board
    )
    current_adjusted = _stage_adjust(
        current_by, out_by, current_board, preserve_scale=1.0,
        reference_board=out_only_board, exclude_minute_decliners=True,
    )

    current_min_map = {
        str(r["Player"]): float(r.get("Projected Min", 0.0) or 0.0)
        for _, r in current_board.iterrows()
    }
    for name in names:
        mins = current_min_map.get(name, 0.0)
        p = profiles.get(name, {})
        if mins <= 0.1 or not p:
            continue
        base_fga_pm = max(float(p.get("FGA", 0.0)), 1e-6)
        base_3pa_pm = max(float(p.get("FG3A", 0.0)), 1e-6)
        base_fta_pm = max(float(p.get("FTA", 0.0)), 1e-6)
        base_ast_pm = max(float(p.get("AST", 0.0)), 1e-6)
        base_reb_pm = max(float(p.get("REB", 0.0)), 1e-6)

        # Combine OUT-stage and restriction-stage residual additions. Each stage
        # starts from its own current synthetic contribution; we take the extra
        # above that stage baseline and add it to today's raw contribution.
        def extra(stage_adj, stage_base, stat):
            return max(
                float(stage_adj.get(name, {}).get(stat, 0.0))
                - float(stage_base.get(name, {}).get(stat, 0.0)), 0.0
            )

        fga_total = float(current_by.get(name, {}).get("FGA", 0.0)) \
            + extra(out_adjusted, out_by, "FGA") + extra(current_adjusted, current_by, "FGA")
        three_total = float(current_by.get(name, {}).get("FG3A", 0.0)) \
            + extra(out_adjusted, out_by, "FG3A") + extra(current_adjusted, current_by, "FG3A")
        fta_total = float(current_by.get(name, {}).get("FTA", 0.0)) \
            + extra(out_adjusted, out_by, "FTA") + extra(current_adjusted, current_by, "FTA")
        ast_total = float(current_by.get(name, {}).get("AST", 0.0)) \
            + extra(out_adjusted, out_by, "AST") + extra(current_adjusted, current_by, "AST")
        reb_total = float(current_by.get(name, {}).get("REB", 0.0)) \
            + extra(out_adjusted, out_by, "REB") + extra(current_adjusted, current_by, "REB")

        usage = float(np.clip((fga_total / mins) / base_fga_pm, 0.85, 1.20))
        three_rate = float(np.clip((three_total / mins) / base_3pa_pm, 0.80, 1.25))
        fta_rate = float(np.clip((fta_total / mins) / base_fta_pm, 0.80, 1.25))
        creation = float(np.clip((ast_total / mins) / base_ast_pm, 0.80, 1.25))
        reb_role = float(np.clip((reb_total / mins) / base_reb_pm, 0.85, 1.15))
        # simulate_player multiplies 3PA and FTA by usage as well; use relative
        # modifiers so the same vacated shot is not counted twice.
        three_role = float(np.clip(three_rate / max(usage, 1e-6), 0.80, 1.25))
        fta_role = float(np.clip(fta_rate / max(usage, 1e-6), 0.80, 1.25))
        player_rows.append({
            "Player": name,
            "Usage fallback": usage,
            "Three-role fallback": three_role,
            "FTA-role fallback": fta_role,
            "Creation fallback": creation,
            "Rebound-role fallback": reb_role,
            "Current Min": mins,
            "OUT minutes shifted": out_shift,
            "Restriction minutes shifted": restriction_shift,
        })

    audit = pd.DataFrame(rows)
    audit.attrs["out_minutes_shifted"] = out_shift
    audit.attrs["restriction_minutes_shifted"] = restriction_shift
    return RotationStateImpact(
        modifiers=mods,
        team_audit=audit,
        current_minutes=current_board,
        out_only_minutes=out_only_board,
        healthy_minutes=healthy_board,
        raw_player_role_modifiers=pd.DataFrame(player_rows),
    )
