from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def _broad_position(position_group=None, position_abbr=None) -> str:
    """Return G/F/C even when the provider only supplies combo positions."""
    raw = str(position_group or "").strip().upper()
    if raw in {"G", "F", "C"}:
        return raw
    raw = str(position_abbr or raw or "").strip().upper().replace(" ", "")
    if not raw or raw == "NAN":
        return ""
    # Combo positions use the frontcourt/primary replacement family rather than
    # becoming 'unknown'.  This is only a fallback prior; empirical replacement
    # evidence can still override it.
    if "C" in raw:
        return "C" if raw in {"C", "C-F", "C/F"} else "F"
    if "F" in raw:
        return "F"
    if "G" in raw:
        return "G"
    return ""


def _position_prior(a, b) -> float:
    a = _broad_position(a)
    b = _broad_position(b)
    if not a or not b:
        return 0.25
    if a == b:
        return 1.00
    pair = {a, b}
    if pair == {"G", "F"}:
        return 0.50
    if pair == {"F", "C"}:
        return 0.65
    if pair == {"G", "C"}:
        return 0.08
    return 0.25


def _team_game_frame(team_db: pd.DataFrame, team_abbr: str) -> pd.DataFrame:
    x = team_db[
        team_db["TEAM_ABBR"].astype(str).str.upper()
        == str(team_abbr).upper()
    ].copy()
    x["GAME_DATE"] = pd.to_datetime(x["GAME_DATE"], errors="coerce")
    x["GAME_ID"] = x["GAME_ID"].astype(str)

    # Mark large blowouts / OT for pair-learning quality.
    margins = []
    for _, row in x.iterrows():
        gid = str(row["GAME_ID"])
        all_rows = team_db[team_db["GAME_ID"].astype(str) == gid]
        opp = all_rows[
            all_rows["TEAM_ABBR"].astype(str).str.upper()
            != str(team_abbr).upper()
        ]
        own_pts = float(row["PTS"]) if pd.notna(row.get("PTS")) else 0.0
        opp_pts = float(opp.iloc[0]["PTS"]) if not opp.empty else own_pts
        margins.append(abs(own_pts - opp_pts))
    x["ABS_MARGIN"] = margins
    if "OT_FLAG" not in x.columns:
        x["OT_FLAG"] = False
    return x.sort_values("GAME_DATE").reset_index(drop=True)


def _minutes_series_for_player(
    player_db: pd.DataFrame,
    team_games: pd.DataFrame,
    player_id,
    team_abbr: str,
) -> pd.Series:
    """
    Fill zero inside the player's tenure window with the team.
    For a current-roster player, tenure begins at first appearance for this team
    and extends through the current latest team game.
    """
    p = player_db[
        (player_db["PLAYER_ID"] == player_id)
        & (
            player_db["TEAM_ABBR"].astype(str).str.upper()
            == str(team_abbr).upper()
        )
    ].copy()
    if p.empty:
        return pd.Series(dtype=float)

    p["GAME_DATE"] = pd.to_datetime(p["GAME_DATE"], errors="coerce")
    p["GAME_ID"] = p["GAME_ID"].astype(str)

    first_date = p["GAME_DATE"].min()
    eligible_games = team_games[team_games["GAME_DATE"] >= first_date].copy()

    minute_map = (
        p.groupby("GAME_ID")["MIN"]
        .sum()
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_dict()
    )
    s = eligible_games.set_index("GAME_ID").index.to_series().map(
        lambda gid: float(minute_map.get(str(gid), 0.0))
    )
    s.index = eligible_games["GAME_ID"].astype(str).values
    return s.astype(float)


def _pair_score(
    a: pd.Series,
    b: pd.Series,
    role_prior: float,
) -> Tuple[float, dict]:
    common = a.index.intersection(b.index)
    n = len(common)
    if n < 2:
        # One game cannot identify an inverse-minute relationship. Keep the
        # structural prior, but do not claim empirical confidence. Five common
        # games remains the maturity point; it is no longer a hard 4->5 cliff.
        score = 0.10 * role_prior
        return score, {
            "games": int(n),
            "neg_slope": 0.0,
            "onoff": 0.0,
            "confidence": 0.0,
            "maturity": float(n / 5.0),
            "role_prior": role_prior,
            "score": score,
        }

    av = a.loc[common].to_numpy(dtype=float)
    bv = b.loc[common].to_numpy(dtype=float)

    # Robustly center each player's minute history. Negative covariance after
    # centering is evidence that one player tends to gain when the other loses.
    ac = av - np.median(av)
    bc = bv - np.median(bv)

    var_a = float(np.var(ac))
    if var_a > 1e-8:
        slope = float(np.cov(ac, bc, ddof=0)[0, 1] / var_a)
    else:
        slope = 0.0
    neg_slope = float(np.clip(-slope, 0.0, 1.25))

    # On/off-like lift: how much B gains when A is at the low end of their
    # tenure distribution versus A's normal/high-minute games.
    active_a = av[av > 0]
    onoff = 0.0
    if len(active_a) >= 4:
        q25 = float(np.quantile(active_a, 0.25))
        q60 = float(np.quantile(active_a, 0.60))
        low_mask = av <= max(5.0, q25)
        high_mask = av >= q60
        if low_mask.sum() >= 2 and high_mask.sum() >= 2:
            b_low = float(np.mean(bv[low_mask]))
            b_high = float(np.mean(bv[high_mask]))
            a_low = float(np.mean(av[low_mask]))
            a_high = float(np.mean(av[high_mask]))
            lost_a = max(a_high - a_low, 1.0)
            onoff = float(np.clip((b_low - b_high) / lost_a, 0.0, 1.25))

    # Five common games is a maturity threshold, not an on/off switch.
    # 2-4 games can contribute, but are strongly shrunk toward the positional
    # prior; 5+ uses the ordinary N/(N+12) reliability curve.
    maturity = float(np.clip(n / 5.0, 0.0, 1.0))
    confidence = float(np.clip((n / (n + 12.0)) * maturity, 0.0, 1.0))

    empirical = 0.65 * neg_slope + 0.35 * onoff

    # Position is NOT the main model. It is only a shrinkage fallback when
    # history is thin/noisy.
    score = (
        confidence * empirical
        + (1.0 - confidence) * 0.18 * role_prior
        + 0.08 * role_prior
    )
    score = float(max(score, 0.001))

    return score, {
        "games": int(n),
        "neg_slope": neg_slope,
        "onoff": onoff,
        "confidence": confidence,
        "maturity": maturity,
        "role_prior": role_prior,
        "score": score,
    }


def learn_redistribution_matrix(
    player_db: pd.DataFrame,
    team_db: pd.DataFrame,
    current_pool: pd.DataFrame,
    team_abbr: str,
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Learn row-normalized teammate substitution weights.

    A row answers:
      "If Player A gains/loses one minute relative to AUTO, which teammates
       historically tend to move the other way?"

    The matrix is learned only among the current roster.
    """
    team_games = _team_game_frame(team_db, team_abbr)
    pool = current_pool[
        current_pool["TEAM_ABBR"].astype(str).str.upper()
        == str(team_abbr).upper()
    ].copy()

    players = []
    series = {}
    positions = {}

    for _, row in pool.iterrows():
        pname = str(row["PLAYER_NAME"])
        pid = row["PLAYER_ID"]
        s = _minutes_series_for_player(
            player_db, team_games, pid, team_abbr
        )
        if s.empty:
            continue
        players.append(pname)
        series[pname] = s
        positions[pname] = _broad_position(
            row.get("POSITION_GROUP"), row.get("POSITION_ABBR")
        )

    matrix: Dict[str, Dict[str, float]] = {}
    audit_rows = []

    for a_name in players:
        raw_scores = {}
        raw_audit = {}

        for b_name in players:
            if b_name == a_name:
                continue
            prior = _position_prior(
                positions.get(a_name),
                positions.get(b_name),
            )
            score, detail = _pair_score(
                series[a_name],
                series[b_name],
                prior,
            )
            raw_scores[b_name] = score
            raw_audit[b_name] = detail

        denom = sum(raw_scores.values())
        if denom <= 0:
            denom = 1.0

        row_weights = {
            b: float(v / denom)
            for b, v in raw_scores.items()
        }
        matrix[a_name] = row_weights

        for b_name, weight in row_weights.items():
            d = raw_audit[b_name]
            audit_rows.append({
                "Focal": a_name,
                "Replacement": b_name,
                "Weight": weight,
                **d,
                "Focal Pos": positions.get(a_name),
                "Replacement Pos": positions.get(b_name),
            })

    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit = audit.sort_values(
            ["Focal", "Weight"],
            ascending=[True, False],
        ).reset_index(drop=True)

    return matrix, audit


def _weighted_capacity_transfer(
    minutes: Dict[str, float],
    amount: float,
    weights: Dict[str, float],
    direction: str,
    excluded: set[str],
    cap: float = 48.0,
) -> float:
    """
    direction='remove' -> take minutes from teammates.
    direction='add'    -> give released minutes to teammates.

    Returns any residual amount that could not be transferred.
    """
    remaining = float(max(amount, 0.0))
    if remaining <= 1e-10:
        return 0.0

    candidates = {
        p: max(float(w), 0.0)
        for p, w in weights.items()
        if p not in excluded and p in minutes
    }

    for _ in range(30):
        if remaining <= 1e-9 or not candidates:
            break

        capacities = {}
        for p in list(candidates):
            if direction == "remove":
                capacities[p] = max(minutes[p], 0.0)
            else:
                capacities[p] = max(cap - minutes[p], 0.0)

        candidates = {
            p: w for p, w in candidates.items()
            if capacities.get(p, 0.0) > 1e-9
        }
        if not candidates:
            break

        denom = sum(candidates.values())
        if denom <= 0:
            normalized = {
                p: 1.0 / len(candidates)
                for p in candidates
            }
        else:
            normalized = {
                p: w / denom
                for p, w in candidates.items()
            }

        used = 0.0
        for p, w in normalized.items():
            desired = remaining * w
            moved = min(desired, capacities[p])
            if direction == "remove":
                minutes[p] -= moved
            else:
                minutes[p] += moved
            used += moved

        remaining -= used
        if used <= 1e-10:
            break

    return float(max(remaining, 0.0))


def _reconcile_total(
    minutes: Dict[str, float],
    total: float,
    fixed_names: set[str],
    cap: float = 48.0,
):
    residual = float(total - sum(minutes.values()))
    if abs(residual) <= 1e-7:
        return

    eligible = [
        p for p in minutes
        if p not in fixed_names
    ]
    if not eligible:
        return

    if residual > 0:
        weights = {
            p: max(cap - minutes[p], 0.0)
            for p in eligible
        }
        _weighted_capacity_transfer(
            minutes, residual, weights, "add", fixed_names, cap
        )
    else:
        weights = {
            p: max(minutes[p], 0.0)
            for p in eligible
        }
        _weighted_capacity_transfer(
            minutes, -residual, weights, "remove", fixed_names, cap
        )


def apply_role_aware_overrides(
    base_minutes: Dict[str, float],
    override_targets: Dict[str, float],
    matrix: Dict[str, Dict[str, float]],
    out_players: set[str] | None = None,
    total_minutes: float = 240.0,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Apply explicit minute overrides around an already context-aware AUTO
    rotation.

    This intentionally does NOT re-handle injury absences that were already
    incorporated by the rotation-similarity AUTO engine. That avoids counting
    the same absence twice.

    Metadata/trader minute targets are treated as explicit deviations from AUTO.
    """
    out_players = out_players or set()
    minutes = {
        p: float(np.clip(v, 0.0, 48.0))
        for p, v in base_minutes.items()
    }

    valid_overrides = {
        p: float(np.clip(v, 0.0, 48.0))
        for p, v in override_targets.items()
        if p in minutes and p not in out_players
    }
    fixed_names = set(valid_overrides) | set(out_players)

    impact_rows = []

    # Largest deviations first; usually only 1-3 trader overrides exist.
    ordered = sorted(
        valid_overrides.items(),
        key=lambda kv: abs(kv[1] - minutes.get(kv[0], 0.0)),
        reverse=True,
    )

    for focal, target in ordered:
        before = float(minutes[focal])
        delta = float(target - before)

        if abs(delta) <= 1e-9:
            minutes[focal] = target
            continue

        row = matrix.get(focal, {})
        eligible_weights = {
            p: w for p, w in row.items()
            if p not in fixed_names and p != focal
        }

        # Fallback if matrix is sparse.
        if not eligible_weights:
            eligible_weights = {
                p: max(minutes[p], 0.1)
                for p in minutes
                if p not in fixed_names and p != focal
            }

        snapshot = dict(minutes)
        minutes[focal] = target

        if delta > 0:
            residual = _weighted_capacity_transfer(
                minutes,
                delta,
                eligible_weights,
                "remove",
                fixed_names | {focal},
            )
        else:
            residual = _weighted_capacity_transfer(
                minutes,
                -delta,
                eligible_weights,
                "add",
                fixed_names | {focal},
            )

        # If learned replacements hit 0/48 caps, reconcile any remainder
        # across all non-fixed players. This is a constraint fallback only.
        if residual > 1e-8:
            fallback = {
                p: (
                    max(minutes[p], 0.1)
                    if delta > 0
                    else max(48.0 - minutes[p], 0.1)
                )
                for p in minutes
                if p not in fixed_names and p != focal
            }
            _weighted_capacity_transfer(
                minutes,
                residual,
                fallback,
                "remove" if delta > 0 else "add",
                fixed_names | {focal},
            )

        for teammate in minutes:
            if teammate == focal:
                continue
            change = minutes[teammate] - snapshot[teammate]
            if abs(change) > 0.01:
                impact_rows.append({
                    "Override Player": focal,
                    "Override From": before,
                    "Override To": target,
                    "Override Delta": delta,
                    "Affected Player": teammate,
                    "Affected Delta": change,
                    "Learned Weight": row.get(teammate, np.nan),
                })

    for p in out_players:
        if p in minutes:
            minutes[p] = 0.0

    _reconcile_total(
        minutes,
        total_minutes,
        fixed_names,
        cap=48.0,
    )

    impact = pd.DataFrame(impact_rows)
    return minutes, impact


def apply_confirmed_outs(
    healthy_minutes: Dict[str, float],
    matrix: Dict[str, Dict[str, float]],
    out_players: set[str] | None = None,
    total_minutes: float = 240.0,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Set confirmed OUT players to zero and redistribute their healthy minutes.

    Redistribution uses the learned teammate replacement matrix. Position enters
    only as a fallback prior inside that matrix, so real historical substitution
    evidence wins when it exists. Multiple OUTs are processed largest-vacated-
    minutes first and can never receive one another's minutes.
    """
    outs = {str(x) for x in (out_players or set())}
    minutes = {
        p: float(np.clip(v, 0.0, 48.0))
        for p, v in healthy_minutes.items()
    }
    impact_rows = []

    ordered = sorted(
        [p for p in outs if p in minutes],
        key=lambda p: minutes.get(p, 0.0),
        reverse=True,
    )

    for focal in ordered:
        released = float(max(minutes.get(focal, 0.0), 0.0))
        minutes[focal] = 0.0
        if released <= 1e-9:
            impact_rows.append({
                "Focal": focal,
                "Healthy Min": 0.0,
                "Released Min": 0.0,
                "Unallocated Min": 0.0,
                "Mode": "confirmed OUT",
            })
            continue

        row = {
            p: float(w)
            for p, w in (matrix.get(focal, {}) or {}).items()
            if p in minutes and p not in outs and p != focal
        }
        if not row:
            # Last-resort fallback is current healthy minute capacity, not equal
            # random sharing. This path should be rare because matrix rows also
            # contain a position prior when empirical history is sparse.
            row = {
                p: max(float(minutes[p]), 0.1)
                for p in minutes
                if p not in outs and p != focal
            }

        before = dict(minutes)
        residual = _weighted_capacity_transfer(
            minutes,
            released,
            row,
            direction="add",
            excluded=outs | {focal},
            cap=48.0,
        )

        moved = []
        for p in minutes:
            delta = float(minutes[p] - before.get(p, 0.0))
            if delta > 1e-8:
                moved.append((p, delta))
        moved.sort(key=lambda kv: kv[1], reverse=True)

        impact_rows.append({
            "Focal": focal,
            "Healthy Min": released,
            "Released Min": released - residual,
            "Unallocated Min": residual,
            "Top replacements": ", ".join(f"{p} +{d:.1f}" for p, d in moved[:5]),
            "Mode": "learned matrix + position fallback",
        })

    # Keep OUTs fixed at zero and reconcile any tiny numerical residual.
    for p in outs:
        if p in minutes:
            minutes[p] = 0.0
    _reconcile_total(minutes, total_minutes, fixed_names=outs, cap=48.0)

    return minutes, pd.DataFrame(impact_rows)
