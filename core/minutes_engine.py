from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import numpy as np
import pandas as pd

from core.exposure import regulation_equivalent_minutes_row

from core.buckets import WeightConfig, split_non_overlapping, active_weights
from core.redistribution import learn_redistribution_matrix, apply_role_aware_overrides, apply_confirmed_outs


@dataclass
class MinutesProjection:
    player_id: object
    player_name: str
    team_abbr: str
    central_raw: float
    sd_raw: float
    low_raw: float
    high_raw: float
    starter_probability: float
    rotation_similarity: float
    regime: str


def resolve_minutes_sd(raw_sd: float, ratio: float = 1.0, source: str = "AUTO") -> float:
    """Resolve Monte-Carlo minute uncertainty without moving the central mean.

    The historical role-conditioned SD is the uncertainty estimate. Trader or
    metadata overrides change the central minute estimate only; they do not
    mechanically make the player more certain and are therefore *not* capped at
    2.25 minutes. Auto allocations may scale the historical SD mildly when the
    248-minute engine materially scales the role. OUT/outside-rotation states
    have zero simulated minute uncertainty.
    """
    src = str(source or "AUTO").upper()
    if src in {"OUT", "OUTSIDE ROTATION"}:
        return 0.0
    base = float(np.clip(float(raw_sd) if np.isfinite(raw_sd) else 2.0, 1.0, 5.5))
    if src in {"TRADER", "METADATA"}:
        return base
    scale = float(np.sqrt(np.clip(float(ratio), 0.6, 1.6)))
    return float(np.clip(base * scale, 1.0, 5.5))


def _truthy_starter(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "y", "starter", "start"}


def _case_lookup(d: dict, name: str, default=None):
    if not isinstance(d, dict):
        return default
    target = str(name).strip().casefold()
    for k, v in d.items():
        if str(k).strip().casefold() == target:
            return v
    return default


def _status(manual_context: dict, player_name: str) -> str:
    info = _case_lookup(manual_context.get("injuries", {}), player_name, {})
    if isinstance(info, dict):
        return str(info.get("status", "")).upper()
    return str(info or "").upper()


def _rotation_regime(manual_context: dict, team_name: str, team_abbr: str) -> str:
    # Accept either:
    # {"rotation_regime":{"LAS":"stable"}}
    # or the older nested game.rotation_regime structure.
    top = manual_context.get("rotation_regime", {})
    nested = manual_context.get("game", {}).get("rotation_regime", {})
    value = (
        _case_lookup(top, team_abbr, None)
        or _case_lookup(top, team_name, None)
        or _case_lookup(nested, team_abbr, None)
        or _case_lookup(nested, team_name, None)
    )
    v = str(value or "stable").strip().lower()
    return "role_change" if v in {"role_change", "change", "changed", "35/20/45"} else "stable"


def _projected_starters(manual_context: dict, team_name: str, team_abbr: str) -> set[str]:
    block = manual_context.get("projected_starters", {})
    vals = (
        _case_lookup(block, team_abbr, None)
        or _case_lookup(block, team_name, None)
        or []
    )
    return {str(x).strip().casefold() for x in vals}


def _team_game_margin(team_db: pd.DataFrame, game_id, team_abbr: str) -> float:
    rows = team_db[
        team_db["GAME_ID"].astype(str).eq(str(game_id))
    ].copy()
    if rows.empty or "PTS" not in rows.columns:
        return 0.0
    own = rows[rows["TEAM_ABBR"].astype(str).str.upper() == str(team_abbr).upper()]
    opp = rows[rows["TEAM_ABBR"].astype(str).str.upper() != str(team_abbr).upper()]
    if own.empty or opp.empty:
        return 0.0
    return float(own.iloc[0]["PTS"] - opp.iloc[0]["PTS"])


def _current_rotation_set(
    player_db: pd.DataFrame,
    current_pool: pd.DataFrame,
    team_abbr: str,
    manual_context: dict,
    min_rotation_avg: float = 4.0,
) -> set[str]:
    names = []
    team_pool = current_pool[
        current_pool["TEAM_ABBR"].astype(str).str.upper() == str(team_abbr).upper()
    ]
    for _, row in team_pool.iterrows():
        pname = str(row["PLAYER_NAME"])
        if _status(manual_context, pname) == "OUT":
            continue
        pid = row["PLAYER_ID"]
        log = player_db[player_db["PLAYER_ID"] == pid].sort_values("GAME_DATE")
        if log.empty:
            continue
        recent = pd.to_numeric(log.tail(5)["MIN"], errors="coerce").dropna()
        avg = float(recent.mean()) if len(recent) else 0.0
        override = _case_lookup(
            manual_context.get("projected_minutes", {}), pname, None
        )
        if avg >= min_rotation_avg or override is not None:
            names.append(pname.casefold())
    return set(names)


def _historical_rotation_sets(player_db: pd.DataFrame, team_abbr: str) -> Dict[str, set[str]]:
    x = player_db[
        player_db["TEAM_ABBR"].astype(str).str.upper() == str(team_abbr).upper()
    ].copy()
    out: Dict[str, set[str]] = {}
    for gid, g in x.groupby("GAME_ID"):
        # Rotation similarity should focus on players who actually occupied
        # meaningful rotation minutes, not 1-minute garbage-time appearances.
        meaningful = g[pd.to_numeric(g["MIN"], errors="coerce").fillna(0) >= 4.0]
        out[str(gid)] = set(
            meaningful["PLAYER_NAME"].astype(str).str.casefold().tolist()
        )
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _starter_probability(player_log: pd.DataFrame) -> float:
    if "STARTER" not in player_log.columns or player_log.empty:
        return 0.5
    recent = player_log.sort_values("GAME_DATE").tail(5)
    vals = recent["STARTER"].map(_truthy_starter).astype(float)
    return float(vals.mean()) if len(vals) else 0.5


def _weighted_bucket_minutes(
    bucket: pd.DataFrame,
    current_rotation: set[str],
    historical_rotations: Dict[str, set[str]],
    today_starter: Optional[bool],
    team_db: pd.DataFrame,
    team_abbr: str,
):
    if bucket.empty:
        return np.nan, np.nan, np.nan

    vals, weights, sims = [], [], []

    for _, row in bucket.iterrows():
        gid = str(row["GAME_ID"])
        hist_rotation = historical_rotations.get(gid, set())
        sim = _jaccard(current_rotation, hist_rotation)

        # Similarity is a modifier inside the already non-overlapping bucket.
        # It does NOT create another sample or change the bucket's outer weight.
        w = 0.55 + 0.45 * sim

        if today_starter is not None and "STARTER" in row:
            historical_starter = _truthy_starter(row["STARTER"])
            w *= 1.10 if historical_starter == today_starter else 0.88

        # OT adds schedule exposure rather than evidence that the normal 48-minute
        # rotation itself expanded. Normalize historical minutes to a regulation
        # equivalent instead of applying a fixed arbitrary OT weight penalty.

        margin = abs(_team_game_margin(team_db, gid, team_abbr))
        if margin >= 20:
            w *= 0.82
        elif margin >= 15:
            w *= 0.92

        m = float(regulation_equivalent_minutes_row(row))
        if not np.isfinite(m):
            continue
        vals.append(m)
        weights.append(max(w, 0.05))
        sims.append(sim)

    vals = np.asarray(vals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mean = float(np.average(vals, weights=weights))
    var = float(np.average((vals - mean) ** 2, weights=weights))
    similarity = float(np.average(np.asarray(sims), weights=weights))
    return mean, float(np.sqrt(max(var, 0.0))), similarity


def project_player_minutes_raw(
    player_log: pd.DataFrame,
    player_db: pd.DataFrame,
    team_db: pd.DataFrame,
    current_pool: pd.DataFrame,
    team_abbr: str,
    team_name: str,
    player_name: str,
    player_id,
    manual_context: dict,
) -> MinutesProjection:
    x = player_log.sort_values("GAME_DATE").copy()

    regime = _rotation_regime(manual_context, team_name, team_abbr)
    cfg = WeightConfig.role_change() if regime == "role_change" else WeightConfig.stable()

    buckets = split_non_overlapping(x)
    outer = active_weights(buckets, cfg)

    current_rotation = _current_rotation_set(
        player_db, current_pool, team_abbr, manual_context
    )
    historical_rotations = _historical_rotation_sets(player_db, team_abbr)

    manual_starters = _projected_starters(manual_context, team_name, team_abbr)
    if manual_starters:
        today_starter = player_name.casefold() in manual_starters
        starter_prob = 1.0 if today_starter else 0.0
    else:
        starter_prob = _starter_probability(x)
        today_starter = starter_prob >= 0.60

    bucket_stats = {}
    for key in ("old", "mid", "l5"):
        bucket_stats[key] = _weighted_bucket_minutes(
            buckets[key],
            current_rotation,
            historical_rotations,
            today_starter,
            team_db,
            team_abbr,
        )

    means, sds, sims, ws = [], [], [], []
    for key in ("old", "mid", "l5"):
        mean, sd, sim = bucket_stats[key]
        w = outer.get(key, 0.0)
        if w > 0 and np.isfinite(mean):
            means.append(mean)
            sds.append(sd if np.isfinite(sd) else 2.0)
            sims.append(sim if np.isfinite(sim) else 0.5)
            ws.append(w)

    if not means:
        recent = pd.to_numeric(x.tail(5)["MIN"], errors="coerce").dropna()
        mean = float(recent.mean()) if len(recent) else 0.0
        sd = float(recent.std(ddof=0)) if len(recent) > 1 else 2.0
        similarity = 0.5
    else:
        ws = np.asarray(ws, dtype=float)
        ws = ws / ws.sum()
        mean = float(np.sum(np.asarray(means) * ws))
        # combine within-bucket variability + disagreement between buckets
        within = float(np.sum((np.asarray(sds) ** 2) * ws))
        between = float(np.sum(((np.asarray(means) - mean) ** 2) * ws))
        sd = float(np.sqrt(within + between))
        similarity = float(np.sum(np.asarray(sims) * ws))

    # Robust stabilizer against a single odd rotation game.
    recent5 = pd.to_numeric(x.tail(5)["MIN"], errors="coerce").dropna()
    if len(recent5):
        median5 = float(recent5.median())
        mean = 0.82 * mean + 0.18 * median5

    mean = float(np.clip(mean, 0.0, 48.0))
    sd = float(np.clip(max(sd, 1.25), 1.25, 5.5))
    low = float(np.clip(mean - 1.20 * sd, 0.0, 48.0))
    high = float(np.clip(mean + 1.20 * sd, 0.0, 48.0))

    return MinutesProjection(
        player_id=player_id,
        player_name=player_name,
        team_abbr=team_abbr,
        central_raw=mean,
        sd_raw=sd,
        low_raw=low,
        high_raw=high,
        starter_probability=starter_prob,
        rotation_similarity=similarity,
        regime=regime,
    )


def _allocate_capped(
    raw: Dict[str, float],
    fixed: Dict[str, float],
    total_minutes: float = 240.0,
    cap: float = 48.0,
) -> Dict[str, float]:
    """
    Allocate remaining team minutes proportionally while honoring fixed trader
    overrides and the WNBA 248-minute regulation constraint.
    """
    out = {k: float(np.clip(v, 0.0, cap)) for k, v in fixed.items()}
    fixed_sum = sum(out.values())
    remaining = max(total_minutes - fixed_sum, 0.0)

    free = {k: max(float(v), 0.0) for k, v in raw.items() if k not in out}
    if not free:
        return out

    active = dict(free)
    allocated = {k: 0.0 for k in free}

    for _ in range(20):
        if remaining <= 1e-8 or not active:
            break
        denom = sum(active.values())
        if denom <= 0:
            share = remaining / len(active)
            proposals = {k: share for k in active}
        else:
            proposals = {
                k: remaining * v / denom
                for k, v in active.items()
            }

        hit_cap = []
        used = 0.0
        for k, proposal in proposals.items():
            room = cap - allocated[k]
            add = min(proposal, room)
            allocated[k] += add
            used += add
            if room - add <= 1e-8:
                hit_cap.append(k)

        remaining -= used
        for k in hit_cap:
            active.pop(k, None)

        if used <= 1e-10:
            break

    out.update(allocated)
    return out



def _context_without_outs(manual_context: dict, out_players: set[str]) -> dict:
    """Return a shallow-safe copy where selected OUT players are treated healthy.

    Used only to estimate the pre-injury rotation baseline before the learned
    replacement matrix redistributes the vacated minutes.
    """
    import copy
    ctx = copy.deepcopy(manual_context or {})
    injuries = ctx.setdefault("injuries", {})
    out_norm = {str(x).strip().casefold() for x in out_players}
    for key in list(injuries):
        if str(key).strip().casefold() in out_norm:
            info = injuries.get(key)
            if isinstance(info, dict):
                info = dict(info)
                info["status"] = "ACTIVE"
                injuries[key] = info
            else:
                injuries[key] = {"status": "ACTIVE"}
    return ctx


def _recent_team_minutes_summary(
    player_db: pd.DataFrame,
    team_db: pd.DataFrame,
    team_abbr: str,
    player_id,
    n5: int = 5,
    n10: int = 10,
) -> dict:
    """Pregame rotation evidence including DNPs as zero team-game minutes."""
    tg = team_db[
        team_db["TEAM_ABBR"].astype(str).str.upper() == str(team_abbr).upper()
    ].copy()
    if tg.empty:
        return {"l5": 0.0, "l10": 0.0, "apps5": 0, "apps10": 0}
    tg["GAME_DATE"] = pd.to_datetime(tg["GAME_DATE"], errors="coerce")
    tg["GAME_ID"] = tg["GAME_ID"].astype(str)
    tg = tg.sort_values("GAME_DATE").drop_duplicates("GAME_ID")
    recent10 = tg.tail(n10)

    pl = player_db[
        (player_db["PLAYER_ID"] == player_id)
        & (player_db["TEAM_ABBR"].astype(str).str.upper() == str(team_abbr).upper())
    ].copy()
    if pl.empty:
        vals = pd.Series(0.0, index=recent10["GAME_ID"].astype(str))
    else:
        pl["GAME_ID"] = pl["GAME_ID"].astype(str)
        minute_map = (
            pl.groupby("GAME_ID")["MIN"].sum().pipe(pd.to_numeric, errors="coerce").fillna(0.0).to_dict()
        )
        vals = recent10["GAME_ID"].astype(str).map(lambda gid: float(minute_map.get(gid, 0.0)))
        vals.index = recent10["GAME_ID"].astype(str).values
    vals10 = pd.to_numeric(vals, errors="coerce").fillna(0.0)
    vals5 = vals10.tail(n5)
    return {
        "l5": float(vals5.mean()) if len(vals5) else 0.0,
        "l10": float(vals10.mean()) if len(vals10) else 0.0,
        "apps5": int((vals5 >= 4.0).sum()),
        "apps10": int((vals10 >= 4.0).sum()),
    }


def _recent_rotation_size(
    player_db: pd.DataFrame,
    team_db: pd.DataFrame,
    team_abbr: str,
) -> int:
    """Median recent count of players with >=4 minutes, clipped to 8-10."""
    tg = team_db[
        team_db["TEAM_ABBR"].astype(str).str.upper() == str(team_abbr).upper()
    ].copy()
    if tg.empty:
        return 9
    tg["GAME_DATE"] = pd.to_datetime(tg["GAME_DATE"], errors="coerce")
    gids = tg.sort_values("GAME_DATE").drop_duplicates("GAME_ID").tail(10)["GAME_ID"].astype(str).tolist()
    p = player_db[
        (player_db["TEAM_ABBR"].astype(str).str.upper() == str(team_abbr).upper())
        & player_db["GAME_ID"].astype(str).isin(gids)
    ].copy()
    if p.empty:
        return 9
    p["MIN"] = pd.to_numeric(p["MIN"], errors="coerce").fillna(0.0)
    counts = p[p["MIN"] >= 4.0].groupby(p["GAME_ID"].astype(str)).size()
    if counts.empty:
        return 9
    return int(np.clip(round(float(counts.median())), 8, 10))


def _select_rotation_names(
    rows: list[dict],
    raw_scores: Dict[str, float],
    recent_info: Dict[str, dict],
    manual_context: dict,
    team_name: str,
    team_abbr: str,
    target_size: int,
    out_players: set[str],
) -> set[str]:
    """Choose today's meaningful rotation before enforcing 240 minutes."""
    starters = _projected_starters(manual_context, team_name, team_abbr)
    explicit = {str(k).strip().casefold() for k in (manual_context.get("projected_minutes", {}) or {})}
    candidates = []
    forced = set()
    for r in rows:
        name = str(r["Player"])
        if name in out_players:
            continue
        ri = recent_info.get(name, {})
        starter = name.casefold() in starters or float(r.get("Starter P", 0.0)) >= 0.60
        has_override = name.casefold() in explicit
        score = (
            0.50 * float(ri.get("l5", 0.0))
            + 0.25 * float(ri.get("l10", 0.0))
            + 0.25 * float(raw_scores.get(name, 0.0))
            + (4.0 if starter else 0.0)
        )
        active_evidence = (
            ri.get("apps5", 0) >= 2
            or ri.get("apps10", 0) >= 4
            or float(raw_scores.get(name, 0.0)) >= 10.0
        )
        if starter or has_override:
            forced.add(name)
        if active_evidence or starter or has_override:
            candidates.append((name, score))

    candidates.sort(key=lambda kv: kv[1], reverse=True)
    chosen = list(forced)
    for name, _ in candidates:
        if name not in chosen:
            chosen.append(name)
        if len(chosen) >= target_size:
            break
    # Safety fallback if recent data are sparse.
    if len(chosen) < min(target_size, 8):
        extras = sorted(
            [(n, v) for n, v in raw_scores.items() if n not in out_players and n not in chosen],
            key=lambda kv: kv[1], reverse=True,
        )
        for name, _ in extras:
            chosen.append(name)
            if len(chosen) >= target_size:
                break
    return set(chosen[:max(target_size, len(forced))])


def project_team_minutes(
    player_db: pd.DataFrame,
    team_db: pd.DataFrame,
    current_pool: pd.DataFrame,
    team_abbr: str,
    team_name: str,
    manual_context: dict,
    ui_overrides: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Project a realistic current rotation and enforce the 248-minute constraint.

    v2.13 separates three layers:
      1) HEALTHY current rotation: recent team-game minutes (including DNP zeros)
         + the existing non-overlap minutes model identify the likely 8-10 player
         rotation before any proportional normalization.
      2) CONFIRMED OUT: vacated healthy minutes are redistributed through the
         learned teammate replacement matrix. Position is only a fallback prior.
      3) Explicit minute restrictions/overrides: applied around the post-OUT
         baseline through the same matrix.

    This prevents a deep roster from scaling every star down simply because all
    current-roster players have historical minutes, and prevents OUT minutes from
    being shared indiscriminately across positions.
    """
    ui_overrides = ui_overrides or {}

    team_pool = current_pool[
        current_pool["TEAM_ABBR"].astype(str).str.upper()
        == str(team_abbr).upper()
    ].copy()

    out_players = {
        str(r["PLAYER_NAME"])
        for _, r in team_pool.iterrows()
        if _status(manual_context, str(r["PLAYER_NAME"])) == "OUT"
    }
    healthy_context = _context_without_outs(manual_context, out_players)

    rows = []
    raw_model = {}
    recent_info = {}

    # Estimate the pre-injury current role for every roster player. Confirmed
    # OUTs are deliberately treated healthy in this stage so their true vacated
    # minutes can be measured and redistributed rather than disappearing before
    # the replacement layer sees them.
    for _, prow in team_pool.iterrows():
        pname = str(prow["PLAYER_NAME"])
        pid = prow["PLAYER_ID"]
        status = _status(manual_context, pname)

        log = player_db[player_db["PLAYER_ID"] == pid].copy()
        if log.empty:
            continue

        proj = project_player_minutes_raw(
            log, player_db, team_db, current_pool,
            team_abbr, team_name, pname, pid, healthy_context
        )
        ri = _recent_team_minutes_summary(
            player_db, team_db, team_abbr, pid
        )
        recent_info[pname] = ri

        # DNP-aware team-game averages are the main guard against inflated bench
        # roles. The older per-player history still supplies role/starter context.
        if pname in out_players:
            # Vacated minutes should reflect the player's healthy role, not be
            # diluted by the zero-DNPs created by the very absence we are
            # trying to model. project_player_minutes_raw uses appearance
            # history and the healthy counterfactual rotation here.
            auto_raw = max(float(proj.central_raw), 0.0)
        else:
            auto_raw = (
                0.35 * max(float(proj.central_raw), 0.0)
                + 0.45 * float(ri["l5"])
                + 0.20 * float(ri["l10"])
            )
        raw_model[pname] = float(np.clip(auto_raw, 0.0, 48.0))

        rows.append({
            "PLAYER_ID": pid,
            "Player": pname,
            "Team": team_abbr,
            "Pos": prow.get("POSITION_ABBR"),
            "Status": status or "ACTIVE/UNK",
            "Raw Auto Min": proj.central_raw,
            "DNP-aware L5 Min": ri["l5"],
            "DNP-aware L10 Min": ri["l10"],
            "Raw SD": proj.sd_raw,
            "Raw Low": proj.low_raw,
            "Raw High": proj.high_raw,
            "Starter P": proj.starter_probability,
            "Rotation Similarity": proj.rotation_similarity,
            "Regime": proj.regime,
            "Source": "OUT" if status == "OUT" else "AUTO",
        })

    if not rows:
        return pd.DataFrame()

    target_size = _recent_rotation_size(player_db, team_db, team_abbr)
    chosen = _select_rotation_names(
        rows, raw_model, recent_info, healthy_context,
        team_name, team_abbr, target_size, out_players=set(),
    )
    # A confirmed OUT who was part of the healthy rotation must remain in the
    # healthy baseline so their minutes can be released. Force them in if their
    # healthy raw role was meaningful.
    for p in out_players:
        if raw_model.get(p, 0.0) >= 6.0:
            chosen.add(p)

    healthy_raw = {
        p: (raw_model[p] if p in chosen else 0.0)
        for p in raw_model
    }
    healthy_alloc = _allocate_capped(
        healthy_raw, {}, total_minutes=240.0, cap=48.0
    )

    matrix, matrix_audit = learn_redistribution_matrix(
        player_db, team_db, current_pool, team_abbr
    )

    # Confirmed OUT redistribution is learned/role-aware, not proportional.
    post_out_alloc, out_impact = apply_confirmed_outs(
        healthy_alloc,
        matrix,
        out_players=out_players,
        total_minutes=240.0,
    )

    # Explicit minute targets are separate current-state restrictions/returns.
    metadata = manual_context.get("projected_minutes", {}) or {}
    metadata_targets = {}
    for pname in post_out_alloc:
        v = _case_lookup(metadata, pname, None)
        if v is not None and pname not in out_players:
            metadata_targets[pname] = float(v)

    explicit_targets = dict(metadata_targets)
    for pname, val in ui_overrides.items():
        if pname in post_out_alloc and pname not in out_players:
            explicit_targets[pname] = float(val)

    final_alloc, override_impact = apply_role_aware_overrides(
        post_out_alloc,
        explicit_targets,
        matrix,
        out_players=out_players,
        total_minutes=240.0,
    )

    out = pd.DataFrame(rows)
    out["In Active Rotation"] = out["Player"].isin(chosen)
    out["Healthy Baseline Min"] = out["Player"].map(healthy_alloc).fillna(0.0)
    out["Auto Baseline Min"] = out["Player"].map(post_out_alloc).fillna(0.0)
    out["Projected Min"] = out["Player"].map(final_alloc).fillna(0.0)
    out["OUT Replacement Delta"] = out["Auto Baseline Min"] - out["Healthy Baseline Min"]
    out["Override Delta"] = out["Projected Min"] - out["Auto Baseline Min"]

    def source_for(name):
        if name in out_players:
            return "OUT"
        if name in ui_overrides and float(ui_overrides[name]) > 0:
            return "TRADER"
        if name in metadata_targets:
            return "METADATA"
        if name not in chosen:
            return "OUTSIDE ROTATION"
        return "AUTO"

    out["Source"] = out["Player"].map(source_for)

    ratio = np.where(
        pd.to_numeric(out["Raw Auto Min"], errors="coerce").fillna(0).to_numpy() > 0.5,
        out["Projected Min"].to_numpy() /
        np.maximum(out["Raw Auto Min"].to_numpy(), 0.5),
        1.0,
    )
    out["Minutes SD"] = [
        resolve_minutes_sd(raw_sd, r, src)
        for raw_sd, r, src in zip(
            pd.to_numeric(out["Raw SD"], errors="coerce").fillna(2.0),
            ratio,
            out["Source"].astype(str),
        )
    ]
    out["Minutes SD Method"] = np.where(
        out["Source"].isin(["TRADER", "METADATA"]),
        "historical conditional SD; override changes mean only",
        np.where(
            out["Source"].isin(["OUT", "OUTSIDE ROTATION"]),
            "none",
            "historical conditional SD + allocation scale",
        ),
    )
    out["Low Min"] = np.clip(
        out["Projected Min"] - 1.20 * out["Minutes SD"], 0, 40
    )
    out["High Min"] = np.clip(
        out["Projected Min"] + 1.20 * out["Minutes SD"], 0, 40
    )

    out.attrs["redistribution_matrix"] = matrix
    out.attrs["redistribution_matrix_audit"] = matrix_audit
    out.attrs["out_redistribution_impact"] = out_impact
    out.attrs["override_impact"] = override_impact
    out.attrs["target_rotation_size"] = target_size

    return out.sort_values(
        "Projected Min", ascending=False
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Public team-context helpers used by Team Markets
# ---------------------------------------------------------------------
def rotation_regime_for_team(
    manual_context: dict,
    team_name: str,
    team_abbr: str,
) -> str:
    """Public wrapper so Team Markets and Player Props use the same regime parser."""
    return _rotation_regime(manual_context, team_name, team_abbr)


def rotation_similarity_weights(
    player_db: pd.DataFrame,
    current_pool: pd.DataFrame,
    team_abbr: str,
    manual_context: dict,
) -> Dict[str, float]:
    """
    Current-rotation similarity modifier by historical GAME_ID.

    This is an INNER-bucket modifier only. It does not create another sample
    and therefore does not double count Old/G6-10/L5 outer weights.
    The same 0.55 + 0.45*Jaccard structure used by the minutes engine is reused
    here so team-market history is softly tilted toward games with a more
    comparable rotation, especially after fresh OUT/IN changes.
    """
    current = _current_rotation_set(
        player_db, current_pool, team_abbr, manual_context
    )
    historical = _historical_rotation_sets(player_db, team_abbr)
    out: Dict[str, float] = {}
    for gid, hist in historical.items():
        sim = _jaccard(current, hist)
        out[str(gid)] = 0.55 + 0.45 * sim
    return out


def residual_rotation_similarity_weights(
    player_db: pd.DataFrame,
    current_pool: pd.DataFrame,
    team_abbr: str,
    manual_context: dict,
    out_players: Optional[Iterable[str]] = None,
    residual_strength: float = 0.15,
) -> Dict[str, float]:
    """Weak residual rotation weighting for Team Markets.

    Confirmed OUT-player identity is handled by the separate exact availability
    state engine. To avoid counting the same absence twice, selected OUT names are
    REMOVED from both current and historical rotation sets before Jaccard is
    calculated. The residual rotation modifier is intentionally weak:
        (1-strength) + strength * Jaccard
    with default strength 0.15 => [0.85, 1.00].
    """
    out_norm = {str(x).strip().casefold() for x in (out_players or [])}
    current = _current_rotation_set(
        player_db, current_pool, team_abbr, manual_context
    ) - out_norm
    historical = _historical_rotation_sets(player_db, team_abbr)
    s = float(np.clip(residual_strength, 0.0, 0.35))
    out: Dict[str, float] = {}
    for gid, hist in historical.items():
        hist_resid = set(hist) - out_norm
        sim = _jaccard(current, hist_resid)
        out[str(gid)] = (1.0 - s) + s * sim
    return out


def h2h_rotation_similarity(
    player_db: pd.DataFrame,
    current_pool: pd.DataFrame,
    team_abbr: str,
    manual_context: dict,
    game_ids: Iterable[str],
) -> float:
    """Mean current-vs-H2H rotation similarity for H2H confidence.

    Unlike residual team-history weighting, confirmed OUT names are NOT removed
    here. If a player is OUT today but played in an old H2H, that H2H should be
    considered less comparable.
    """
    gids = {str(x) for x in game_ids}
    if not gids:
        return 0.0
    current = _current_rotation_set(
        player_db, current_pool, team_abbr, manual_context
    )
    historical = _historical_rotation_sets(player_db, team_abbr)
    sims = [_jaccard(current, historical.get(gid, set())) for gid in gids if gid in historical]
    return float(np.mean(sims)) if sims else 0.0
