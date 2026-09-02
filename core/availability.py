from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class AvailabilityAudit:
    team_abbr: str
    out_players: tuple[str, ...]
    eligibility_start: Optional[pd.Timestamp]
    eligible_games: int
    exact_games: int
    natural_share: float
    confidence: float
    target_share: float
    exact_weight: float
    other_weight: float
    k: float

    def as_dict(self) -> dict:
        return {
            "Team": self.team_abbr,
            "Confirmed OUT state": ", ".join(self.out_players) if self.out_players else "—",
            "Eligibility start": self.eligibility_start.date().isoformat() if self.eligibility_start is not None else "—",
            "Eligible games": self.eligible_games,
            "Exact-state games": self.exact_games,
            "Natural exact share": self.natural_share,
            "State confidence": self.confidence,
            "Target exact share": self.target_share,
            "Exact-game inner weight": self.exact_weight,
            "Other eligible inner weight": self.other_weight,
            "Shrink K": self.k,
        }


def _norm_name(v) -> str:
    return str(v).strip().casefold()


def _status(info) -> str:
    if isinstance(info, dict):
        return str(info.get("status", "")).strip().upper()
    return str(info or "").strip().upper()


def _player_team_from_pool(current_pool: pd.DataFrame, player_name: str) -> str:
    if current_pool is None or current_pool.empty:
        return ""
    hit = current_pool[
        current_pool["PLAYER_NAME"].astype(str).str.casefold() == _norm_name(player_name)
    ]
    if hit.empty:
        return ""
    return str(hit.iloc[0].get("TEAM_ABBR", "")).upper()


def confirmed_out_players(
    manual_context: dict,
    current_pool: pd.DataFrame,
    team_abbr: str,
) -> list[str]:
    """Return only CONFIRMED OUT players for the requested team.

    QUESTIONABLE/GTD/DOUBTFUL are intentionally ignored by the availability-state
    engine. The user explicitly decides when a player becomes OUT.
    """
    injuries = (manual_context or {}).get("injuries", {}) or {}
    out = []
    for name, info in injuries.items():
        if _status(info) != "OUT":
            continue
        info_team = str(info.get("team", "") if isinstance(info, dict) else "").upper()
        pool_team = _player_team_from_pool(current_pool, str(name))
        if info_team == str(team_abbr).upper() or pool_team == str(team_abbr).upper():
            out.append(str(name))
    # Stable order for audit/session reproducibility.
    return sorted(set(out), key=str.casefold)


def _first_team_appearance(
    player_db: pd.DataFrame,
    team_abbr: str,
    player_name: str,
) -> Optional[pd.Timestamp]:
    if player_db is None or player_db.empty:
        return None
    x = player_db[
        player_db["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())
        & player_db["PLAYER_NAME"].astype(str).str.casefold().eq(_norm_name(player_name))
    ].copy()
    if x.empty:
        return None
    dates = pd.to_datetime(x["GAME_DATE"], errors="coerce").dropna()
    return dates.min() if len(dates) else None


def _historical_presence(
    player_db: pd.DataFrame,
    team_abbr: str,
    player_names: Iterable[str],
    min_minutes: float = 1.0,
) -> Dict[str, set[str]]:
    names = {_norm_name(x) for x in player_names}
    if not names or player_db is None or player_db.empty:
        return {}
    x = player_db[
        player_db["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())
        & player_db["PLAYER_NAME"].astype(str).str.casefold().isin(names)
    ].copy()
    mins = pd.to_numeric(x.get("MIN", 0), errors="coerce").fillna(0.0)
    x = x[mins >= float(min_minutes)].copy()
    out: Dict[str, set[str]] = {}
    for gid, g in x.groupby("GAME_ID"):
        out[str(gid)] = set(g["PLAYER_NAME"].astype(str).str.casefold().tolist())
    return out


def availability_state_weights(
    player_db: pd.DataFrame,
    team_log: pd.DataFrame,
    team_abbr: str,
    out_players: Iterable[str],
    k: float = 6.0,
    max_target_share: float = 0.70,
    exclude_opponent_abbr: str | None = None,
) -> Tuple[Dict[str, float], pd.DataFrame, set[str]]:
    """Exact confirmed-OUT state reweighting with disjoint evidence groups.

    The historical games are divided into two DISJOINT groups once every selected
    OUT player had first appeared for this team:
      exact: all selected OUT players absent from that game
      other: at least one selected OUT player played

    No fourth sample is created. Instead, exact/other rows receive inner weights
    whose total exact-state share is driven by N/(N+K), but is never lower than
    the natural historical exact-state share. This is transparent partial pooling.

    Games before all selected players had first appeared for the team are neutral
    (weight 1.0) rather than falsely classified as 'OUT' games.
    """
    out_players = tuple(sorted({str(x) for x in out_players if str(x).strip()}, key=str.casefold))
    neutral_audit = AvailabilityAudit(
        team_abbr=str(team_abbr).upper(), out_players=out_players,
        eligibility_start=None, eligible_games=0, exact_games=0,
        natural_share=0.0, confidence=0.0, target_share=0.0,
        exact_weight=1.0, other_weight=1.0, k=float(k),
    )
    if not out_players or team_log is None or team_log.empty:
        return {}, pd.DataFrame([neutral_audit.as_dict()]), set()

    starts = [_first_team_appearance(player_db, team_abbr, p) for p in out_players]
    if any(s is None for s in starts):
        # If we cannot establish that every selected player was actually on this
        # team historically, do not treat pre-roster games as injury matches.
        return {}, pd.DataFrame([neutral_audit.as_dict()]), set()

    eligibility_start = max(starts)
    t = team_log.copy()
    t["_DATE"] = pd.to_datetime(t["GAME_DATE"], errors="coerce")
    eligible = t[t["_DATE"] >= eligibility_start].copy()
    if exclude_opponent_abbr and "OPP_ABBR" in eligible.columns:
        eligible = eligible[
            ~eligible["OPP_ABBR"].astype(str).str.upper().eq(str(exclude_opponent_abbr).upper())
        ].copy()
    if eligible.empty:
        audit = neutral_audit
        audit.eligibility_start = eligibility_start
        return {}, pd.DataFrame([audit.as_dict()]), set()

    selected_norm = {_norm_name(x) for x in out_players}
    presence = _historical_presence(player_db, team_abbr, out_players, min_minutes=1.0)

    exact_ids: set[str] = set()
    eligible_ids = [str(g) for g in eligible["GAME_ID"].astype(str).tolist()]
    for gid in eligible_ids:
        played = presence.get(str(gid), set())
        if selected_norm.isdisjoint(played):
            exact_ids.add(str(gid))

    n = len(eligible_ids)
    n_exact = len(exact_ids)
    n_other = n - n_exact
    if n_exact == 0 or n_other == 0:
        audit = AvailabilityAudit(
            team_abbr=str(team_abbr).upper(), out_players=out_players,
            eligibility_start=eligibility_start, eligible_games=n,
            exact_games=n_exact, natural_share=(n_exact / n if n else 0.0),
            # No contrast group => no identifiable exact-state effect. Keep
            # confidence neutral so the roster-synthetic fallback is not muted.
            confidence=0.0,
            target_share=(n_exact / n if n else 0.0),
            exact_weight=1.0, other_weight=1.0, k=float(k),
        )
        return {}, pd.DataFrame([audit.as_dict()]), exact_ids

    natural = n_exact / n
    confidence = n_exact / (n_exact + float(k))
    target = min(float(max_target_share), max(float(natural), float(confidence)))

    # Choose per-row weights so exact-state rows sum to target*n and other rows
    # sum to (1-target)*n. Mean eligible inner weight therefore stays exactly 1.
    w_exact = target * n / n_exact
    w_other = (1.0 - target) * n / n_other

    weights: Dict[str, float] = {}
    for gid in eligible_ids:
        weights[str(gid)] = float(w_exact if str(gid) in exact_ids else w_other)

    audit = AvailabilityAudit(
        team_abbr=str(team_abbr).upper(), out_players=out_players,
        eligibility_start=eligibility_start, eligible_games=n,
        exact_games=n_exact, natural_share=natural, confidence=confidence,
        target_share=target, exact_weight=float(w_exact),
        other_weight=float(w_other), k=float(k),
    )
    return weights, pd.DataFrame([audit.as_dict()]), exact_ids



# ---------------------------------------------------------------------------
# v2.15 similarity-state engine
# ---------------------------------------------------------------------------

_STAT_SOURCE = {
    "PTS": "PTS", "FGA": "FGA", "3PA": "FG3A", "FTA": "FTA",
    "AST": "AST", "REB": "REB", "OREB": "OREB", "DREB": "DREB",
    "TOV": "TOV", "PF": "PF", "STL": "STL", "BLK": "BLK",
    # Efficiency-state relevance is driven by the volume that changes the mix.
    "3P_PCT": "FG3A", "2P_PCT": "FGA", "3P_SHARE": "FG3A",
}

def _broad_pos_from_row(row) -> str:
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

def _player_position(player_db: pd.DataFrame, team_abbr: str, player_name: str, current_pool=None) -> str:
    if current_pool is not None and not current_pool.empty:
        hit = current_pool[
            current_pool["PLAYER_NAME"].astype(str).str.casefold().eq(_norm_name(player_name))
            & current_pool["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())
        ]
        if not hit.empty:
            return _broad_pos_from_row(hit.iloc[0])
    x = player_db[
        player_db["PLAYER_NAME"].astype(str).str.casefold().eq(_norm_name(player_name))
        & player_db["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())
    ].copy()
    if x.empty:
        return ""
    x["GAME_DATE"] = pd.to_datetime(x["GAME_DATE"], errors="coerce")
    return _broad_pos_from_row(x.sort_values("GAME_DATE").iloc[-1])

def _stat_position_compat(stat: str, focal_pos: str, absent_pos: str) -> float:
    """Structural prior only; actual near-state games determine the outcome rate.

    The prior follows the manual model logic built during review: creation transfers
    mostly among handlers/wings, rebound opportunity mostly within the frontcourt,
    and shot volume is more portable than either. It is deliberately broad and is
    only used to decide how relevant an absence is for the focal stat.
    """
    f, a = str(focal_pos or ""), str(absent_pos or "")
    if not f or not a:
        return 0.45
    if f == a:
        return 1.0
    pair = {f, a}
    stat = str(stat).upper()
    if stat in {"REB", "OREB", "DREB"}:
        return 0.90 if pair == {"F", "C"} else (0.40 if pair == {"G", "F"} else 0.18)
    if stat in {"AST", "TOV"}:
        return 0.62 if pair == {"G", "F"} else (0.28 if pair == {"F", "C"} else 0.12)
    if stat in {"3PA", "3P_PCT", "3P_SHARE"}:
        return 0.78 if pair == {"G", "F"} else (0.32 if pair == {"F", "C"} else 0.12)
    if stat in {"FGA", "PTS", "FTA", "2P_PCT"}:
        return 0.68 if pair == {"G", "F"} else (0.72 if pair == {"F", "C"} else 0.22)
    return 0.55 if pair in ({"G", "F"}, {"F", "C"}) else 0.25

def _player_event_volume(player_db: pd.DataFrame, team_abbr: str, player_name: str, stat: str) -> float:
    """Robust current-season event volume used only as an absence-relevance prior.

    It is NOT added to Old/G6-10/L5 outcomes. Using a median active-minute role
    times the player's season per-minute rate avoids counting injury DNP zeros as
    evidence that a currently absent player had no role.
    """
    src = _STAT_SOURCE.get(str(stat).upper(), str(stat).upper())
    x = player_db[
        player_db["TEAM_ABBR"].astype(str).str.upper().eq(str(team_abbr).upper())
        & player_db["PLAYER_NAME"].astype(str).str.casefold().eq(_norm_name(player_name))
    ].copy()
    if x.empty or src not in x.columns:
        return 0.0
    mins = pd.to_numeric(x["MIN"], errors="coerce").fillna(0.0)
    vals = pd.to_numeric(x[src], errors="coerce").fillna(0.0)
    active = mins >= 1.0
    if not active.any():
        return 0.0
    total_min = float(mins[active].sum())
    rate = float(vals[active].sum()) / max(total_min, 1.0)
    role_min = float(np.median(mins[active]))
    return max(rate * role_min, 0.0)

def _absence_relevance(
    player_db: pd.DataFrame, team_abbr: str, out_players: Iterable[str], stat: str,
    current_pool: pd.DataFrame | None = None, focal_player: str | None = None,
) -> tuple[Dict[str, float], Dict[str, float], list[str]]:
    """Return normalized relevance only for materially related absences.

    v2.16: the focal player's near-state no longer conditions on every selected
    OUT simply because they are on the injury list.  For player props we first
    form a stat-specific relevance score from event volume x positional/role
    compatibility, then remove weak tail absences before normalizing.  This keeps
    e.g. a frontcourt REB sample from being diluted by unrelated creator absences.

    Team-level calls (no focal_player) keep every OUT because there is no focal
    position to define a role family.
    """
    outs = [str(x) for x in out_players if str(x).strip()]
    if not outs:
        return {}, {}, []
    focal_pos = _player_position(player_db, team_abbr, focal_player, current_pool) if focal_player else ""
    raw = {}
    compats = {}
    for name in outs:
        volume = _player_event_volume(player_db, team_abbr, name, stat)
        # sqrt keeps one high-volume star from making every other absence irrelevant.
        volume_term = np.sqrt(max(volume, 0.03))
        if focal_player:
            apos = _player_position(player_db, team_abbr, name, current_pool)
            compat = _stat_position_compat(stat, focal_pos, apos)
        else:
            compat = 1.0
        compats[name] = float(compat)
        raw[name] = max(float(volume_term * compat), 1e-4)

    if not focal_player:
        kept = list(outs)
    else:
        stat_u = str(stat).upper()
        # Positional floor keeps only plausible role-family transfers.  A second
        # relative-score floor prevents a tiny same-family role from remaining
        # merely because its broad position label matches.
        compat_floor = {
            "REB": 0.50, "OREB": 0.55, "DREB": 0.50,
            "AST": 0.45, "TOV": 0.45,
            "3PA": 0.50, "3P_PCT": 0.50, "3P_SHARE": 0.50,
            "FGA": 0.40, "PTS": 0.40, "FTA": 0.40, "2P_PCT": 0.40,
        }.get(stat_u, 0.40)
        mx = max(raw.values()) if raw else 0.0
        rel_floor = 0.38 * mx
        kept = [
            n for n in outs
            if compats.get(n, 0.0) >= compat_floor and raw.get(n, 0.0) >= rel_floor
        ]
        # Never return an empty relevant state: retain the single strongest OUT.
        if not kept and raw:
            kept = [max(raw, key=raw.get)]

    selected_raw = {n: raw[n] for n in kept}
    den = sum(selected_raw.values())
    if den <= 0:
        relevance = {n: 1.0 / len(kept) for n in kept}
    else:
        relevance = {n: float(v / den) for n, v in selected_raw.items()}
    excluded = [n for n in outs if n not in kept]
    return relevance, raw, excluded

def availability_similarity_weight_maps(
    player_db: pd.DataFrame,
    team_log: pd.DataFrame,
    team_abbr: str,
    out_players: Iterable[str],
    stats: Iterable[str],
    current_pool: pd.DataFrame | None = None,
    focal_player: str | None = None,
    k: float = 6.0,
    maturity_games: float = 5.0,
    temperature: float = 1.5,
    exclude_opponent_abbr: str | None = None,
):
    """Single-score-per-game near-state weighting, separately by stat.

    Every historical game receives exactly ONE similarity score for a given stat.
    There are no nested 4/5 -> 3/5 -> 2/5 samples, so a 4/5 game can never be
    counted again as a 3/5 or 2/5 observation. The resulting map is only an
    INNER weight inside Old/G6-10/L5.

    Small samples are not hard-zeroed. Evidence mass is sum(similarity**2); five
    fully comparable games are treated as a maturity point, while 1-4 games are
    progressively shrunk toward neutral. This is a conservative partial-pooling
    rule, not a new outer sample.
    """
    outs = tuple(sorted({str(x) for x in out_players if str(x).strip()}, key=str.casefold))
    stats = tuple(dict.fromkeys(str(s).upper() for s in stats))
    if not outs or team_log is None or team_log.empty:
        audit = pd.DataFrame([{
            "Stat": s, "Confirmed OUT state": "—", "Eligible games": 0,
            "Exact-state games": 0, "Evidence mass": 0.0, "Maturity": 0.0,
            "State confidence": 0.0, "Mean similarity": 0.0, "Max similarity": 0.0,
        } for s in stats])
        return {s: {} for s in stats}, audit, {s: {} for s in stats}

    t = team_log.copy()
    t["_DATE"] = pd.to_datetime(t["GAME_DATE"], errors="coerce")
    if exclude_opponent_abbr and "OPP_ABBR" in t.columns:
        t = t[~t["OPP_ABBR"].astype(str).str.upper().eq(str(exclude_opponent_abbr).upper())].copy()
    gids = t["GAME_ID"].astype(str).tolist()
    dates = dict(zip(t["GAME_ID"].astype(str), t["_DATE"]))

    presence = _historical_presence(player_db, team_abbr, outs, min_minutes=1.0)
    starts = {name: _first_team_appearance(player_db, team_abbr, name) for name in outs}

    maps, audits, score_maps = {}, [], {}
    for stat in stats:
        relevance, relevance_raw, relevance_excluded = _absence_relevance(
            player_db, team_abbr, outs, stat, current_pool=current_pool, focal_player=focal_player
        )
        scores = {}
        for gid in gids:
            played = presence.get(str(gid), set())
            gdate = dates.get(str(gid))
            s = 0.0
            eligible_rel = {
                name: rel for name, rel in relevance.items()
                if starts.get(name) is not None and pd.notna(gdate) and gdate >= starts.get(name)
            }
            rel_den = sum(eligible_rel.values())
            if rel_den <= 0:
                scores[str(gid)] = 0.0
                continue
            for name, rel in eligible_rel.items():
                absent = _norm_name(name) not in played
                if absent:
                    s += rel / rel_den
            scores[str(gid)] = float(np.clip(s, 0.0, 1.0))

        arr = np.asarray(list(scores.values()), dtype=float)
        evidence = float(np.sum(arr ** 2))
        mature = float(np.clip(evidence / max(float(maturity_games), 1e-6), 0.0, 1.0))
        empirical_conf = float(evidence / (evidence + max(float(k), 1e-6))) if evidence > 0 else 0.0
        confidence = float(np.clip(empirical_conf * mature, 0.0, 1.0))
        mean_s = float(np.mean(arr)) if len(arr) else 0.0

        # Kernel tilt around the natural mean. Mean weight is normalized to 1,
        # preserving the outer Old/G6-10/L5 sample weights exactly.
        raw_w = {gid: float(np.exp(float(temperature) * confidence * (sc - mean_s))) for gid, sc in scores.items()}
        mean_w = float(np.mean(list(raw_w.values()))) if raw_w else 1.0
        weights = {gid: float(np.clip(w / max(mean_w, 1e-9), 0.35, 2.85)) for gid, w in raw_w.items()}

        exact = sum(1 for v in scores.values() if v >= 0.999999)
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        audits.append({
            "Stat": stat,
            "Focal player": focal_player or "TEAM",
            "Confirmed OUT state": ", ".join(outs),
            "Relevant OUT state": ", ".join(relevance.keys()) if relevance else "—",
            "Excluded low-relevance OUT": ", ".join(relevance_excluded) if relevance_excluded else "—",
            "Eligible games": int(len(scores)),
            "Exact-state games": int(exact),
            "Evidence mass": evidence,
            "Maturity": mature,
            "State confidence": confidence,
            "Mean similarity": mean_s,
            "Max similarity": float(np.max(arr)) if len(arr) else 0.0,
            "Top near-state games": ", ".join(f"{gid}:{sc:.2f}" for gid, sc in top),
            "Relevance": ", ".join(f"{n}:{w:.2f}" for n, w in sorted(relevance.items(), key=lambda kv: kv[1], reverse=True)),
            "Raw relevance": ", ".join(f"{n}:{w:.3f}" for n, w in sorted(relevance_raw.items(), key=lambda kv: kv[1], reverse=True)),
            "Shrink K": float(k),
            "Maturity games": float(maturity_games),
        })
        maps[stat] = weights
        score_maps[stat] = scores

    return maps, pd.DataFrame(audits), score_maps

def confidence_by_stat(audit: pd.DataFrame) -> Dict[str, float]:
    if audit is None or audit.empty:
        return {}
    out = {}
    for _, r in audit.iterrows():
        stat = str(r.get("Stat", "")).upper()
        if stat:
            out[stat] = float(pd.to_numeric(pd.Series([r.get("State confidence", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    return out

def combine_stat_weight_maps(
    stat_maps: Dict[str, Dict[str, float]],
    common_map: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Combine a stat-specific availability map with one common residual map.

    Each GAME_ID still occurs only once per stat. This is multiplication of two
    different relevance dimensions, not nested absence-state sampling.
    """
    if not stat_maps:
        return {}
    return {stat: combine_game_weights(w, common_map) for stat, w in stat_maps.items()}

def combine_game_weights(*weight_maps: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Multiply independent INNER-bucket relevance weights.

    This function deliberately does not introduce any outer sample weight. The
    Old/G6-10/L5 weights remain unchanged. Missing ids are neutral (1.0).
    """
    maps = [m for m in weight_maps if m]
    if not maps:
        return {}
    ids = set().union(*(set(m.keys()) for m in maps))
    out = {}
    for gid in ids:
        w = 1.0
        for m in maps:
            w *= float(m.get(str(gid), 1.0))
        out[str(gid)] = float(np.clip(w, 0.20, 4.00))
    return out
