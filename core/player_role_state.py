from __future__ import annotations

"""Adaptive latent player-role state for per-minute opportunity rates.

The stable player model deliberately keeps non-overlapping Old/G6-10/L5 buckets.
That is a good prior when a player's role is stationary, but it can lag a genuine
within-season role change (for example, a sustained creation or shot-volume shift
that is not caused by a confirmed teammate absence).

This module adds a *separate* empirical state-space correction.  It does not
replace minutes, opponent, availability, H2H, or shooting-skill layers.

For each opportunity count C_t observed in M_t minutes we use a Poisson-rate
approximation on the log scale:

    z_t = log((C_t + 1/2) / M_t)
    z_t ~ N(theta_t, 1 / (C_t + 1/2))
    theta_t = theta_{t-1} + eta_t,  eta_t ~ N(0, q)

The +1/2 is the standard Jeffreys-style small-count stabilization for a Poisson
rate.  The process variance q is NOT hand-set: it is selected by one-step-ahead
walk-forward predictive likelihood over a data-scaled grid.  q=0 is included,
so a stable player is allowed to remain static.

To avoid automatically trusting an extra dynamic parameter, we use Akaike model
averaging between the q=0 (static) and best-q (dynamic) filters.  The final
correction is therefore near 1 when the dynamic model does not improve predictive
fit, and it grows only when the player's own history supports a changing latent
opportunity state.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RoleStateResult:
    feature: str
    games: int
    static_rate: float
    dynamic_rate: float
    applied_rate: float
    modifier: float
    q_selected: float
    nll_static: float
    nll_dynamic: float
    dynamic_weight: float
    z_change: float
    active: bool
    note: str


def _row_weights(df: pd.DataFrame, game_weights: Optional[Dict[str, float]]) -> np.ndarray:
    if not game_weights or "GAME_ID" not in df.columns:
        return np.ones(len(df), dtype=float)
    return np.asarray(
        [max(float(game_weights.get(str(gid), 1.0)), 0.0) for gid in df["GAME_ID"]],
        dtype=float,
    )


def _numeric_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(float)


def _count_for_feature(df: pd.DataFrame, feature: str) -> np.ndarray:
    fga = _numeric_col(df, "FGA")
    a3 = _numeric_col(df, "FG3A")
    if feature == "two_pa_pm":
        return np.maximum(fga - a3, 0.0)
    if feature == "three_pa_pm":
        return np.maximum(a3, 0.0)
    col = {
        "fta_pm": "FTA",
        "reb_pm": "REB",
        "ast_pm": "AST",
    }.get(feature)
    if col is None:
        raise KeyError(f"Unsupported role-state feature: {feature}")
    return np.maximum(_numeric_col(df, col), 0.0)


def _observations(
    df: pd.DataFrame,
    feature: str,
    game_weights: Optional[Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df.sort_values("GAME_DATE").copy()
    mins = pd.to_numeric(x["MIN"], errors="coerce").fillna(0.0).to_numpy(float)
    counts = _count_for_feature(x, feature)
    w = _row_weights(x, game_weights)

    keep = np.isfinite(mins) & np.isfinite(counts) & np.isfinite(w) & (mins >= 4.0) & (w > 0)
    mins = mins[keep]
    counts = counts[keep]
    w = w[keep]

    # Inner availability-similarity weights alter information mass, not the
    # observed per-minute rate itself.
    exposure = mins * w
    eff_count = counts * w
    z = np.log((eff_count + 0.5) / np.maximum(exposure, 1e-9))
    r = 1.0 / (eff_count + 0.5)
    return z, r, exposure


def _filter(z: np.ndarray, r: np.ndarray, q: float, warmup: int = 5):
    n = len(z)
    if n == 0:
        return np.nan, np.nan, np.inf

    # Diffuse but data-scaled initialization.  It is washed out before scoring
    # because predictive likelihood begins only after the warmup games.
    m = float(z[0])
    if n > 1:
        init_var = float(np.nanvar(z[: min(n, warmup)]))
    else:
        init_var = float(r[0])
    p = max(init_var, float(r[0]), 1e-6)
    nll = 0.0

    for t in range(1, n):
        a = m
        R = p + q
        fvar = max(R + float(r[t]), 1e-9)
        err = float(z[t] - a)
        if t >= warmup:
            nll += 0.5 * (np.log(2.0 * np.pi * fvar) + err * err / fvar)
        k = R / fvar
        m = a + k * err
        p = max((1.0 - k) * R, 1e-9)
    return float(m), float(p), float(nll)


def _q_grid(z: np.ndarray) -> np.ndarray:
    if len(z) < 3:
        return np.asarray([0.0])
    d = np.diff(z)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.asarray([0.0])
    # Robust scale from the player's own sequential changes.  No fixed role-
    # change weight is encoded here; the candidate range expands/contracts with
    # the observed series itself.
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    robust_var = max((1.4826 * mad) ** 2, float(np.var(d)) * 0.25, 1e-5)
    upper = max(robust_var * 2.0, 1e-4)
    positive = np.geomspace(max(upper / 1000.0, 1e-6), upper, 28)
    return np.unique(np.concatenate(([0.0], positive)))


def adaptive_role_state(
    df: pd.DataFrame,
    feature: str,
    baseline_rate: float,
    game_weights: Optional[Dict[str, float]] = None,
    min_games: int = 12,
) -> RoleStateResult:
    """Return an evidence-weighted dynamic modifier for one opportunity rate."""
    z, r, _ = _observations(df, feature, game_weights)
    n = len(z)
    base = float(baseline_rate) if np.isfinite(baseline_rate) and baseline_rate > 0 else np.nan
    if n < min_games or not np.isfinite(base):
        return RoleStateResult(
            feature, n, base, base, base, 1.0, 0.0, np.nan, np.nan, 0.0, 0.0,
            False, f"Neutral: need >= {min_games} usable games for dynamic role-state inference.",
        )

    m0, p0, nll0 = _filter(z, r, 0.0)
    best = (nll0, 0.0, m0, p0)
    for q in _q_grid(z)[1:]:
        m, p, nll = _filter(z, r, float(q))
        if nll < best[0]:
            best = (nll, float(q), m, p)

    nll1, q1, m1, p1 = best

    # One extra process-variance parameter: Akaike weight protects stable
    # players from an unnecessary dynamic model while remaining fully
    # data-driven.  Equal prior support for static/dynamic models.
    score_static = float(nll0)                 # k=0
    score_dynamic = float(nll1 + (1.0 if q1 > 0 else 0.0))  # NLL + k
    delta = np.clip(score_dynamic - score_static, -60.0, 60.0)
    dyn_w = float(1.0 / (1.0 + np.exp(delta))) if q1 > 0 else 0.0

    log_static = float(m0)
    log_dynamic = float(m1)
    log_applied = log_static + dyn_w * (log_dynamic - log_static)
    static_rate = float(np.exp(log_static))
    dynamic_rate = float(np.exp(log_dynamic))
    applied_state_rate = float(np.exp(log_applied))

    # The role-state layer is a correction to the existing non-overlapping
    # baseline, not a replacement for it.  Compare dynamic vs static latent
    # states so the stable model is exactly neutral when no temporal drift is
    # supported.
    modifier = float(applied_state_rate / max(static_rate, 1e-12))
    applied_rate = float(base * modifier)
    z_change = float((log_dynamic - log_static) / np.sqrt(max(p0 + p1, 1e-9)))
    active = bool(q1 > 0 and dyn_w > 0.05 and abs(np.log(max(modifier, 1e-12))) > 0.005)

    note = (
        "Adaptive local-level state applied from walk-forward predictive evidence."
        if active else
        "Dynamic candidate did not earn enough predictive weight; baseline effectively retained."
    )
    return RoleStateResult(
        feature=feature,
        games=n,
        static_rate=static_rate,
        dynamic_rate=dynamic_rate,
        applied_rate=applied_rate,
        modifier=modifier,
        q_selected=q1,
        nll_static=float(nll0),
        nll_dynamic=float(nll1),
        dynamic_weight=dyn_w,
        z_change=z_change,
        active=active,
        note=note,
    )


def role_state_table(results) -> pd.DataFrame:
    rows = []
    labels = {
        "two_pa_pm": "2PA/min",
        "three_pa_pm": "3PA/min",
        "fta_pm": "FTA/min",
        "reb_pm": "REB/min",
        "ast_pm": "AST/min",
    }
    for r in results:
        rows.append({
            "Opportunity": labels.get(r.feature, r.feature),
            "Usable games": r.games,
            "Static latent rate": r.static_rate,
            "Dynamic latent rate": r.dynamic_rate,
            "Applied profile rate": r.applied_rate,
            "Applied modifier": r.modifier,
            "Selected q": r.q_selected,
            "Static predictive NLL": r.nll_static,
            "Dynamic predictive NLL": r.nll_dynamic,
            "Dynamic model weight": r.dynamic_weight,
            "State-change z": r.z_change,
            "Active": r.active,
            "Note": r.note,
        })
    return pd.DataFrame(rows)
