from __future__ import annotations

"""Optional, calibrated market-margin prior.

The market spread is treated as a second forecast of final scoring margin, not
as ground truth.  A convex combination weight is estimated on historical games
and accepted only if it improves a later chronological holdout.  Without a
calibration file the market has ZERO numerical influence.

When active, the already-coherent joint Monte Carlo is exponentially tilted to
match the blended target margin.  The same simulation-row index is used for
both teams, so possession/stat correlations and box-score identities are
preserved rather than editing individual markets by hand.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class MarketMarginCalibration:
    active: bool = False
    weight: float = 0.0
    rows: int = 0
    train_rows: int = 0
    holdout_rows: int = 0
    model_holdout_rmse: float = np.nan
    combined_holdout_rmse: float = np.nan
    market_holdout_rmse: float = np.nan
    reason: str = "no calibration data"


def _rmse(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    if not np.any(good):
        return np.nan
    return float(np.sqrt(np.mean(np.square(a[good] - b[good]))))


def fit_market_margin_calibration(df: pd.DataFrame) -> MarketMarginCalibration:
    """Fit a convex model/market margin weight on a chronological split.

    Required columns:
      MODEL_HOME_MARGIN
      MARKET_HOME_SPREAD   (sportsbook convention: home -6.5 => -6.5)
      ACTUAL_HOME_MARGIN

    Optional GAME_DATE controls chronology; otherwise input row order is used.
    """
    if df is None or df.empty:
        return MarketMarginCalibration()
    x = df.copy()
    req = ["MODEL_HOME_MARGIN", "MARKET_HOME_SPREAD", "ACTUAL_HOME_MARGIN"]
    if not all(c in x.columns for c in req):
        return MarketMarginCalibration(rows=len(x), reason="missing required calibration columns")
    for c in req:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=req).copy()
    if "GAME_DATE" in x.columns:
        x["GAME_DATE"] = pd.to_datetime(x["GAME_DATE"], errors="coerce")
        x = x.sort_values("GAME_DATE", na_position="last")
    if len(x) < 30:
        return MarketMarginCalibration(rows=len(x), reason="need >=30 historical games")

    model = x["MODEL_HOME_MARGIN"].to_numpy(dtype=float)
    # A home spread of -6.5 means the market forecast margin is +6.5.
    market = -x["MARKET_HOME_SPREAD"].to_numpy(dtype=float)
    actual = x["ACTUAL_HOME_MARGIN"].to_numpy(dtype=float)

    cut = max(20, int(len(x) * 0.70))
    if len(x) - cut < 8:
        return MarketMarginCalibration(rows=len(x), reason="insufficient later holdout")

    d = market[:cut] - model[:cut]
    y = actual[:cut] - model[:cut]
    denom = float(np.sum(d * d))
    if denom <= 1e-9:
        w = 0.0
    else:
        w = float(np.clip(np.sum(d * y) / denom, 0.0, 1.0))

    pred_comb = model[cut:] + w * (market[cut:] - model[cut:])
    rm_model = _rmse(actual[cut:], model[cut:])
    rm_comb = _rmse(actual[cut:], pred_comb)
    rm_market = _rmse(actual[cut:], market[cut:])
    active = bool(np.isfinite(rm_model) and np.isfinite(rm_comb) and rm_comb < rm_model and w > 0)
    if not active:
        w = 0.0
    return MarketMarginCalibration(
        active=active,
        weight=w,
        rows=len(x),
        train_rows=cut,
        holdout_rows=len(x)-cut,
        model_holdout_rmse=rm_model,
        combined_holdout_rmse=rm_comb,
        market_holdout_rmse=rm_market,
        reason="later-holdout improvement" if active else "market blend did not improve later holdout",
    )


def _tilt_probabilities(margin: np.ndarray, target_mean: float) -> Tuple[np.ndarray, float]:
    m = np.asarray(margin, dtype=float)
    good = np.isfinite(m)
    if not np.all(good) or len(m) == 0 or float(np.std(m)) < 1e-9:
        return np.full(len(m), 1.0 / max(len(m), 1)), 0.0
    target = float(np.clip(target_mean, np.min(m) + 1e-8, np.max(m) - 1e-8))
    current = float(np.mean(m))
    if abs(target - current) < 1e-10:
        return np.full(len(m), 1.0 / len(m)), 0.0

    def weighted_mean(lam: float) -> tuple[float, np.ndarray]:
        z = lam * m
        z = z - np.max(z)
        w = np.exp(np.clip(z, -700, 0))
        w /= np.sum(w)
        return float(np.sum(w * m)), w

    lo, hi = -0.01, 0.01
    ml, _ = weighted_mean(lo); mh, _ = weighted_mean(hi)
    for _ in range(40):
        if ml <= target <= mh:
            break
        if target < ml:
            hi = lo; mh = ml; lo *= 2.0; ml, _ = weighted_mean(lo)
        else:
            lo = hi; ml = mh; hi *= 2.0; mh, _ = weighted_mean(hi)
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        mm, _ = weighted_mean(mid)
        if mm < target:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    _, p = weighted_mean(lam)
    return p, float(lam)


def apply_market_margin_prior(
    home_sim: pd.DataFrame,
    away_sim: pd.DataFrame,
    home_spread: float | None,
    calibration: MarketMarginCalibration | None,
    seed: int = 1818,
):
    """Return a market-augmented joint simulation plus an audit dict.

    If calibration is inactive or no spread is supplied, the original frames
    are returned unchanged.
    """
    audit = {
        "active": False,
        "calibration_weight": 0.0,
        "model_margin_before": np.nan,
        "market_margin": np.nan,
        "target_blended_margin": np.nan,
        "margin_after": np.nan,
        "total_before": np.nan,
        "total_after": np.nan,
        "tilt_lambda": 0.0,
        "reason": "inactive",
    }
    if home_sim is None or away_sim is None or len(home_sim) != len(away_sim) or len(home_sim) == 0:
        audit["reason"] = "invalid simulations"
        return home_sim, away_sim, audit

    model_margin = float(np.mean(home_sim["PTS"].to_numpy(dtype=float) - away_sim["PTS"].to_numpy(dtype=float)))
    total_before = float(np.mean(home_sim["PTS"].to_numpy(dtype=float) + away_sim["PTS"].to_numpy(dtype=float)))
    audit["model_margin_before"] = model_margin
    audit["total_before"] = total_before

    if home_spread is None or not np.isfinite(float(home_spread)):
        audit["reason"] = "no current market spread"
        return home_sim, away_sim, audit
    market_margin = -float(home_spread)
    audit["market_margin"] = market_margin

    if calibration is None or not calibration.active or calibration.weight <= 0:
        audit["reason"] = "no validated historical market calibration"
        return home_sim, away_sim, audit

    w = float(calibration.weight)
    target = model_margin + w * (market_margin - model_margin)
    margin = home_sim["PTS"].to_numpy(dtype=float) - away_sim["PTS"].to_numpy(dtype=float)
    probs, lam = _tilt_probabilities(margin, target)
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(len(home_sim)), size=len(home_sim), replace=True, p=probs)
    h = home_sim.iloc[idx].reset_index(drop=True)
    a = away_sim.iloc[idx].reset_index(drop=True)
    margin_after = float(np.mean(h["PTS"].to_numpy(dtype=float) - a["PTS"].to_numpy(dtype=float)))
    total_after = float(np.mean(h["PTS"].to_numpy(dtype=float) + a["PTS"].to_numpy(dtype=float)))
    audit.update({
        "active": True,
        "calibration_weight": w,
        "target_blended_margin": target,
        "margin_after": margin_after,
        "total_after": total_after,
        "tilt_lambda": lam,
        "reason": calibration.reason,
    })
    return h, a, audit
