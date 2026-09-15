from __future__ import annotations

import numpy as np
import pandas as pd


def empirical_crps(samples, actual: float) -> float:
    x = np.sort(np.asarray(samples, dtype=float))
    x = x[np.isfinite(x)]
    if len(x) == 0 or not np.isfinite(actual):
        return np.nan
    n = len(x)
    first = float(np.mean(np.abs(x - float(actual))))
    coeff = 2.0 * np.arange(1, n + 1) - n - 1.0
    half_pair = float(np.sum(coeff * x) / (n * n))
    return first - half_pair


def sample_summary(samples, actual: float) -> dict:
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    q05, q10, q25, q50, q75, q90, q95 = np.quantile(x, [0.05,0.10,0.25,0.50,0.75,0.90,0.95])
    return {
        "Projection": float(np.mean(x)),
        "Median": float(q50),
        "SD": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "Q05": float(q05), "Q10": float(q10), "Q25": float(q25),
        "Q75": float(q75), "Q90": float(q90), "Q95": float(q95),
        "Cover50": float(q25 <= actual <= q75),
        "Cover80": float(q10 <= actual <= q90),
        "Cover90": float(q05 <= actual <= q95),
        "CRPS": empirical_crps(x, actual),
    }


def pseudo_line_rows(samples, actual: float, market: str, scope: str, game_id: str, team: str | None = None):
    """Pregame-defined half-point test lines around the simulated center.

    Lines are generated from the simulated median only, never from the actual.
    This makes probability calibration testable without historical bookmaker lines.
    """
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return []
    med = float(np.median(x))
    scale = max(float(np.std(x, ddof=1)), 1.0)
    offsets = [-1.25, -0.75, -0.35, 0.35, 0.75, 1.25]
    lines = []
    seen = set()
    for z in offsets:
        raw = med + z * scale
        line = float(np.floor(raw) + 0.5)
        if line in seen:
            continue
        seen.add(line)
        p_over = float(np.mean(x > line))
        p_under = float(np.mean(x < line))
        push = float(np.mean(x == line))
        if p_over + p_under > 0:
            # Conditional on no push, mirroring a two-way half-point market.
            den = p_over + p_under
            p_over /= den
            p_under /= den
        lines.append({
            "GAME_ID": str(game_id), "Scope": scope, "Team": team or "TOTAL",
            "Market": market, "Line": line,
            "P_Over": p_over, "P_Under": p_under, "P_Push": push,
            "Over_Win": float(actual > line), "Under_Win": float(actual < line),
        })
    return lines


def metric_table(detail: pd.DataFrame) -> pd.DataFrame:
    if detail is None or detail.empty:
        return pd.DataFrame()
    d = detail.copy()
    d["Error"] = d["Projection"] - d["Actual"]
    rows = []
    for (scope, market), g in d.groupby(["Scope", "Market"], dropna=False):
        e = g["Error"].to_numpy(float)
        rows.append({
            "Scope": scope,
            "Market": market,
            "N": int(len(g)),
            "Bias": float(np.mean(e)),
            "MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e * e))),
            "50% coverage": float(g["Cover50"].mean()),
            "80% coverage": float(g["Cover80"].mean()),
            "90% coverage": float(g["Cover90"].mean()),
            "Mean CRPS": float(g["CRPS"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["Scope", "Market"]).reset_index(drop=True)


def calibration_bins(prob_rows: pd.DataFrame, side: str = "Over") -> pd.DataFrame:
    if prob_rows is None or prob_rows.empty:
        return pd.DataFrame()
    pcol = f"P_{side}"
    ycol = f"{side}_Win"
    d = prob_rows[["GAME_ID","Scope","Market",pcol,ycol]].dropna().copy()
    bins = np.arange(0.0, 1.0001, 0.05)
    d["Bin"] = pd.cut(d[pcol], bins=bins, include_lowest=True, right=False)
    rows = []
    for (scope, market, b), g in d.groupby(["Scope","Market","Bin"], observed=True):
        if len(g) < 5:
            continue
        p = g[pcol].to_numpy(float)
        y = g[ycol].to_numpy(float)
        rows.append({
            "Scope": scope, "Market": market, "Probability bin": str(b),
            "N": int(len(g)), "Mean predicted": float(np.mean(p)),
            "Actual hit rate": float(np.mean(y)),
            "Calibration gap": float(np.mean(y) - np.mean(p)),
            "Brier": float(np.mean((p-y)**2)),
        })
    return pd.DataFrame(rows)


def most_summary(most_rows: pd.DataFrame) -> pd.DataFrame:
    if most_rows is None or most_rows.empty:
        return pd.DataFrame()
    d = most_rows.copy()
    rows = []
    for market, g in d.groupby("Market"):
        p = g[["P_Home","P_Away","P_Tie"]].to_numpy(float)
        y = g[["Y_Home","Y_Away","Y_Tie"]].to_numpy(float)
        brier = float(np.mean(np.sum((p-y)**2, axis=1)))
        rows.append({
            "Market": market, "N": int(len(g)), "Multiclass Brier": brier,
            "Pred home": float(g["P_Home"].mean()), "Actual home": float(g["Y_Home"].mean()),
            "Pred away": float(g["P_Away"].mean()), "Actual away": float(g["Y_Away"].mean()),
            "Pred tie": float(g["P_Tie"].mean()), "Actual tie": float(g["Y_Tie"].mean()),
        })
    return pd.DataFrame(rows)
