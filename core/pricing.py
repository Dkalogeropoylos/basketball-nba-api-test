import numpy as np
import pandas as pd


def _fair_decimal(win_prob: float, push_prob: float = 0.0) -> float:
    """Push-aware fair decimal odds."""
    return (1.0 - push_prob) / win_prob if win_prob > 0 else np.inf


def required_odds_for_ev(
    win_prob: float,
    push_prob: float = 0.0,
    target_ev: float = 0.06,
) -> float:
    """
    Minimum decimal odds required for a target model EV.

        EV = p_win * odds + p_push - 1
        odds_required = (1 + target_ev - p_push) / p_win

    Example: fair 1.70 on a half-line + 6% target EV -> 1.80, not 1.71/1.73.
    """
    if win_prob <= 0:
        return np.inf
    return (1.0 + float(target_ev) - float(push_prob)) / float(win_prob)


def price(values, line, over_odds, under_odds):
    x = np.asarray(values)
    po = float(np.mean(x > line))
    pu = float(np.mean(x < line))
    pp = max(0.0, 1.0 - po - pu)
    return {
        "p_over": po,
        "p_under": pu,
        "p_push": pp,
        "fair_over": _fair_decimal(po, pp),
        "fair_under": _fair_decimal(pu, pp),
        "be_over": (1.0 - pp) / over_odds,
        "be_under": (1.0 - pp) / under_odds,
        "ev_over": po * over_odds + pp - 1.0,
        "ev_under": pu * under_odds + pp - 1.0,
    }


def _line_probs(values, line):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    po = float(np.mean(x > line))
    pu = float(np.mean(x < line))
    pp = max(0.0, 1.0 - po - pu)
    return po, pu, pp


def _candidate_half_lines(values, pad=2):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.asarray([], dtype=float)
    lo = int(np.floor(np.quantile(x, 0.02))) - int(pad)
    hi = int(np.ceil(np.quantile(x, 0.98))) + int(pad)
    return np.arange(lo, hi + 1, dtype=float) + 0.5


def model_line(values):
    """Convert a simulated distribution into a sportsbook-style HALF line."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "projection": np.nan,
            "median": np.nan,
            "line": np.nan,
            "p_over": np.nan,
            "p_under": np.nan,
            "fair_over": np.nan,
            "fair_under": np.nan,
        }

    mean = float(np.mean(x))
    median = float(np.median(x))
    candidates = _candidate_half_lines(x, pad=1)

    best = None
    for line in candidates:
        po, pu, _ = _line_probs(x, line)
        score = (abs(po - pu), abs(line - median), abs(line - mean))
        if best is None or score < best[0]:
            best = (score, float(line), po, pu)

    _, line, po, pu = best
    return {
        "projection": mean,
        "median": median,
        "line": line,
        "p_over": po,
        "p_under": pu,
        "fair_over": 1.0 / po if po > 0 else np.inf,
        "fair_under": 1.0 / pu if pu > 0 else np.inf,
    }


def _reference_line_thresholds(values, reference_odds=1.90, target_ev=0.06):
    """
    At a familiar reference price (default 1.90), report:
      - highest half-line where Over still reaches target EV
      - lowest half-line where Under still reaches target EV
    This lets the trader compare a book line without typing anything.
    """
    lines = _candidate_half_lines(values, pad=3)
    if lines.size == 0:
        return np.nan, np.nan

    over_ok = []
    under_ok = []
    for line in lines:
        po, pu, pp = _line_probs(values, line)
        if np.isfinite(po) and po * reference_odds + pp - 1.0 >= target_ev:
            over_ok.append(float(line))
        if np.isfinite(pu) and pu * reference_odds + pp - 1.0 >= target_ev:
            under_ok.append(float(line))

    max_over = max(over_ok) if over_ok else np.nan
    min_under = min(under_ok) if under_ok else np.nan
    return max_over, min_under


def line_ladder(values, center_line=None, radius=3, target_ev=0.06):
    """Compact sportsbook half-line ladder with fair odds and disciplined play-from prices."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return pd.DataFrame()

    if center_line is None:
        center_line = model_line(x)["line"]
    center_line = float(center_line)
    lines = [center_line + i for i in range(-int(radius), int(radius) + 1)]

    rows = []
    for line in lines:
        po, pu, pp = _line_probs(x, line)
        rows.append({
            "Line": float(line),
            "P Over": po,
            "Fair Over": _fair_decimal(po, pp),
            f"Play Over from ({target_ev:.0%} EV)": required_odds_for_ev(po, pp, target_ev),
            "P Under": pu,
            "Fair Under": _fair_decimal(pu, pp),
            f"Play Under from ({target_ev:.0%} EV)": required_odds_for_ev(pu, pp, target_ev),
        })
    return pd.DataFrame(rows)


def auto_market_table(
    sim,
    markets,
    stress_low=None,
    stress_high=None,
    target_ev=0.06,
    reference_odds=1.90,
):
    rows = []
    for market in markets:
        if market not in sim.columns:
            continue
        vals = sim[market].to_numpy()
        ml = model_line(vals)
        po, pu, pp = _line_probs(vals, ml["line"])
        max_o, min_u = _reference_line_thresholds(
            vals, reference_odds=reference_odds, target_ev=target_ev
        )
        row = {
            "Market": market,
            "Projection": ml["projection"],
            "Median": ml["median"],
            "Model line": ml["line"],
            "P Over": po,
            "Fair Over": _fair_decimal(po, pp),
            f"Play O from ({target_ev:.0%} EV)": required_odds_for_ev(po, pp, target_ev),
            "P Under": pu,
            "Fair Under": _fair_decimal(pu, pp),
            f"Play U from ({target_ev:.0%} EV)": required_odds_for_ev(pu, pp, target_ev),
            f"Max O line @ {reference_odds:.2f}": max_o,
            f"Min U line @ {reference_odds:.2f}": min_u,
        }
        # Stress remains an audit feature, never the main player-pricing table.
        if stress_low is not None and market in stress_low.columns:
            row["Low projection"] = float(np.mean(stress_low[market]))
        if stress_high is not None and market in stress_high.columns:
            row["High projection"] = float(np.mean(stress_high[market]))
        rows.append(row)
    return pd.DataFrame(rows)


def most_market(home_values, away_values):
    h = np.asarray(home_values)
    a = np.asarray(away_values)
    n = min(len(h), len(a))
    h, a = h[:n], a[:n]
    ph = float(np.mean(h > a))
    pa = float(np.mean(a > h))
    pt = float(np.mean(a == h))
    return {
        "p_home": ph,
        "p_tie": pt,
        "p_away": pa,
        "fair_home": 1.0 / ph if ph > 0 else np.inf,
        "fair_tie": 1.0 / pt if pt > 0 else np.inf,
        "fair_away": 1.0 / pa if pa > 0 else np.inf,
    }




def _team_stat_values(df: pd.DataFrame, market: str) -> pd.Series:
    """Return one integer/counting stat from team game logs."""
    m = str(market).upper()
    direct = {
        "3PM": "FG3M", "3PA": "FG3A", "FTM": "FTM", "FTA": "FTA",
        "REB": "REB", "OREB": "OREB", "DREB": "DREB", "AST": "AST",
        "STL": "STL", "BLK": "BLK", "TOV": "TOV", "PF": "PF",
    }
    if m in direct and direct[m] in df.columns:
        return pd.to_numeric(df[direct[m]], errors="coerce")
    if m == "2PA":
        return pd.to_numeric(df["FGA"], errors="coerce") - pd.to_numeric(df["FG3A"], errors="coerce")
    if m == "2PM":
        return pd.to_numeric(df["FGM"], errors="coerce") - pd.to_numeric(df["FG3M"], errors="coerce")
    raise KeyError(f"Unsupported team-with-most market: {market}")


def _league_difference_calibration(team_logs: pd.DataFrame, market: str):
    """Empirical same-game difference spread for a team-counting stat."""
    if team_logs is None or team_logs.empty or "GAME_ID" not in team_logs.columns:
        return None
    rows = []
    for gid, g in team_logs.groupby("GAME_ID", sort=False):
        if len(g) < 2:
            continue
        g = g.iloc[:2].copy()
        try:
            vals = _team_stat_values(g, market).to_numpy(dtype=float)
        except Exception:
            continue
        if len(vals) != 2 or not np.isfinite(vals).all():
            continue
        rows.append((float(vals[0]), float(vals[1])))
    if len(rows) < 30:
        return None
    arr = np.asarray(rows, dtype=float)
    diffs = arr[:, 0] - arr[:, 1]
    totals = arr[:, 0] + arr[:, 1]
    return {
        "games": int(len(arr)),
        "diff_sd": float(np.std(diffs, ddof=1)),
        "mean_total": float(np.mean(totals)),
        "tie_rate": float(np.mean(diffs == 0)),
    }


def _linear_discrete_side_probs(x: np.ndarray):
    """
    Convert a continuous adjusted difference into integer outcome probabilities
    without Monte-Carlo rounding noise. Each value is linearly split between
    floor(x) and ceil(x), preserving its expectation.
    """
    x = np.asarray(x, dtype=float)
    lo = np.floor(x).astype(int)
    frac = x - lo
    hi = lo + 1
    w_lo = 1.0 - frac
    w_hi = frac
    ph = float(np.mean(w_lo * (lo > 0) + w_hi * (hi > 0)))
    pa = float(np.mean(w_lo * (lo < 0) + w_hi * (hi < 0)))
    pt = float(np.mean(w_lo * (lo == 0) + w_hi * (hi == 0)))
    total = ph + pa + pt
    if total > 0:
        ph, pa, pt = ph / total, pa / total, pt / total
    return ph, pt, pa


def most_market_calibrated(
    home_values,
    away_values,
    team_logs: pd.DataFrame | None = None,
    market: str | None = None,
    calibration_strength: float = 0.60,
):
    """
    Price a 3-way 'team with most' market from the coupled simulations, while
    calibrating ONLY the spread of the team difference to actual historical
    same-game dispersion from the league database. No bookmaker prices are used.

    Why: team means can be good while independent residual noise makes the
    difference too wide and tie probability too small. We preserve the simulated
    expected difference, but shrink/expand its residual SD toward the empirical
    league SD at the current scoring/stat scale.
    """
    h = np.asarray(home_values, dtype=float)
    a = np.asarray(away_values, dtype=float)
    n = min(len(h), len(a))
    h, a = h[:n], a[:n]
    d = h - a

    raw = most_market(h, a)
    if n < 100 or team_logs is None or market is None:
        return {**raw, "calibrated": False, "raw_p_tie": raw["p_tie"]}

    hist = _league_difference_calibration(team_logs, market)
    if not hist or hist["mean_total"] <= 0:
        return {**raw, "calibrated": False, "raw_p_tie": raw["p_tie"]}

    mu = float(np.mean(d))
    raw_sd = float(np.std(d, ddof=1))
    current_total = float(np.mean(h + a))
    target_sd = float(hist["diff_sd"] * np.sqrt(max(current_total, 1e-9) / hist["mean_total"]))
    strength = float(np.clip(calibration_strength, 0.0, 1.0))
    applied_sd = (1.0 - strength) * raw_sd + strength * target_sd

    if not np.isfinite(raw_sd) or raw_sd <= 1e-9 or not np.isfinite(applied_sd):
        return {**raw, "calibrated": False, "raw_p_tie": raw["p_tie"]}

    # Prevent any one historical calibration from radically rewriting the model.
    scale = float(np.clip(applied_sd / raw_sd, 0.80, 1.20))
    d_adj = mu + (d - mu) * scale
    ph, pt, pa = _linear_discrete_side_probs(d_adj)
    return {
        "p_home": ph,
        "p_tie": pt,
        "p_away": pa,
        "fair_home": 1.0 / ph if ph > 0 else np.inf,
        "fair_tie": 1.0 / pt if pt > 0 else np.inf,
        "fair_away": 1.0 / pa if pa > 0 else np.inf,
        "calibrated": True,
        "raw_p_home": raw["p_home"],
        "raw_p_tie": raw["p_tie"],
        "raw_p_away": raw["p_away"],
        "raw_diff_sd": raw_sd,
        "league_target_diff_sd": target_sd,
        "applied_diff_sd": raw_sd * scale,
        "league_tie_rate": hist["tie_rate"],
        "calibration_games": hist["games"],
    }


def market_table(sim, stress_low, stress_high, markets):
    """Legacy compact projection table."""
    rows = []
    for market in markets:
        if market not in sim.columns:
            continue
        x = sim[market].to_numpy()
        rows.append({
            "Market": market,
            "Mean": float(np.mean(x)),
            "Median": float(np.median(x)),
            "Low mean": float(np.mean(stress_low[market])) if market in stress_low.columns else np.nan,
            "High mean": float(np.mean(stress_high[market])) if market in stress_high.columns else np.nan,
        })
    return pd.DataFrame(rows)
