from __future__ import annotations

"""Walk-forward team shooting-efficiency calibration.

v2.17 fixed the possession / shot-allocation side of Team Markets.  v2.17.2
keeps that structure intact and replaces the remaining strongly hand-shrunk
3P% / 2P% matchup layer with an attempt-weighted binomial model.

For each historical team-game, using only information available before tip:

    actual makes_t ~ Binomial(attempts_t, p_t)
    logit(p_t) = logit(own_offense_posterior)
                 + beta * [logit(opponent_allowed_posterior)
                           - logit(league_prior)]

The offense and defense posterior strengths are selected from a data-scaled
attempt grid.  beta is selected from a broad non-negative grid.  Parameters are
chosen on the early chronological sample and the layer activates only if it
beats an own-offense-only baseline on held-out later games by binomial log loss.

This is intentionally an efficiency-only layer.  It does NOT modify FGA, 3P
share, 3PA, 2PA, pace, TOV, FTA, AST, or rebounds.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ShootingEfficiencyModel:
    feature: str
    active: bool
    beta: float
    own_prior_attempts: float
    defense_prior_attempts: float
    rows: int
    train_nll: float
    test_nll: float
    baseline_test_nll: float
    relative_test_gain: float


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _inv_logit(z):
    z = np.clip(np.asarray(z, dtype=float), -14.0, 14.0)
    return 1.0 / (1.0 + np.exp(-z))


def _feature_counts(df: pd.DataFrame, feature: str) -> Tuple[np.ndarray, np.ndarray]:
    if feature == "3P_PCT":
        m = pd.to_numeric(df.get("FG3M", 0), errors="coerce").fillna(0.0).to_numpy(float)
        a = pd.to_numeric(df.get("FG3A", 0), errors="coerce").fillna(0.0).to_numpy(float)
        return np.maximum(m, 0.0), np.maximum(a, 0.0)
    if feature == "2P_PCT":
        fgm = pd.to_numeric(df.get("FGM", 0), errors="coerce").fillna(0.0).to_numpy(float)
        fga = pd.to_numeric(df.get("FGA", 0), errors="coerce").fillna(0.0).to_numpy(float)
        m3 = pd.to_numeric(df.get("FG3M", 0), errors="coerce").fillna(0.0).to_numpy(float)
        a3 = pd.to_numeric(df.get("FG3A", 0), errors="coerce").fillna(0.0).to_numpy(float)
        return np.maximum(fgm - m3, 0.0), np.maximum(fga - a3, 0.0)
    raise KeyError(feature)


def _pregame_table(team_logs: pd.DataFrame, feature: str) -> pd.DataFrame:
    if team_logs is None or team_logs.empty:
        return pd.DataFrame()
    x = team_logs.copy()
    x["GAME_DATE"] = pd.to_datetime(x["GAME_DATE"], errors="coerce")
    if "GAME_ID" not in x.columns:
        x["GAME_ID"] = np.arange(len(x)).astype(str)
    x = x.dropna(subset=["GAME_DATE"]).copy()
    x["TEAM_ABBR"] = x["TEAM_ABBR"].astype(str).str.upper()
    x["OPP_ABBR"] = x["OPP_ABBR"].astype(str).str.upper()
    m, a = _feature_counts(x, feature)
    x["M_CAL"] = m
    x["A_CAL"] = a
    x = x.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBR"]).reset_index(drop=True)

    own = {}      # team -> [m,a]
    defense = {}  # opponent -> outcomes allowed [m,a]
    lg_m = 0.0
    lg_a = 0.0
    rows = []

    games = (
        x[["GAME_DATE", "GAME_ID"]]
        .drop_duplicates()
        .sort_values(["GAME_DATE", "GAME_ID"])
        .itertuples(index=False, name=None)
    )
    for gd, gid in games:
        cur = x[(x["GAME_DATE"] == gd) & (x["GAME_ID"] == gid)]
        if lg_a > 0:
            for _, r in cur.iterrows():
                team = str(r["TEAM_ABBR"])
                opp = str(r["OPP_ABBR"])
                om, oa = own.get(team, (0.0, 0.0))
                dm, da = defense.get(opp, (0.0, 0.0))
                aa = float(r["A_CAL"])
                mm = float(r["M_CAL"])
                if oa > 0 and da > 0 and aa > 0:
                    rows.append({
                        "GAME_DATE": gd, "GAME_ID": str(gid),
                        "TEAM_ABBR": team, "OPP_ABBR": opp,
                        "OWN_M": om, "OWN_A": oa,
                        "DEF_M": dm, "DEF_A": da,
                        "LG_M": lg_m, "LG_A": lg_a,
                        "M": mm, "A": aa,
                    })

        # Update only after both rows of the game have been converted to priors.
        for _, r in cur.iterrows():
            team = str(r["TEAM_ABBR"])
            opp = str(r["OPP_ABBR"])
            mm = float(r["M_CAL"]); aa = float(r["A_CAL"])
            om, oa = own.get(team, (0.0, 0.0))
            dm, da = defense.get(opp, (0.0, 0.0))
            own[team] = (om + mm, oa + aa)
            defense[opp] = (dm + mm, da + aa)
            lg_m += mm; lg_a += aa

    return pd.DataFrame(rows)


def _predict_prob(df: pd.DataFrame, k_own: float, k_def: float, beta: float) -> np.ndarray:
    lg = np.clip((df["LG_M"].to_numpy(float) + 0.5) / (df["LG_A"].to_numpy(float) + 1.0), 1e-5, 1-1e-5)
    own = (
        df["OWN_M"].to_numpy(float) + float(k_own) * lg
    ) / np.maximum(df["OWN_A"].to_numpy(float) + float(k_own), 1e-9)
    deff = (
        df["DEF_M"].to_numpy(float) + float(k_def) * lg
    ) / np.maximum(df["DEF_A"].to_numpy(float) + float(k_def), 1e-9)
    z = _logit(own) + float(beta) * (_logit(deff) - _logit(lg))
    return np.asarray(_inv_logit(z), float)


def _binomial_nll(df: pd.DataFrame, p: np.ndarray) -> float:
    m = df["M"].to_numpy(float)
    a = df["A"].to_numpy(float)
    p = np.clip(np.asarray(p, float), 1e-9, 1-1e-9)
    # Combination term is constant across candidate parameters, so omit it.
    return float(-np.sum(m*np.log(p) + (a-m)*np.log(1-p)))


def _candidate_grid(df: pd.DataFrame):
    med = max(float(np.median(df["A"].to_numpy(float))), 1.0)
    # Attempt-scaled candidate masses; the selected value is data-driven.
    k_grid = np.unique(np.concatenate([[0.0], med * np.logspace(0.0, 5.0/3.0, 7)])).tolist()
    beta_grid = np.linspace(0.0, 1.25, 26).tolist()
    return k_grid, beta_grid


def _fit_one(team_logs: pd.DataFrame, feature: str) -> ShootingEfficiencyModel:
    df = _pregame_table(team_logs, feature)
    n = len(df)
    if n < 120:
        return ShootingEfficiencyModel(feature, False, 0.0, 0.0, 0.0, n, np.nan, np.nan, np.nan, np.nan)

    df = df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBR"]).reset_index(drop=True)
    cut = max(int(n * 0.70), 80)
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    if len(test) < 30:
        return ShootingEfficiencyModel(feature, False, 0.0, 0.0, 0.0, n, np.nan, np.nan, np.nan, np.nan)

    k_grid, beta_grid = _candidate_grid(train)

    # Baseline chooses its own offense shrinkage but has beta=0 (no opponent).
    base_best = (np.inf, 0.0)
    for ko in k_grid:
        nll = _binomial_nll(train, _predict_prob(train, ko, 0.0, 0.0))
        if nll < base_best[0]:
            base_best = (nll, float(ko))
    base_ko = base_best[1]
    base_test = _binomial_nll(test, _predict_prob(test, base_ko, 0.0, 0.0))

    best = (np.inf, 0.0, 0.0, 0.0)
    for ko in k_grid:
        for kd in k_grid:
            for beta in beta_grid:
                p = _predict_prob(train, ko, kd, beta)
                nll = _binomial_nll(train, p)
                if nll < best[0]:
                    best = (nll, float(ko), float(kd), float(beta))

    train_nll, ko, kd, beta = best
    test_nll = _binomial_nll(test, _predict_prob(test, ko, kd, beta))
    active = bool(test_nll < base_test and beta > 0)
    if not active:
        return ShootingEfficiencyModel(
            feature, False, 0.0, base_ko, 0.0, n,
            float(base_best[0]), float(base_test), float(base_test), 0.0,
        )
    gain = float((base_test - test_nll) / max(base_test, 1e-9))
    return ShootingEfficiencyModel(
        feature, True, beta, ko, kd, n,
        float(train_nll), float(test_nll), float(base_test), gain,
    )


def fit_shooting_efficiency_models(team_logs: pd.DataFrame):
    models: Dict[str, ShootingEfficiencyModel] = {}
    rows = []
    for feature in ("3P_PCT", "2P_PCT"):
        model = _fit_one(team_logs, feature)
        models[feature] = model
        rows.append({
            "Feature": feature,
            "Active": model.active,
            "Rows": model.rows,
            "Learned defense beta": model.beta,
            "Own prior attempts": model.own_prior_attempts,
            "Defense prior attempts": model.defense_prior_attempts,
            "Train binomial NLL": model.train_nll,
            "Held-out binomial NLL": model.test_nll,
            "Own-only held-out NLL": model.baseline_test_nll,
            "Relative held-out gain": model.relative_test_gain,
        })
    return models, pd.DataFrame(rows)


def _current_counts(team_logs: pd.DataFrame, feature: str, team_abbr: str, opponent_abbr: str):
    x = team_logs.copy()
    x["TEAM_ABBR"] = x["TEAM_ABBR"].astype(str).str.upper()
    x["OPP_ABBR"] = x["OPP_ABBR"].astype(str).str.upper()
    m, a = _feature_counts(x, feature)
    x["M_CAL"] = m; x["A_CAL"] = a
    team = str(team_abbr).upper(); opp = str(opponent_abbr).upper()
    own = x[x["TEAM_ABBR"].eq(team)]
    # No explicit H2H shooting-percentage layer exists, so H2H rows are allowed
    # here exactly once as part of the opponent's defensive shooting history.
    deff = x[x["OPP_ABBR"].eq(opp)]
    return (
        float(own["M_CAL"].sum()), float(own["A_CAL"].sum()),
        float(deff["M_CAL"].sum()), float(deff["A_CAL"].sum()),
        float(x["M_CAL"].sum()), float(x["A_CAL"].sum()),
    )


def predict_shooting_efficiency_modifiers(
    team_logs: pd.DataFrame,
    team_abbr: str,
    opponent_abbr: str,
    own_profile: dict,
    models: Dict[str, ShootingEfficiencyModel],
):
    out = {"3P_PCT": 1.0, "2P_PCT": 1.0}
    rows = []
    own_key = {"3P_PCT": "three_pct", "2P_PCT": "two_pct"}
    for feature in ("3P_PCT", "2P_PCT"):
        model = models.get(feature)
        base = float(own_profile.get(own_key[feature], np.nan))
        if model is None or not model.active or not (np.isfinite(base) and 0 < base < 1):
            rows.append({"Feature": feature, "Active": False, "Applied modifier": 1.0})
            continue
        om, oa, dm, da, lm, la = _current_counts(team_logs, feature, team_abbr, opponent_abbr)
        if min(oa, da, la) <= 0:
            rows.append({"Feature": feature, "Active": False, "Applied modifier": 1.0})
            continue
        lg = float(np.clip((lm + 0.5)/(la + 1.0), 1e-5, 1-1e-5))
        # Live own shooting skill remains the existing large-sample team profile.
        # The binomial calibrator is used only to learn how much opponent defense
        # should move that skill.  This avoids replacing a stable offense prior
        # with a second independently-shrunk offense estimate.
        own_post_calibration = float((om + model.own_prior_attempts*lg)/(oa + model.own_prior_attempts))
        def_post = float((dm + model.defense_prior_attempts*lg)/(da + model.defense_prior_attempts))
        target = float(_inv_logit(_logit([base])[0] + model.beta*(_logit([def_post])[0] - _logit([lg])[0])))
        mod = float(target/base)
        out[feature] = mod
        rows.append({
            "Feature": feature, "Active": True,
            "Historical profile pct": base,
            "Calibration own posterior pct": own_post_calibration,
            "Opponent allowed posterior pct": def_post,
            "League pct": lg,
            "Learned defense beta": model.beta,
            "Own prior attempts (fit only)": model.own_prior_attempts,
            "Defense prior attempts": model.defense_prior_attempts,
            "Target pct": target,
            "Applied modifier": mod,
        })
    return out, pd.DataFrame(rows)
