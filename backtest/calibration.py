from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from core.buckets import WeightConfig
from core.pace_engine import fit_pace_control
from core.matchup import fit_opponent_elasticities
from core.structural_calibration import (
    FEATURES,
    StructuralModel,
    fit_structural_rate_models,
    _single_game_features,
    _mean_feature,
    _cfg_baseline,
    _transform,
    _inverse,
)
from core.shooting_efficiency import fit_shooting_efficiency_models


@dataclass
class LeagueCalibration:
    train_season: int
    pace: dict
    opponent_elasticities: dict
    structural_models: Dict[str, StructuralModel]
    shooting_models: dict
    opponent_audit: pd.DataFrame
    structural_audit: pd.DataFrame
    shooting_audit: pd.DataFrame


def fit_league_calibration(team_logs: pd.DataFrame, train_season: int) -> LeagueCalibration:
    """Fit only league-level parameters on the designated development season."""
    pace = fit_pace_control(team_logs)
    elasticities, opp_audit = fit_opponent_elasticities(team_logs)
    structural_models, structural_audit = fit_structural_rate_models(team_logs)
    shooting_models, shooting_audit = fit_shooting_efficiency_models(team_logs)
    return LeagueCalibration(
        train_season=int(train_season),
        pace=pace,
        opponent_elasticities=elasticities,
        structural_models=structural_models,
        shooting_models=shooting_models,
        opponent_audit=opp_audit,
        structural_audit=structural_audit,
        shooting_audit=shooting_audit,
    )


def _pair_residuals_frozen(
    team_logs: pd.DataFrame,
    team_abbr: str,
    opponent_abbr: str,
    feature: str,
    beta: float,
    cfg: WeightConfig,
    feature_frame: pd.DataFrame | None = None,
) -> list[float]:
    """Build same-season pregame H2H residuals using a frozen opponent beta.

    This deliberately does NOT refit beta/K on the test season. For each earlier
    H2H, it reconstructs the own and opponent states from games completed before
    that H2H, then stores the realized residual against the frozen no-H2H model.
    """
    g = (feature_frame.copy() if feature_frame is not None else _single_game_features(team_logs))
    g = g.dropna(subset=["GAME_DATE", feature]).copy()
    if g.empty:
        return []
    g = g.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBR"]).reset_index(drop=True)
    team = str(team_abbr).upper()
    opp = str(opponent_abbr).upper()
    h = g[g["TEAM_ABBR"].eq(team) & g["OPP_ABBR"].eq(opp)].copy()
    if h.empty:
        return []

    residuals: list[float] = []
    # Do not use namedtuple attribute access here. Structural feature names such
    # as ``3P_SHARE`` are valid DataFrame column labels but are NOT valid Python
    # identifiers, so pandas.itertuples() silently renames them (e.g. to _N).
    # Label-based Series access keeps the frozen H2H residual path valid for all
    # FEATURES and avoids selectively dropping rematches from walk-forward tests.
    for _, r in h.iterrows():
        row_date = r["GAME_DATE"]
        row_game_id = r["GAME_ID"]
        prior = g[
            (g["GAME_DATE"] < row_date)
            | ((g["GAME_DATE"] == row_date) & (g["GAME_ID"].astype(str) < str(row_game_id)))
        ].copy()
        if len(prior) < 80:
            continue
        league = _mean_feature(prior, feature, np.nan)
        if not np.isfinite(league):
            continue
        own_hist = prior[prior["TEAM_ABBR"].eq(team) & ~prior["OPP_ABBR"].eq(opp)].copy()
        opp_hist = prior[prior["OPP_ABBR"].eq(opp) & ~prior["TEAM_ABBR"].eq(team)].copy()
        if len(own_hist) < 8 or len(opp_hist) < 8:
            continue
        own = _cfg_baseline(own_hist, feature, league, cfg)
        opp_allowed = _cfg_baseline(opp_hist, feature, league, WeightConfig.stable())
        actual = float(r[feature])
        if not all(np.isfinite(v) for v in (own, opp_allowed, actual)):
            continue
        lg_t = float(_transform(feature, [league])[0])
        own_t = float(_transform(feature, [own])[0])
        opp_t = float(_transform(feature, [opp_allowed])[0])
        actual_t = float(_transform(feature, [actual])[0])
        no_h2h_t = own_t + float(beta) * (opp_t - lg_t)
        residuals.append(actual_t - no_h2h_t)
    return residuals


def predict_structural_modifiers_frozen(
    team_logs: pd.DataFrame,
    team_abbr: str,
    opponent_abbr: str,
    frozen_models: Dict[str, StructuralModel],
    cfg: WeightConfig,
    h2h_rotation_similarity: float = 1.0,
):
    """Current-season prediction with beta/K frozen from the train season."""
    g = _single_game_features(team_logs).dropna(subset=["GAME_DATE"]).copy()
    team = str(team_abbr).upper()
    opp = str(opponent_abbr).upper()
    own_hist = g[g["TEAM_ABBR"].eq(team) & ~g["OPP_ABBR"].eq(opp)].copy()
    opp_hist = g[g["OPP_ABBR"].eq(opp) & ~g["TEAM_ABBR"].eq(team)].copy()
    raw_h2h = g[g["TEAM_ABBR"].eq(team) & g["OPP_ABBR"].eq(opp)].copy()

    mods = {"3P_SHARE": 1.0, "FTA": 1.0, "TOV": 1.0, "OREB": 1.0, "DREB": 1.0, "AST": 1.0}
    mapping = {
        "3P_SHARE": "3P_SHARE",
        "FTA": "FTA",
        "TOV": "TOV",
        "OREB_PER_MISS": "OREB",
        "DREB_CAPTURE": "DREB",
        "AST_PER_MAKE": "AST",
    }
    audit = []

    for feature in FEATURES:
        model = frozen_models.get(feature)
        league = _mean_feature(g, feature, np.nan)
        baseline = _cfg_baseline(own_hist, feature, league, cfg) if np.isfinite(league) else np.nan
        opp_state = _cfg_baseline(opp_hist, feature, league, WeightConfig.stable()) if np.isfinite(league) else np.nan
        pred_no_h2h = baseline
        pred = baseline
        h2h_weight = 0.0
        h2h_raw_resid = np.nan
        residuals: list[float] = []

        if (
            model is not None and model.active and np.isfinite(league)
            and np.isfinite(baseline) and baseline > 0
            and np.isfinite(opp_state) and opp_state > 0
            and len(own_hist) >= 8 and len(opp_hist) >= 8
        ):
            lg_t = float(_transform(feature, [league])[0])
            own_t = float(_transform(feature, [baseline])[0])
            opp_t = float(_transform(feature, [opp_state])[0])
            no_h2h_t = own_t + float(model.opponent_beta) * (opp_t - lg_t)
            pred_no_h2h = float(_inverse(feature, [no_h2h_t])[0])

            residuals = _pair_residuals_frozen(
                team_logs, team, opp, feature, float(model.opponent_beta), cfg, feature_frame=g
            )
            if residuals and np.isfinite(model.h2h_prior_k):
                h2h_raw_resid = float(np.mean(residuals))
                h2h_weight = float(len(residuals) / (len(residuals) + max(float(model.h2h_prior_k), 1e-9)))
                h2h_weight *= float(np.clip(h2h_rotation_similarity, 0.0, 1.0))
                no_h2h_t += h2h_weight * h2h_raw_resid
            pred = float(_inverse(feature, [no_h2h_t])[0])

        if feature == "3P_SHARE":
            pred = float(np.clip(pred, 0.06, 0.75))
        elif feature == "TOV":
            pred = float(np.clip(pred, 0.03, 0.30))
        elif feature == "FTA":
            pred = float(np.clip(pred, 0.05, 0.55))
        elif feature == "OREB_PER_MISS":
            pred = float(np.clip(pred, 0.05, 0.55))
        elif feature == "DREB_CAPTURE":
            pred = float(np.clip(pred, 0.70, 0.995))
        elif feature == "AST_PER_MAKE":
            pred = float(np.clip(pred, 0.20, 0.95))

        mod = float(pred / baseline) if np.isfinite(pred) and np.isfinite(baseline) and baseline > 0 else 1.0
        mods[mapping[feature]] = mod
        audit.append({
            "Feature": feature,
            "Frozen active": bool(model.active) if model else False,
            "Frozen opponent beta": float(model.opponent_beta) if model else 0.0,
            "Frozen H2H K": float(model.h2h_prior_k) if model else np.inf,
            "Own current state": baseline,
            "Opponent current allowed state": opp_state,
            "Current same-season H2H": int(len(raw_h2h)),
            "Usable pregame H2H residuals": int(len(residuals)),
            "H2H rotation similarity": float(h2h_rotation_similarity),
            "Applied H2H weight": h2h_weight,
            "Final prediction": pred,
            "Applied modifier": mod,
        })
    return mods, pd.DataFrame(audit)
