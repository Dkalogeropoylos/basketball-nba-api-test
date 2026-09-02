from __future__ import annotations

"""Walk-forward, opponent-adjusted Team-Market structural rates (v2.18).

The v2.17 ridge model used eight correlated predictors (Own and Opponent
Old/G6-10/L5 plus two H2H terms).  v2.18 replaces that specification with a
more transparent decomposition:

    transformed prediction
      = current own non-H2H state
      + beta * opponent-allowed deviation from league
      + shrunk repeat-matchup residual

where:
  * Old / G6-10 / L5 remain non-overlapping;
  * beta >= 0 is selected from chronological validation, and beta=0 is allowed;
  * H2H is NOT a raw second opponent boost.  It is the residual left after the
    own + opponent model at the time of each previous H2H;
  * the H2H prior mass K is selected chronologically, with K=inf explicitly
    disabling H2H;
  * the final model must beat the own-state baseline on a genuinely later
    holdout and must not degrade the extreme-opponent subset.

This is deliberately a small model.  It avoids coefficient sign instability
from highly collinear recency buckets and makes the opponent and H2H
contributions auditable in basketball units.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from core.buckets import WeightConfig, split_non_overlapping, active_weights


FEATURES = ("3P_SHARE", "FTA", "TOV", "OREB_PER_MISS", "DREB_CAPTURE", "AST_PER_MAKE")


@dataclass
class StructuralModel:
    feature: str
    active: bool
    opponent_beta: float
    h2h_prior_k: float
    rows: int
    tune_rmse: float
    baseline_tune_rmse: float
    holdout_rmse: float
    baseline_holdout_rmse: float
    extreme_holdout_rmse: float
    baseline_extreme_holdout_rmse: float
    holdout_rows: int
    extreme_holdout_rows: int
    training_table: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    # Backward-compatible audit attributes used by older UI/tests.
    @property
    def ridge_lambda(self) -> float:
        return np.nan

    @property
    def cv_rmse(self) -> float:
        return self.holdout_rmse

    @property
    def baseline_cv_rmse(self) -> float:
        return self.baseline_holdout_rmse

    @property
    def coefficients(self) -> Dict[str, float]:
        return {
            "Opponent beta": float(self.opponent_beta),
            "H2H prior K": float(self.h2h_prior_k),
        }


def _estimate_possessions(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(df["FGA"], errors="coerce").fillna(0.0)
        - pd.to_numeric(df["OREB"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["TOV"], errors="coerce").fillna(0.0)
        + 0.44 * pd.to_numeric(df["FTA"], errors="coerce").fillna(0.0)
    )


def _single_game_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in ["FGA", "FGM", "FG3A", "FTA", "TOV", "AST", "OREB", "DREB"]:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    poss = _estimate_possessions(x).clip(lower=1e-6)
    misses = (x["FGA"] - x["FGM"]).clip(lower=1e-6)
    out = pd.DataFrame({
        "GAME_DATE": pd.to_datetime(x["GAME_DATE"], errors="coerce"),
        "GAME_ID": x["GAME_ID"].astype(str),
        "TEAM_ABBR": x["TEAM_ABBR"].astype(str).str.upper(),
        "OPP_ABBR": x["OPP_ABBR"].astype(str).str.upper(),
        "3P_SHARE": x["FG3A"] / x["FGA"].replace(0, np.nan),
        "FTA": x["FTA"] / poss,
        "TOV": x["TOV"] / poss,
        "OREB_PER_MISS": x["OREB"] / misses,
        # One official team assist can be credited to at most one made FG.
        "AST_PER_MAKE": x["AST"] / x["FGM"].replace(0, np.nan),
    })

    # Defensive rebounding is conditional on the OPPONENT'S missed shots that
    # were not recovered offensively. Build the game-paired opportunity rate
    # here so it can receive the same opponent + residualized-H2H treatment as
    # the offensive structural rates.
    own = x[["GAME_ID", "TEAM_ABBR", "OPP_ABBR", "DREB"]].copy()
    own["GAME_ID"] = own["GAME_ID"].astype(str)
    own["TEAM_ABBR"] = own["TEAM_ABBR"].astype(str).str.upper()
    own["OPP_ABBR"] = own["OPP_ABBR"].astype(str).str.upper()
    opp = x[["GAME_ID", "TEAM_ABBR", "FGA", "FGM", "OREB"]].copy()
    opp["GAME_ID"] = opp["GAME_ID"].astype(str)
    opp["TEAM_ABBR"] = opp["TEAM_ABBR"].astype(str).str.upper()
    opp = opp.rename(columns={
        "TEAM_ABBR": "OPP_TEAM_ABBR", "FGA": "OPP_FGA",
        "FGM": "OPP_FGM", "OREB": "OPP_OREB",
    })
    paired = own.merge(opp, on="GAME_ID", how="left")
    paired = paired[paired["OPP_ABBR"].eq(paired["OPP_TEAM_ABBR"])].copy()
    if not paired.empty:
        chances = (
            pd.to_numeric(paired["OPP_FGA"], errors="coerce")
            - pd.to_numeric(paired["OPP_FGM"], errors="coerce")
            - pd.to_numeric(paired["OPP_OREB"], errors="coerce")
        )
        paired["DREB_CAPTURE"] = pd.to_numeric(paired["DREB"], errors="coerce") / chances.where(chances > 0)
        cap = paired[["GAME_ID", "TEAM_ABBR", "DREB_CAPTURE"]].drop_duplicates(["GAME_ID", "TEAM_ABBR"])
        out = out.merge(cap, on=["GAME_ID", "TEAM_ABBR"], how="left")
    else:
        out["DREB_CAPTURE"] = np.nan

    return out.replace([np.inf, -np.inf], np.nan)


def _is_probability(feature: str) -> bool:
    return feature in {"3P_SHARE", "TOV", "OREB_PER_MISS", "DREB_CAPTURE", "AST_PER_MAKE"}


def _transform(feature: str, v):
    a = np.asarray(v, dtype=float)
    if _is_probability(feature):
        a = np.clip(a, 1e-4, 1.0 - 1e-4)
        return np.log(a / (1.0 - a))
    return np.log(np.clip(a, 1e-4, None))


def _inverse(feature: str, z):
    a = np.asarray(z, dtype=float)
    if _is_probability(feature):
        a = np.clip(a, -12.0, 12.0)
        return 1.0 / (1.0 + np.exp(-a))
    return np.exp(np.clip(a, -12.0, 12.0))


def _mean_feature(df: pd.DataFrame, feature: str, default=np.nan) -> float:
    if df is None or df.empty or feature not in df.columns:
        return float(default)
    v = pd.to_numeric(df[feature], errors="coerce").dropna()
    return float(v.mean()) if len(v) else float(default)


def _cfg_baseline(df: pd.DataFrame, feature: str, league: float, cfg: WeightConfig) -> float:
    """Non-overlapping Old/G6-10/L5 state in raw feature space."""
    if df is None or df.empty:
        return float(league)
    buckets = split_non_overlapping(df.sort_values("GAME_DATE"))
    weights = active_weights(buckets, cfg)
    vals, ws = [], []
    for k in ("old", "mid", "l5"):
        v = _mean_feature(buckets[k], feature, np.nan)
        if np.isfinite(v) and weights.get(k, 0.0) > 0:
            vals.append(v)
            ws.append(weights[k])
    return float(np.average(vals, weights=ws)) if vals else float(league)


def _training_table(team_logs: pd.DataFrame, feature: str) -> pd.DataFrame:
    """One row per team-game with strictly pregame/disjoint states."""
    g = _single_game_features(team_logs).dropna(subset=["GAME_DATE", feature]).copy()
    g = g.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBR"]).reset_index(drop=True)
    if g.empty:
        return pd.DataFrame()

    records = []
    games = (
        g[["GAME_DATE", "GAME_ID"]]
        .drop_duplicates()
        .sort_values(["GAME_DATE", "GAME_ID"])
        .itertuples(index=False, name=None)
    )
    prior = g.iloc[0:0].copy()
    stable = WeightConfig.stable()
    for game_date, game_id in games:
        cur = g[(g["GAME_DATE"] == game_date) & (g["GAME_ID"] == game_id)]
        if len(prior) >= 80:
            league = _mean_feature(prior, feature, np.nan)
            if np.isfinite(league):
                lg_t = float(_transform(feature, [league])[0])
                for _, r in cur.iterrows():
                    team = str(r["TEAM_ABBR"]).upper()
                    opp = str(r["OPP_ABBR"]).upper()
                    # Disjoint current-pair removal on BOTH sides.
                    own_hist = prior[prior["TEAM_ABBR"].eq(team) & ~prior["OPP_ABBR"].eq(opp)]
                    opp_hist = prior[prior["OPP_ABBR"].eq(opp) & ~prior["TEAM_ABBR"].eq(team)]
                    if len(own_hist) < 8 or len(opp_hist) < 8:
                        continue
                    own = _cfg_baseline(own_hist, feature, league, stable)
                    opp_allowed = _cfg_baseline(opp_hist, feature, league, stable)
                    actual = float(r[feature])
                    if not all(np.isfinite(v) for v in (own, opp_allowed, actual)):
                        continue
                    own_dev = float(_transform(feature, [own])[0] - lg_t)
                    opp_dev = float(_transform(feature, [opp_allowed])[0] - lg_t)
                    y = float(_transform(feature, [actual])[0] - lg_t)
                    records.append({
                        "GAME_DATE": game_date,
                        "GAME_ID": str(game_id),
                        "TEAM_ABBR": team,
                        "OPP_ABBR": opp,
                        "OWN_DEV": own_dev,
                        "OPP_DEV": opp_dev,
                        "Y": y,
                        "BASE_Y": own_dev,
                    })
        prior = pd.concat([prior, cur], ignore_index=True)
    return pd.DataFrame(records).sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBR"]).reset_index(drop=True)


def _pair_key(row) -> tuple[str, str]:
    return str(row.TEAM_ABBR), str(row.OPP_ABBR)


def _h2h_term(resids: list[float], k: float) -> tuple[float, float]:
    n = len(resids)
    if n == 0 or not np.isfinite(k):
        return 0.0, 0.0
    w = float(n / (n + max(float(k), 1e-9)))
    return float(w * np.mean(resids)), w


def _sequential_predictions(df: pd.DataFrame, beta: float, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Pregame predictions; pair residual memory is updated only AFTER a row."""
    pred = np.empty(len(df), dtype=float)
    h2h_terms = np.zeros(len(df), dtype=float)
    pair_resids: Dict[tuple[str, str], list[float]] = {}
    for i, r in enumerate(df.itertuples(index=False)):
        key = _pair_key(r)
        base = float(r.OWN_DEV) + float(beta) * float(r.OPP_DEV)
        h2h, _ = _h2h_term(pair_resids.get(key, []), k)
        pred[i] = base + h2h
        h2h_terms[i] = h2h
        # Residual is always against the no-H2H expectation.  This prevents a
        # recursive H2H term from learning its own previous correction.
        resid = float(r.Y) - base
        pair_resids.setdefault(key, []).append(resid)
    return pred, h2h_terms


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(y - p))))


def _select_params(df: pd.DataFrame):
    """Tune on earlier blocked windows; reserve the last 30% for activation."""
    n = len(df)
    if n < 100:
        return 0.0, np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0

    hold_start = max(int(n * 0.70), 70)
    tune_end = hold_start
    # Two expanding, chronologically later tuning blocks inside the first 70%.
    starts = sorted(set([max(60, int(tune_end * 0.55)), max(70, int(tune_end * 0.72))]))
    tune_idx = []
    block = max(15, int(tune_end * 0.12))
    for s in starts:
        e = min(s + block, tune_end)
        if e > s:
            tune_idx.extend(range(s, e))
    tune_idx = np.asarray(sorted(set(tune_idx)), dtype=int)
    if len(tune_idx) < 20:
        tune_idx = np.arange(max(60, int(tune_end * 0.60)), tune_end, dtype=int)

    y = df["Y"].to_numpy(dtype=float)
    base = df["BASE_Y"].to_numpy(dtype=float)
    beta_grid = np.round(np.arange(0.0, 2.0001, 0.05), 2)
    k_grid = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf]

    best = (np.inf, 0.0, np.inf)
    for beta in beta_grid:
        for k in k_grid:
            p, _ = _sequential_predictions(df.iloc[:tune_end], float(beta), float(k))
            score = _rmse(y[tune_idx], p[tune_idx])
            # Prefer simpler/no-H2H model on exact ties.
            complexity = (0 if np.isinf(k) else 1, float(beta))
            best_complexity = (0 if np.isinf(best[2]) else 1, float(best[1]))
            if score < best[0] - 1e-12 or (abs(score - best[0]) <= 1e-12 and complexity < best_complexity):
                best = (score, float(beta), float(k))

    tune_rmse, beta, k = best
    baseline_tune = _rmse(y[tune_idx], base[tune_idx])

    # Completely later holdout, not used to choose beta/K.
    pred_all, _ = _sequential_predictions(df, beta, k)
    hold_idx = np.arange(hold_start, n, dtype=int)
    hold_rmse = _rmse(y[hold_idx], pred_all[hold_idx])
    base_hold = _rmse(y[hold_idx], base[hold_idx])

    # Stress-test the tails of opponent context.  A model that wins on average
    # but breaks against extreme defenses is not activated.
    opp_abs = np.abs(df["OPP_DEV"].to_numpy(dtype=float)[hold_idx])
    if len(opp_abs) >= 20:
        q = float(np.quantile(opp_abs, 0.75))
        ext_local = np.where(opp_abs >= q)[0]
        ext_idx = hold_idx[ext_local]
    else:
        ext_idx = np.asarray([], dtype=int)
    ext_rmse = _rmse(y[ext_idx], pred_all[ext_idx]) if len(ext_idx) else np.nan
    base_ext = _rmse(y[ext_idx], base[ext_idx]) if len(ext_idx) else np.nan

    return (
        beta, k, tune_rmse, baseline_tune,
        hold_rmse, base_hold, ext_rmse, base_ext,
        len(hold_idx), len(ext_idx),
    )


def fit_structural_rate_models(team_logs: pd.DataFrame):
    models: Dict[str, StructuralModel] = {}
    audit_rows = []
    for feature in FEATURES:
        df = _training_table(team_logs, feature)
        if len(df) < 100:
            model = StructuralModel(
                feature, False, 0.0, np.inf, len(df),
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, df,
            )
            models[feature] = model
            audit_rows.append({
                "Feature": feature, "Active": False, "Rows": len(df),
                "Reason": "insufficient chronological rows",
            })
            continue

        beta, k, tune_rmse, base_tune, hold_rmse, base_hold, ext_rmse, base_ext, hn, en = _select_params(df)
        overall_ok = bool(np.isfinite(hold_rmse) and np.isfinite(base_hold) and hold_rmse < base_hold)
        extreme_ok = bool(
            en < 12
            or not (np.isfinite(ext_rmse) and np.isfinite(base_ext))
            or ext_rmse <= base_ext
        )
        active = overall_ok and extreme_ok
        model = StructuralModel(
            feature, active, beta, k, len(df), tune_rmse, base_tune,
            hold_rmse, base_hold, ext_rmse, base_ext, hn, en, df,
        )
        models[feature] = model
        audit_rows.append({
            "Feature": feature,
            "Active": active,
            "Rows": len(df),
            "Opponent beta": beta,
            "H2H prior K": k,
            "Tune RMSE": tune_rmse,
            "Own-only tune RMSE": base_tune,
            "Later holdout RMSE": hold_rmse,
            "Own-only holdout RMSE": base_hold,
            "Extreme-opponent holdout RMSE": ext_rmse,
            "Own-only extreme RMSE": base_ext,
            "Holdout rows": hn,
            "Extreme holdout rows": en,
            "Reason": (
                "later holdout + extreme-opponent improvement"
                if active else
                ("no later holdout improvement" if not overall_ok else "fails extreme-opponent stress test")
            ),
        })
    return models, pd.DataFrame(audit_rows)


def _current_pair_residuals(model: StructuralModel, team: str, opp: str) -> list[float]:
    if model.training_table is None or model.training_table.empty:
        return []
    h = model.training_table[
        model.training_table["TEAM_ABBR"].eq(team)
        & model.training_table["OPP_ABBR"].eq(opp)
    ]
    if h.empty:
        return []
    base = h["OWN_DEV"].to_numpy(dtype=float) + float(model.opponent_beta) * h["OPP_DEV"].to_numpy(dtype=float)
    return (h["Y"].to_numpy(dtype=float) - base).tolist()


def predict_structural_modifiers(
    team_logs: pd.DataFrame,
    team_abbr: str,
    opponent_abbr: str,
    models: Dict[str, StructuralModel],
    cfg: WeightConfig,
    h2h_rotation_similarity: float = 1.0,
):
    """Return current opponent + residualized-H2H modifiers and a full audit."""
    g = _single_game_features(team_logs).dropna(subset=["GAME_DATE"]).copy()
    team = str(team_abbr).upper()
    opp = str(opponent_abbr).upper()
    own_hist = g[g["TEAM_ABBR"].eq(team) & ~g["OPP_ABBR"].eq(opp)].copy()
    opp_hist = g[g["OPP_ABBR"].eq(opp) & ~g["TEAM_ABBR"].eq(team)].copy()
    raw_h2h = g[g["TEAM_ABBR"].eq(team) & g["OPP_ABBR"].eq(opp)].copy()

    mods = {"3P_SHARE": 1.0, "FTA": 1.0, "TOV": 1.0, "OREB": 1.0, "DREB": 1.0, "AST": 1.0}
    audit = []
    mapping = {
        "3P_SHARE": "3P_SHARE",
        "FTA": "FTA",
        "TOV": "TOV",
        "OREB_PER_MISS": "OREB",
        "DREB_CAPTURE": "DREB",
        "AST_PER_MAKE": "AST",
    }

    for feature in FEATURES:
        model = models.get(feature)
        league = _mean_feature(g, feature, np.nan)
        baseline = _cfg_baseline(own_hist, feature, league, cfg) if np.isfinite(league) else np.nan
        opp_state = _cfg_baseline(opp_hist, feature, league, WeightConfig.stable()) if np.isfinite(league) else np.nan
        pred_no_h2h = baseline
        pred = baseline
        h2h_raw_resid = np.nan
        h2h_weight = 0.0
        h2h_effect_t = 0.0
        usable_h2h = 0

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

            residuals = _current_pair_residuals(model, team, opp)
            usable_h2h = len(residuals)
            if usable_h2h:
                h2h_raw_resid = float(np.mean(residuals))
                if np.isfinite(model.h2h_prior_k):
                    h2h_weight = float(usable_h2h / (usable_h2h + max(model.h2h_prior_k, 1e-9)))
                    h2h_weight *= float(np.clip(h2h_rotation_similarity, 0.0, 1.0))
                    h2h_effect_t = h2h_weight * h2h_raw_resid
            pred = float(_inverse(feature, [no_h2h_t + h2h_effect_t])[0])

        # Physical bounds only.  These are not matchup-strength caps.
        if feature == "3P_SHARE":
            pred = float(np.clip(pred, 0.06, 0.75)); pred_no_h2h = float(np.clip(pred_no_h2h, 0.06, 0.75))
        elif feature == "TOV":
            pred = float(np.clip(pred, 0.03, 0.30)); pred_no_h2h = float(np.clip(pred_no_h2h, 0.03, 0.30))
        elif feature == "FTA":
            pred = float(np.clip(pred, 0.05, 0.55)); pred_no_h2h = float(np.clip(pred_no_h2h, 0.05, 0.55))
        elif feature == "OREB_PER_MISS":
            pred = float(np.clip(pred, 0.05, 0.55)); pred_no_h2h = float(np.clip(pred_no_h2h, 0.05, 0.55))
        elif feature == "DREB_CAPTURE":
            pred = float(np.clip(pred, 0.70, 0.995)); pred_no_h2h = float(np.clip(pred_no_h2h, 0.70, 0.995))
        elif feature == "AST_PER_MAKE":
            pred = float(np.clip(pred, 0.20, 0.95)); pred_no_h2h = float(np.clip(pred_no_h2h, 0.20, 0.95))

        mod = float(pred / baseline) if np.isfinite(pred) and np.isfinite(baseline) and baseline > 0 else 1.0
        mods[mapping[feature]] = mod
        audit.append({
            "Feature": feature,
            "Model active": bool(model.active) if model else False,
            "Own non-H2H state": baseline,
            "Opponent allowed state": opp_state,
            "Opponent beta": float(model.opponent_beta) if model else 0.0,
            "Prediction without H2H": pred_no_h2h,
            "Raw H2H games": int(len(raw_h2h)),
            "Usable residual H2H games": usable_h2h,
            "H2H residual mean (transformed)": h2h_raw_resid,
            "H2H prior K": float(model.h2h_prior_k) if model else np.inf,
            "H2H rotation similarity": float(h2h_rotation_similarity),
            "Effective H2H weight": h2h_weight,
            "H2H transformed delta": h2h_effect_t,
            "Final learned prediction": pred,
            "H2H raw-unit delta": (pred - pred_no_h2h) if np.isfinite(pred) and np.isfinite(pred_no_h2h) else np.nan,
            "Applied modifier": mod,
            "Training rows": int(model.rows) if model else 0,
            "Later holdout RMSE": float(model.holdout_rmse) if model else np.nan,
            "Own-only holdout RMSE": float(model.baseline_holdout_rmse) if model else np.nan,
            "Extreme holdout RMSE": float(model.extreme_holdout_rmse) if model else np.nan,
            "Own-only extreme RMSE": float(model.baseline_extreme_holdout_rmse) if model else np.nan,
        })
    return mods, pd.DataFrame(audit)


def coefficient_audit(models: Dict[str, StructuralModel]) -> pd.DataFrame:
    rows = []
    for feature, model in models.items():
        rows.extend([
            {
                "Feature": feature,
                "Term": "Opponent beta",
                "Coefficient": float(model.opponent_beta),
                "Active": model.active,
                "Rows": model.rows,
                "Later holdout RMSE": model.holdout_rmse,
                "Baseline holdout RMSE": model.baseline_holdout_rmse,
            },
            {
                "Feature": feature,
                "Term": "H2H prior K",
                "Coefficient": float(model.h2h_prior_k),
                "Active": model.active,
                "Rows": model.rows,
                "Later holdout RMSE": model.holdout_rmse,
                "Baseline holdout RMSE": model.baseline_holdout_rmse,
            },
        ])
    return pd.DataFrame(rows)
