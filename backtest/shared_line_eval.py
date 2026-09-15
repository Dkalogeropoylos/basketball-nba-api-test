from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import _game_pairs
from backtest.projection import MARKETS, project_game_pregame
from backtest.variance_experiment import NBA_FTA_LOG_SIGMA_EARLY_HALF_2024_25


B3_FGA_PROCESS = "rebound_chain"
B3_FTA_LOG_SIGMA = float(NBA_FTA_LOG_SIGMA_EARLY_HALF_2024_25)


def load_fixed_a_lines(path: str | Path) -> pd.DataFrame:
    """Load the frozen baseline-A half-point lines/probabilities for the late 498 games."""
    df = pd.read_csv(path, dtype={"GAME_ID": str})
    required = {
        "GAME_ID", "Scope", "Team", "Market", "Line",
        "P_Over", "P_Under", "P_Push", "Over_Win", "Under_Win",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Frozen A reference is missing columns: {sorted(missing)}")

    for c in ["Line", "P_Over", "P_Under", "P_Push", "Over_Win", "Under_Win"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Scope"] = df["Scope"].astype(str)
    df["Market"] = df["Market"].astype(str)
    df["Team"] = df["Team"].fillna("").astype(str)
    df = df.dropna(subset=["Line", "P_Over", "Over_Win"]).reset_index(drop=True)
    return df


def reference_audit(a_lines: pd.DataFrame) -> dict:
    if a_lines is None or a_lines.empty:
        return {"games": 0, "rows": 0, "push_rows": 0, "non_half_lines": 0}
    frac = np.mod(np.abs(a_lines["Line"].to_numpy(dtype=float)), 1.0)
    half = np.isclose(frac, 0.5, atol=1e-9)
    return {
        "games": int(a_lines["GAME_ID"].astype(str).nunique()),
        "rows": int(len(a_lines)),
        "push_rows": int((pd.to_numeric(a_lines["P_Push"], errors="coerce").fillna(0.0) > 0).sum()),
        "non_half_lines": int((~half).sum()),
    }


def _prob_at_lines(values: np.ndarray, lines: np.ndarray):
    v = np.asarray(values, dtype=float)
    l = np.asarray(lines, dtype=float)
    gt = (v[:, None] > l[None, :]).mean(axis=0)
    lt = (v[:, None] < l[None, :]).mean(axis=0)
    eq = (v[:, None] == l[None, :]).mean(axis=0)
    return gt, lt, eq


def run_shared_line_batch(
    team_db: pd.DataFrame,
    player_db: pd.DataFrame,
    calibration,
    a_lines: pd.DataFrame,
    *,
    start_index: int,
    max_games: int,
    min_prior_games: int = 15,
    n_sims: int = 1500,
    use_rotation_similarity: bool = True,
    progress_callback=None,
):
    """Rerun B3 only, but score it on the exact frozen baseline-A lines.

    Seeds use the same absolute season index as the C4 walk-forward backtest.
    Baseline A is never rerun: its lines/probabilities/outcomes come from a_lines.
    """
    all_games = _game_pairs(team_db, min_prior_games=min_prior_games)
    eligible = len(all_games)
    start_index = int(max(start_index, 0))
    end_index = min(eligible, start_index + int(max_games))
    games = all_games[start_index:end_index]

    rows: list[dict] = []
    failures: list[dict] = []
    team_dates = pd.to_datetime(team_db["GAME_DATE"], errors="coerce")
    player_dates = (
        pd.to_datetime(player_db["GAME_DATE"], errors="coerce")
        if player_db is not None and not player_db.empty
        else None
    )

    # Index once so we do not scan 150k frozen rows for every historical game.
    a_by_game = {gid: g for gid, g in a_lines.groupby(a_lines["GAME_ID"].astype(str), sort=False)}

    for local_i, (gid, date, hr, ar, hp, ap) in enumerate(games):
        global_i = start_index + local_i
        gid = str(gid)
        try:
            ref = a_by_game.get(gid)
            if ref is None or ref.empty:
                raise ValueError(f"No frozen A lines found for GAME_ID={gid}")

            team_hist = team_db.loc[team_dates < date].copy()
            player_hist = (
                player_db.loc[player_dates < date].copy()
                if player_dates is not None
                else pd.DataFrame()
            )
            home = str(hr["TEAM_ABBR"]).upper()
            away = str(ar["TEAM_ABBR"]).upper()

            pack = project_game_pregame(
                team_hist,
                player_hist,
                home,
                away,
                calibration,
                n_sims=int(n_sims),
                seed=10000 + global_i,
                use_rotation_similarity=bool(use_rotation_similarity),
                fga_process=B3_FGA_PROCESS,
                fta_log_sigma=B3_FTA_LOG_SIGMA,
            )
            hs, as_ = pack["home"], pack["away"]
            total_sim = hs[MARKETS].add(as_[MARKETS], fill_value=0)

            for (scope, team, market), rg in ref.groupby(
                ["Scope", "Team", "Market"], dropna=False, sort=False
            ):
                market = str(market)
                if market not in MARKETS:
                    continue

                if str(scope) == "TOTAL":
                    sim = total_sim[market].to_numpy(dtype=float)
                else:
                    team_s = str(team).upper()
                    if team_s == home:
                        sim = hs[market].to_numpy(dtype=float)
                    elif team_s == away:
                        sim = as_[market].to_numpy(dtype=float)
                    else:
                        raise ValueError(
                            f"Reference team {team_s} not in {home}-{away} for GAME_ID={gid}"
                        )

                rg = rg.reset_index(drop=True)
                lines = rg["Line"].to_numpy(dtype=float)
                po, pu, pp = _prob_at_lines(sim, lines)
                for j, rr in rg.iterrows():
                    rows.append(
                        {
                            "GAME_ID": gid,
                            "GAME_DATE": pd.Timestamp(date),
                            "Scope": str(scope),
                            "Team": str(team),
                            "Market": market,
                            "Line": float(rr["Line"]),
                            "P_A_Over": float(rr["P_Over"]),
                            "P_A_Under": float(rr["P_Under"]),
                            "P_A_Push": float(rr["P_Push"]),
                            "P_B3_Over": float(po[j]),
                            "P_B3_Under": float(pu[j]),
                            "P_B3_Push": float(pp[j]),
                            "Y_Over": float(rr["Over_Win"]),
                            "Y_Under": float(rr["Under_Win"]),
                            "Home": home,
                            "Away": away,
                        }
                    )
        except Exception as exc:
            failures.append({"GAME_ID": gid, "GAME_DATE": date, "Error": repr(exc)})

        if progress_callback is not None:
            progress_callback(local_i + 1, len(games))

    return pd.DataFrame(rows), pd.DataFrame(failures), int(end_index), int(eligible)


def _safe_logloss(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def shared_line_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Paired A-vs-B3 scoring on identical binary events/lines."""
    if rows is None or rows.empty:
        return pd.DataFrame()

    x = rows.copy()
    x["Brier_A"] = (x["P_A_Over"] - x["Y_Over"]) ** 2
    x["Brier_B3"] = (x["P_B3_Over"] - x["Y_Over"]) ** 2
    x["LogLoss_A"] = _safe_logloss(x["P_A_Over"].to_numpy(), x["Y_Over"].to_numpy())
    x["LogLoss_B3"] = _safe_logloss(x["P_B3_Over"].to_numpy(), x["Y_Over"].to_numpy())

    out: list[dict] = []
    for (scope, market), g in x.groupby(["Scope", "Market"], sort=True):
        # Cluster first at GAME_ID so the confidence interval does not pretend
        # that several synthetic lines from the same game are independent bets.
        by_game = (
            g.assign(delta_brier=g["Brier_B3"] - g["Brier_A"])
            .groupby("GAME_ID", as_index=False)["delta_brier"]
            .mean()
        )
        vals = by_game["delta_brier"].to_numpy(dtype=float)
        if len(vals) >= 2:
            rng = np.random.default_rng(20260916)
            idx = rng.integers(0, len(vals), size=(2000, len(vals)))
            boot = vals[idx].mean(axis=1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
        else:
            lo = hi = np.nan

        out.append(
            {
                "Scope": scope,
                "Market": market,
                "N lines": int(len(g)),
                "N games": int(g["GAME_ID"].astype(str).nunique()),
                "Brier A": float(g["Brier_A"].mean()),
                "Brier B3": float(g["Brier_B3"].mean()),
                "Brier Δ B3-A": float((g["Brier_B3"] - g["Brier_A"]).mean()),
                "Brier Δ 95% low": float(lo),
                "Brier Δ 95% high": float(hi),
                "LogLoss A": float(g["LogLoss_A"].mean()),
                "LogLoss B3": float(g["LogLoss_B3"].mean()),
                "LogLoss Δ B3-A": float((g["LogLoss_B3"] - g["LogLoss_A"]).mean()),
                "Mean |P_B3-P_A|": float((g["P_B3_Over"] - g["P_A_Over"]).abs().mean()),
            }
        )
    return pd.DataFrame(out)


def shared_line_calibration_bins(rows: pd.DataFrame, model_col: str) -> pd.DataFrame:
    if rows is None or rows.empty:
        return pd.DataFrame()
    if model_col not in {"P_A_Over", "P_B3_Over"}:
        raise ValueError("model_col must be P_A_Over or P_B3_Over")

    x = rows.copy()
    bins = np.arange(0.0, 1.0001, 0.05)
    x["Probability bin"] = pd.cut(
        x[model_col], bins=bins, include_lowest=True, right=False
    )
    out: list[dict] = []
    for (scope, market, b), g in x.groupby(
        ["Scope", "Market", "Probability bin"], observed=True
    ):
        if len(g) == 0:
            continue
        mp = float(g[model_col].mean())
        hit = float(g["Y_Over"].mean())
        out.append(
            {
                "Scope": scope,
                "Market": market,
                "Probability bin": str(b),
                "N": int(len(g)),
                "Mean predicted": mp,
                "Actual hit rate": hit,
                "Calibration gap": hit - mp,
                "Brier": float(np.mean((g[model_col] - g["Y_Over"]) ** 2)),
            }
        )
    return pd.DataFrame(out)


def checkpoint_bytes(rows: pd.DataFrame, failures: pd.DataFrame, meta: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("shared_line_rows.csv", rows.to_csv(index=False))
        z.writestr("failures.csv", failures.to_csv(index=False))
        z.writestr("meta.json", json.dumps(meta, indent=2))
    return buf.getvalue()


def restore_checkpoint(raw: bytes):
    with zipfile.ZipFile(io.BytesIO(raw), "r") as z:
        def read_csv(name: str):
            with z.open(name) as f:
                try:
                    return pd.read_csv(f, dtype={"GAME_ID": str})
                except pd.errors.EmptyDataError:
                    return pd.DataFrame()

        rows = read_csv("shared_line_rows.csv")
        failures = read_csv("failures.csv")
        meta = json.loads(z.read("meta.json").decode("utf-8"))
        return rows, failures, meta
