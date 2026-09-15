from __future__ import annotations

import io
import json
import time
import zipfile

import numpy as np
import pandas as pd

from backtest.engine import _actual_stat, _game_pairs
from backtest.evaluation import most_summary, pseudo_line_rows, sample_summary
from backtest.projection import MARKETS, project_game_pregame
from backtest.shared_line_eval import B3_FGA_PROCESS, B3_FTA_LOG_SIGMA, shared_line_summary, shared_line_calibration_bins


FINAL_MIN_PRIOR_GAMES = 15
FINAL_N_SIMS = 1500
FINAL_ROTATION = True
BASELINE_FGA_PROCESS = "poisson"
BASELINE_FTA_LOG_SIGMA = 0.12


def _prob_at_lines(values: np.ndarray, lines: np.ndarray):
    v = np.asarray(values, dtype=float)
    l = np.asarray(lines, dtype=float)
    gt = (v[:, None] > l[None, :]).mean(axis=0)
    lt = (v[:, None] < l[None, :]).mean(axis=0)
    eq = (v[:, None] == l[None, :]).mean(axis=0)
    return gt, lt, eq


def _append_summary_rows(
    rows: list[dict],
    *,
    variant: str,
    gid: str,
    date,
    scope: str,
    team: str,
    home: str,
    away: str,
    market: str,
    actual: float,
    sim: np.ndarray,
    hp: int,
    ap: int,
    projected_possessions: float,
):
    s = sample_summary(sim, actual)
    rows.append(
        {
            "Variant": variant,
            "GAME_ID": gid,
            "GAME_DATE": pd.Timestamp(date),
            "Scope": scope,
            "Team": team,
            "Home": home,
            "Away": away,
            "Market": market,
            "Actual": float(actual),
            "Prior_games_home": int(hp),
            "Prior_games_away": int(ap),
            "Projected_possessions": float(projected_possessions),
            **s,
        }
    )


def run_final_exam_batch(
    team_db: pd.DataFrame,
    player_db: pd.DataFrame,
    calibration,
    *,
    start_index: int,
    max_games: int,
    progress_callback=None,
):
    """Run the frozen final 2025-26 exam for baseline A and candidate B3.

    Protocol is deliberately fixed:
      - min prior current-season games/team = 15
      - simulations/game = 1500
      - rotation similarity = ON
      - A = frozen C3 baseline stochastic layer
      - B3 = frozen FGA rebound-chain + frozen FTA dispersion

    The function also creates SAME-LINE probability rows. The half-point lines are
    generated from baseline A pregame simulations only (never from the actual), and
    both A and B3 are scored on those exact lines.
    """
    t0 = time.time()
    all_games = _game_pairs(team_db, min_prior_games=FINAL_MIN_PRIOR_GAMES)
    eligible = len(all_games)
    start_index = int(max(start_index, 0))
    end_index = min(eligible, start_index + int(max_games))
    games = all_games[start_index:end_index]

    detail_rows: list[dict] = []
    shared_rows: list[dict] = []
    most_rows: list[dict] = []
    failures: list[dict] = []

    team_dates = pd.to_datetime(team_db["GAME_DATE"], errors="coerce")
    player_dates = (
        pd.to_datetime(player_db["GAME_DATE"], errors="coerce")
        if player_db is not None and not player_db.empty
        else None
    )

    most_markets = ["3PM", "3PA", "2PM", "2PA", "FTM", "FTA", "REB", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF"]

    for local_i, (gid, date, hr, ar, hp, ap) in enumerate(games):
        global_i = start_index + local_i
        gid = str(gid)
        try:
            team_hist = team_db.loc[team_dates < date].copy()
            player_hist = (
                player_db.loc[player_dates < date].copy()
                if player_dates is not None
                else pd.DataFrame()
            )
            home = str(hr["TEAM_ABBR"]).upper()
            away = str(ar["TEAM_ABBR"]).upper()
            seed = 10000 + global_i

            pack_a = project_game_pregame(
                team_hist,
                player_hist,
                home,
                away,
                calibration,
                n_sims=FINAL_N_SIMS,
                seed=seed,
                use_rotation_similarity=FINAL_ROTATION,
                fga_process=BASELINE_FGA_PROCESS,
                fta_log_sigma=BASELINE_FTA_LOG_SIGMA,
            )
            pack_b = project_game_pregame(
                team_hist,
                player_hist,
                home,
                away,
                calibration,
                n_sims=FINAL_N_SIMS,
                seed=seed,
                use_rotation_similarity=FINAL_ROTATION,
                fga_process=B3_FGA_PROCESS,
                fta_log_sigma=B3_FTA_LOG_SIGMA,
            )

            a_home, a_away = pack_a["home"], pack_a["away"]
            b_home, b_away = pack_b["home"], pack_b["away"]
            a_total = a_home[MARKETS].add(a_away[MARKETS], fill_value=0)
            b_total = b_home[MARKETS].add(b_away[MARKETS], fill_value=0)

            scopes = [
                ("TEAM", home, a_home, b_home, hr),
                ("TEAM", away, a_away, b_away, ar),
            ]

            for scope, team, sim_a_df, sim_b_df, actual_row in scopes:
                for market in MARKETS:
                    actual = _actual_stat(actual_row, market)
                    sim_a = sim_a_df[market].to_numpy(dtype=float)
                    sim_b = sim_b_df[market].to_numpy(dtype=float)

                    _append_summary_rows(
                        detail_rows,
                        variant="A",
                        gid=gid,
                        date=date,
                        scope=scope,
                        team=team,
                        home=home,
                        away=away,
                        market=market,
                        actual=actual,
                        sim=sim_a,
                        hp=hp,
                        ap=ap,
                        projected_possessions=float(pack_a["pace"]["central"]),
                    )
                    _append_summary_rows(
                        detail_rows,
                        variant="B3",
                        gid=gid,
                        date=date,
                        scope=scope,
                        team=team,
                        home=home,
                        away=away,
                        market=market,
                        actual=actual,
                        sim=sim_b,
                        hp=hp,
                        ap=ap,
                        projected_possessions=float(pack_b["pace"]["central"]),
                    )

                    # Shared-line final test: A defines the pregame half-point grid.
                    ref = pseudo_line_rows(sim_a, actual, market, scope, gid, team)
                    if ref:
                        lines = np.asarray([r["Line"] for r in ref], dtype=float)
                        pbo, pbu, pbp = _prob_at_lines(sim_b, lines)
                        for j, r in enumerate(ref):
                            shared_rows.append(
                                {
                                    "GAME_ID": gid,
                                    "GAME_DATE": pd.Timestamp(date),
                                    "Scope": scope,
                                    "Team": team,
                                    "Home": home,
                                    "Away": away,
                                    "Market": market,
                                    "Line": float(r["Line"]),
                                    "P_A_Over": float(r["P_Over"]),
                                    "P_A_Under": float(r["P_Under"]),
                                    "P_A_Push": float(r["P_Push"]),
                                    "P_B3_Over": float(pbo[j]),
                                    "P_B3_Under": float(pbu[j]),
                                    "P_B3_Push": float(pbp[j]),
                                    "Y_Over": float(r["Over_Win"]),
                                    "Y_Under": float(r["Under_Win"]),
                                }
                            )

            for market in MARKETS:
                actual = _actual_stat(hr, market) + _actual_stat(ar, market)
                sim_a = a_total[market].to_numpy(dtype=float)
                sim_b = b_total[market].to_numpy(dtype=float)
                _append_summary_rows(
                    detail_rows,
                    variant="A",
                    gid=gid,
                    date=date,
                    scope="TOTAL",
                    team="TOTAL",
                    home=home,
                    away=away,
                    market=market,
                    actual=actual,
                    sim=sim_a,
                    hp=hp,
                    ap=ap,
                    projected_possessions=float(pack_a["pace"]["central"]),
                )
                _append_summary_rows(
                    detail_rows,
                    variant="B3",
                    gid=gid,
                    date=date,
                    scope="TOTAL",
                    team="TOTAL",
                    home=home,
                    away=away,
                    market=market,
                    actual=actual,
                    sim=sim_b,
                    hp=hp,
                    ap=ap,
                    projected_possessions=float(pack_b["pace"]["central"]),
                )

                ref = pseudo_line_rows(sim_a, actual, market, "TOTAL", gid, None)
                if ref:
                    lines = np.asarray([r["Line"] for r in ref], dtype=float)
                    pbo, pbu, pbp = _prob_at_lines(sim_b, lines)
                    for j, r in enumerate(ref):
                        shared_rows.append(
                            {
                                "GAME_ID": gid,
                                "GAME_DATE": pd.Timestamp(date),
                                "Scope": "TOTAL",
                                "Team": "TOTAL",
                                "Home": home,
                                "Away": away,
                                "Market": market,
                                "Line": float(r["Line"]),
                                "P_A_Over": float(r["P_Over"]),
                                "P_A_Under": float(r["P_Under"]),
                                "P_A_Push": float(r["P_Push"]),
                                "P_B3_Over": float(pbo[j]),
                                "P_B3_Under": float(pbu[j]),
                                "P_B3_Push": float(pbp[j]),
                                "Y_Over": float(r["Over_Win"]),
                                "Y_Under": float(r["Under_Win"]),
                            }
                        )

            for market in most_markets:
                actual_h = _actual_stat(hr, market)
                actual_a = _actual_stat(ar, market)
                for variant, hs, aw in [
                    ("A", a_home, a_away),
                    ("B3", b_home, b_away),
                ]:
                    hv = hs[market].to_numpy(dtype=float)
                    av = aw[market].to_numpy(dtype=float)
                    most_rows.append(
                        {
                            "Variant": variant,
                            "GAME_ID": gid,
                            "GAME_DATE": pd.Timestamp(date),
                            "Home": home,
                            "Away": away,
                            "Market": market,
                            "P_Home": float(np.mean(hv > av)),
                            "P_Away": float(np.mean(av > hv)),
                            "P_Tie": float(np.mean(hv == av)),
                            "Y_Home": float(actual_h > actual_a),
                            "Y_Away": float(actual_a > actual_h),
                            "Y_Tie": float(actual_h == actual_a),
                        }
                    )
        except Exception as exc:
            failures.append({"GAME_ID": gid, "GAME_DATE": date, "Error": repr(exc)})

        if progress_callback is not None:
            progress_callback(local_i + 1, len(games))

    return {
        "detail": pd.DataFrame(detail_rows),
        "shared": pd.DataFrame(shared_rows),
        "most": pd.DataFrame(most_rows),
        "failures": pd.DataFrame(failures),
        "runtime_seconds": float(time.time() - t0),
        "start_index": int(start_index),
        "end_index": int(end_index),
        "eligible_games": int(eligible),
    }


def variant_metric_table(detail: pd.DataFrame) -> pd.DataFrame:
    if detail is None or detail.empty:
        return pd.DataFrame()
    d = detail.copy()
    d["Error"] = pd.to_numeric(d["Projection"], errors="coerce") - pd.to_numeric(d["Actual"], errors="coerce")
    rows = []
    for (variant, scope, market), g in d.groupby(["Variant", "Scope", "Market"], sort=True):
        e = g["Error"].to_numpy(dtype=float)
        rows.append(
            {
                "Variant": variant,
                "Scope": scope,
                "Market": market,
                "N": int(len(g)),
                "Bias": float(np.mean(e)),
                "MAE": float(np.mean(np.abs(e))),
                "RMSE": float(np.sqrt(np.mean(e * e))),
                "50% coverage": float(pd.to_numeric(g["Cover50"], errors="coerce").mean()),
                "80% coverage": float(pd.to_numeric(g["Cover80"], errors="coerce").mean()),
                "90% coverage": float(pd.to_numeric(g["Cover90"], errors="coerce").mean()),
                "Mean CRPS": float(pd.to_numeric(g["CRPS"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def center_head_to_head(detail: pd.DataFrame) -> pd.DataFrame:
    """Compare A and B3 centers on exactly the same game/team/market rows."""
    if detail is None or detail.empty:
        return pd.DataFrame()
    keys = ["GAME_ID", "Scope", "Team", "Market"]
    cols = keys + ["Actual", "Projection"]
    a = detail[detail["Variant"].eq("A")][cols].copy().rename(columns={"Projection": "Projection_A", "Actual": "Actual_A"})
    b = detail[detail["Variant"].eq("B3")][cols].copy().rename(columns={"Projection": "Projection_B3", "Actual": "Actual_B3"})
    x = a.merge(b, on=keys, how="inner")
    if x.empty:
        return pd.DataFrame()
    x["Actual"] = x["Actual_A"]
    x["AbsErr_A"] = (x["Projection_A"] - x["Actual"]).abs()
    x["AbsErr_B3"] = (x["Projection_B3"] - x["Actual"]).abs()
    x["SqErr_A"] = (x["Projection_A"] - x["Actual"]) ** 2
    x["SqErr_B3"] = (x["Projection_B3"] - x["Actual"]) ** 2
    tol = 1e-12
    x["B3_closer"] = x["AbsErr_B3"] + tol < x["AbsErr_A"]
    x["A_closer"] = x["AbsErr_A"] + tol < x["AbsErr_B3"]
    x["Tie"] = ~(x["B3_closer"] | x["A_closer"])

    rows = []
    for (scope, market), g in x.groupby(["Scope", "Market"], sort=True):
        rows.append(
            {
                "Scope": scope,
                "Market": market,
                "N": int(len(g)),
                "MAE A": float(g["AbsErr_A"].mean()),
                "MAE B3": float(g["AbsErr_B3"].mean()),
                "RMSE A": float(np.sqrt(g["SqErr_A"].mean())),
                "RMSE B3": float(np.sqrt(g["SqErr_B3"].mean())),
                "B3 closer": int(g["B3_closer"].sum()),
                "A closer": int(g["A_closer"].sum()),
                "Ties": int(g["Tie"].sum()),
                "Mean |center shift|": float((g["Projection_B3"] - g["Projection_A"]).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def final_most_summary(most_rows: pd.DataFrame) -> pd.DataFrame:
    if most_rows is None or most_rows.empty:
        return pd.DataFrame()
    out = []
    for variant, g in most_rows.groupby("Variant", sort=True):
        s = most_summary(g)
        if not s.empty:
            s.insert(0, "Variant", variant)
            out.append(s)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def final_shared_summary(shared_rows: pd.DataFrame) -> pd.DataFrame:
    return shared_line_summary(shared_rows)


def final_shared_calibration(shared_rows: pd.DataFrame, model: str) -> pd.DataFrame:
    col = "P_A_Over" if model == "A" else "P_B3_Over"
    return shared_line_calibration_bins(shared_rows, col)


def checkpoint_bytes(detail: pd.DataFrame, shared: pd.DataFrame, most: pd.DataFrame, failures: pd.DataFrame, meta: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("detail.csv", detail.to_csv(index=False))
        z.writestr("shared_line_rows.csv", shared.to_csv(index=False))
        z.writestr("most_rows.csv", most.to_csv(index=False))
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

        return (
            read_csv("detail.csv"),
            read_csv("shared_line_rows.csv"),
            read_csv("most_rows.csv"),
            read_csv("failures.csv"),
            json.loads(z.read("meta.json").decode("utf-8")),
        )
