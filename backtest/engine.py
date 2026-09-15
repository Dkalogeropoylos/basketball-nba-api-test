from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
import pandas as pd

from backtest.projection import MARKETS, project_game_pregame
from backtest.evaluation import sample_summary, pseudo_line_rows


@dataclass
class BacktestResult:
    detail: pd.DataFrame
    probability_rows: pd.DataFrame
    most_rows: pd.DataFrame
    failures: pd.DataFrame
    runtime_seconds: float


def _game_pairs(team_db: pd.DataFrame, min_prior_games: int = 15, start_date=None, end_date=None):
    x = team_db.copy()
    x["GAME_DATE"] = pd.to_datetime(x["GAME_DATE"], errors="coerce")
    x = x.dropna(subset=["GAME_DATE"]).sort_values(["GAME_DATE","GAME_ID","TEAM_ABBR"])
    if start_date is not None:
        x = x[x["GAME_DATE"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        x = x[x["GAME_DATE"] <= pd.Timestamp(end_date)]

    full = team_db.copy()
    full["GAME_DATE"] = pd.to_datetime(full["GAME_DATE"], errors="coerce")
    out = []
    for gid, g in x.groupby("GAME_ID", sort=False):
        if len(g) != 2:
            continue
        homes = g[g.get("HOME_AWAY", "").astype(str).str.lower().str.startswith("h")] if "HOME_AWAY" in g.columns else pd.DataFrame()
        aways = g[g.get("HOME_AWAY", "").astype(str).str.lower().str.startswith("a")] if "HOME_AWAY" in g.columns else pd.DataFrame()
        if len(homes) != 1 or len(aways) != 1:
            continue
        hr, ar = homes.iloc[0], aways.iloc[0]
        date = pd.Timestamp(hr["GAME_DATE"])
        hp = full[(full["TEAM_ABBR"].astype(str).str.upper() == str(hr["TEAM_ABBR"]).upper()) & (full["GAME_DATE"] < date)]["GAME_ID"].nunique()
        ap = full[(full["TEAM_ABBR"].astype(str).str.upper() == str(ar["TEAM_ABBR"]).upper()) & (full["GAME_DATE"] < date)]["GAME_ID"].nunique()
        if hp < min_prior_games or ap < min_prior_games:
            continue
        out.append((str(gid), date, hr, ar, int(hp), int(ap)))
    return out


def _actual_stat(row, market: str) -> float:
    def num(key):
        return float(pd.to_numeric(pd.Series([row.get(key, np.nan)]), errors="coerce").iloc[0])
    if market == "3PA": return num("FG3A")
    if market == "3PM": return num("FG3M")
    if market == "2PA": return num("FGA") - num("FG3A")
    if market == "2PM": return num("FGM") - num("FG3M")
    return num(market)


def run_walk_forward(
    team_db: pd.DataFrame,
    player_db: pd.DataFrame,
    calibration,
    min_prior_games: int = 15,
    n_sims: int = 2500,
    max_games: int | None = None,
    start_date=None,
    end_date=None,
    use_rotation_similarity: bool = True,
    progress_callback=None,
) -> BacktestResult:
    t0 = time.time()
    games = _game_pairs(team_db, min_prior_games=min_prior_games, start_date=start_date, end_date=end_date)
    if max_games is not None and max_games > 0:
        games = games[: int(max_games)]

    detail_rows = []
    prob_rows = []
    most_rows = []
    failures = []

    for i, (gid, date, hr, ar, hp, ap) in enumerate(games):
        try:
            team_hist = team_db[pd.to_datetime(team_db["GAME_DATE"], errors="coerce") < date].copy()
            player_hist = player_db[pd.to_datetime(player_db["GAME_DATE"], errors="coerce") < date].copy()
            home = str(hr["TEAM_ABBR"]).upper(); away = str(ar["TEAM_ABBR"]).upper()
            pack = project_game_pregame(
                team_hist, player_hist, home, away, calibration,
                n_sims=n_sims, seed=10000 + i, use_rotation_similarity=use_rotation_similarity,
            )
            hs, as_ = pack["home"], pack["away"]
            for scope, team, sim, actual_row in [
                ("TEAM", home, hs, hr), ("TEAM", away, as_, ar)
            ]:
                for m in MARKETS:
                    actual = _actual_stat(actual_row, m)
                    s = sample_summary(sim[m].to_numpy(), actual)
                    detail_rows.append({
                        "GAME_ID": gid, "GAME_DATE": date, "Scope": scope, "Team": team,
                        "Home": home, "Away": away, "Market": m, "Actual": actual,
                        "Prior_games_home": hp, "Prior_games_away": ap,
                        "Projected_possessions": pack["pace"]["central"], **s,
                    })
                    prob_rows.extend(pseudo_line_rows(sim[m].to_numpy(), actual, m, scope, gid, team))

            total_sim = hs[MARKETS].add(as_[MARKETS], fill_value=0)
            for m in MARKETS:
                hact = _actual_stat(hr, m)
                aact = _actual_stat(ar, m)
                actual = hact + aact
                s = sample_summary(total_sim[m].to_numpy(), actual)
                detail_rows.append({
                    "GAME_ID": gid, "GAME_DATE": date, "Scope": "TOTAL", "Team": "TOTAL",
                    "Home": home, "Away": away, "Market": m, "Actual": actual,
                    "Prior_games_home": hp, "Prior_games_away": ap,
                    "Projected_possessions": pack["pace"]["central"], **s,
                })
                prob_rows.extend(pseudo_line_rows(total_sim[m].to_numpy(), actual, m, "TOTAL", gid, None))

            for m in ["3PM","3PA","2PM","2PA","FTM","FTA","REB","OREB","DREB","AST","STL","BLK","TOV","PF"]:
                hv = hs[m].to_numpy(); av = as_[m].to_numpy()
                p_home = float(np.mean(hv > av)); p_away = float(np.mean(av > hv)); p_tie = float(np.mean(hv == av))
                ha = _actual_stat(hr, m)
                aa = _actual_stat(ar, m)
                most_rows.append({
                    "GAME_ID": gid, "GAME_DATE": date, "Home": home, "Away": away, "Market": m,
                    "P_Home": p_home, "P_Away": p_away, "P_Tie": p_tie,
                    "Y_Home": float(ha > aa), "Y_Away": float(aa > ha), "Y_Tie": float(ha == aa),
                })
        except Exception as exc:
            failures.append({"GAME_ID": gid, "GAME_DATE": date, "Error": repr(exc)})

        if progress_callback is not None:
            progress_callback(i + 1, len(games))

    return BacktestResult(
        detail=pd.DataFrame(detail_rows),
        probability_rows=pd.DataFrame(prob_rows),
        most_rows=pd.DataFrame(most_rows),
        failures=pd.DataFrame(failures),
        runtime_seconds=float(time.time() - t0),
    )
