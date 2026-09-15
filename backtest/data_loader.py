from __future__ import annotations

import numpy as np
import pandas as pd

from providers.sportsdataverse_nba import SportsDataverseNBA


def _finalize_normalized(player: pd.DataFrame, team: pd.DataFrame):
    player = player.copy(); team = team.copy()
    if "SEASON_TYPE" in player.columns:
        reg = player[player["SEASON_TYPE"] == 2].copy()
        if not reg.empty: player = reg
    if "SEASON_TYPE" in team.columns:
        reg = team[team["SEASON_TYPE"] == 2].copy()
        if not reg.empty: team = reg

    nba_ids = set(range(1, 31))
    if "TEAM_ID" in team.columns:
        team = team[pd.to_numeric(team["TEAM_ID"], errors="coerce").isin(nba_ids)].copy()
    if "TEAM_ID" in player.columns:
        player = player[pd.to_numeric(player["TEAM_ID"], errors="coerce").isin(nba_ids)].copy()

    counts = team.groupby("TEAM_ABBR")["GAME_ID"].nunique()
    over82 = counts[counts > 82].index.astype(str).tolist()
    if len(over82) == 2:
        pair = team[team["TEAM_ABBR"].astype(str).isin(over82) & team["OPP_ABBR"].astype(str).isin(over82)].copy()
        pair_dates = pair.groupby("GAME_ID")["GAME_DATE"].min().sort_values()
        for gid in pair_dates.index.astype(str):
            test = team[team["GAME_ID"].astype(str) != gid]
            tc = test.groupby("TEAM_ABBR")["GAME_ID"].nunique()
            if tc.get(over82[0], 0) == 82 and tc.get(over82[1], 0) == 82:
                team = test.copy(); player = player[player["GAME_ID"].astype(str) != gid].copy(); break

    pm = player[["GAME_ID","TEAM_ABBR","MIN"]].copy()
    pm["GAME_ID"] = pm["GAME_ID"].astype(str)
    pm["_MIN"] = pd.to_numeric(pm["MIN"], errors="coerce").fillna(0.0)
    team_len = (pm.groupby(["GAME_ID","TEAM_ABBR"])["_MIN"].sum()/5.0).reset_index(name="_LEN")
    game_len = team_len.groupby("GAME_ID")["_LEN"].median()
    ot = np.maximum(np.rint((game_len-48.0)/5.0), 0).astype(int)
    length = 48.0 + 5.0*ot
    lmap = length.to_dict(); omap = pd.Series(ot, index=game_len.index).to_dict()
    for df in (player, team):
        gid = df["GAME_ID"].astype(str)
        df["GAME_LENGTH_MIN"] = gid.map(lmap).fillna(48.0).astype(float)
        df["OT_COUNT"] = gid.map(omap).fillna(0).astype(int)
        df["OT_FLAG"] = df["OT_COUNT"] > 0
    return player.sort_values(["GAME_DATE","GAME_ID"]).reset_index(drop=True), team.sort_values(["GAME_DATE","GAME_ID","TEAM_ABBR"]).reset_index(drop=True)


def load_local_csvs(team_csv: str, player_csv: str):
    p = SportsDataverseNBA()
    traw = pd.read_csv(team_csv)
    praw = pd.read_csv(player_csv)
    team = p.normalize_team_box(traw)
    player = p.normalize_player_box(praw)
    player, team = _finalize_normalized(player, team)
    return {"team": team, "player": player, "sources": {"team_box": team_csv, "player_box": player_csv}}


def season_label(season: int) -> str:
    return f"{int(season)-1}-{str(int(season))[-2:]}"
