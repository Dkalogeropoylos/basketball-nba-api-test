import pandas as pd
from core.schema import NUMERIC_PLAYER_COLUMNS, NUMERIC_TEAM_COLUMNS


def clean_player_log(df):
    x=df.copy()
    if "GAME_DATE" in x:
        x["GAME_DATE"]=pd.to_datetime(x["GAME_DATE"],errors="coerce")
    for c in NUMERIC_PLAYER_COLUMNS:
        if c in x:
            x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna(subset=["GAME_DATE","MIN"])
    x=x[x.MIN>0].sort_values("GAME_DATE").reset_index(drop=True)
    return x


def clean_team_log(df):
    x=df.copy()
    if "GAME_DATE" in x:
        x["GAME_DATE"]=pd.to_datetime(x["GAME_DATE"],errors="coerce")
    for c in NUMERIC_TEAM_COLUMNS:
        if c in x:
            x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna(subset=["GAME_DATE","FGA"])
    return x.sort_values("GAME_DATE").reset_index(drop=True)
