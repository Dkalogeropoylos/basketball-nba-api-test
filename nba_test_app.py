
import streamlit as st
import pandas as pd

st.set_page_config(page_title="NBA API Safe Test", page_icon="🏀", layout="wide")
st.title("🏀 NBA API Safe Test")
st.caption("Standalone test only. It does NOT import or modify the WNBA pricing model.")

try:
    from nba_api.stats.endpoints import teamgamelogs, playergamelogs
except Exception as exc:
    st.error("nba_api is not installed. Install dependencies from requirements.txt.")
    st.exception(exc)
    st.stop()

def _opp_from_matchup(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    # NBA strings are usually "BOS vs. NYK" or "BOS @ NYK"
    parts = s.replace("vs.", "@").replace("vs", "@").split("@")
    if len(parts) == 2:
        return parts[1].strip().split()[-1]
    toks = s.split()
    return toks[-1] if toks else None

def normalize_team(df):
    x = df.copy()
    rename = {
        "TEAM_ABBREVIATION": "TEAM_ABBR",
        "FG3A": "FG3A",
        "FG3M": "FG3M",
    }
    x = x.rename(columns=rename)
    if "GAME_DATE" in x:
        x["GAME_DATE"] = pd.to_datetime(x["GAME_DATE"], errors="coerce")
    if "MATCHUP" in x:
        x["OPP_ABBR"] = x["MATCHUP"].map(_opp_from_matchup)
        x["HOME_AWAY"] = x["MATCHUP"].astype(str).apply(lambda s: "A" if "@" in s else "H")
    if "FG3A" in x and "FGA" in x:
        x["2PA"] = pd.to_numeric(x["FGA"], errors="coerce") - pd.to_numeric(x["FG3A"], errors="coerce")
    if "FG3M" in x and "FGM" in x:
        x["2PM"] = pd.to_numeric(x["FGM"], errors="coerce") - pd.to_numeric(x["FG3M"], errors="coerce")
    keep = [c for c in [
        "SEASON_YEAR","TEAM_ID","TEAM_NAME","TEAM_ABBR","GAME_ID","GAME_DATE",
        "MATCHUP","HOME_AWAY","OPP_ABBR","WL","MIN","FGM","FGA","FG3M","FG3A",
        "2PM","2PA","FTM","FTA","OREB","DREB","REB","AST","STL","BLK","TOV","PF","PTS"
    ] if c in x.columns]
    return x[keep].sort_values(["GAME_DATE","GAME_ID","TEAM_ABBR"], na_position="last").reset_index(drop=True)

def normalize_player(df):
    x = df.copy().rename(columns={"TEAM_ABBREVIATION":"TEAM_ABBR"})
    if "GAME_DATE" in x:
        x["GAME_DATE"] = pd.to_datetime(x["GAME_DATE"], errors="coerce")
    if "MATCHUP" in x:
        x["OPP_ABBR"] = x["MATCHUP"].map(_opp_from_matchup)
        x["HOME_AWAY"] = x["MATCHUP"].astype(str).apply(lambda s: "A" if "@" in s else "H")
    if "FG3A" in x and "FGA" in x:
        x["2PA"] = pd.to_numeric(x["FGA"], errors="coerce") - pd.to_numeric(x["FG3A"], errors="coerce")
    if "FG3M" in x and "FGM" in x:
        x["2PM"] = pd.to_numeric(x["FGM"], errors="coerce") - pd.to_numeric(x["FG3M"], errors="coerce")
    keep = [c for c in [
        "SEASON_YEAR","PLAYER_ID","PLAYER_NAME","TEAM_ID","TEAM_NAME","TEAM_ABBR",
        "GAME_ID","GAME_DATE","MATCHUP","HOME_AWAY","OPP_ABBR","WL","MIN",
        "FGM","FGA","FG3M","FG3A","2PM","2PA","FTM","FTA","OREB","DREB","REB",
        "AST","STL","BLK","TOV","PF","PTS"
    ] if c in x.columns]
    return x[keep].sort_values(["GAME_DATE","GAME_ID","TEAM_ABBR","PLAYER_NAME"], na_position="last").reset_index(drop=True)

season = st.text_input("NBA season", value="2025-26")
season_type = st.selectbox("Season type", ["Regular Season", "Playoffs"], index=0)

if st.button("Load NBA API data", type="primary"):
    with st.spinner("Calling NBA Stats API..."):
        try:
            team_raw = teamgamelogs.TeamGameLogs(
                season_nullable=season,
                season_type_nullable=season_type,
                timeout=60,
            ).get_data_frames()[0]

            player_raw = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                season_type_nullable=season_type,
                timeout=60,
            ).get_data_frames()[0]

            team = normalize_team(team_raw)
            player = normalize_player(player_raw)

            st.session_state["nba_team"] = team
            st.session_state["nba_player"] = player
            st.success(f"Loaded {len(team):,} team-game rows and {len(player):,} player-game rows.")
        except Exception as exc:
            st.error(
                "The NBA Stats API call failed. This can happen if stats.nba.com blocks the "
                "IP used by your hosting environment. Nothing in the WNBA model was changed."
            )
            st.exception(exc)

team = st.session_state.get("nba_team")
player = st.session_state.get("nba_player")

if isinstance(team, pd.DataFrame) and not team.empty:
    st.subheader("Team-game data")
    c1,c2,c3 = st.columns(3)
    c1.metric("Rows", f"{len(team):,}")
    c2.metric("Teams", f"{team['TEAM_ABBR'].nunique():,}" if "TEAM_ABBR" in team else "—")
    c3.metric("Through", str(team["GAME_DATE"].max().date()) if "GAME_DATE" in team else "—")
    st.dataframe(team.tail(50), use_container_width=True, hide_index=True)
    st.download_button(
        "Download normalized team CSV",
        team.to_csv(index=False).encode("utf-8"),
        file_name=f"nba_team_games_{season.replace('-','_')}.csv",
        mime="text/csv",
    )

if isinstance(player, pd.DataFrame) and not player.empty:
    st.subheader("Player-game data")
    c1,c2,c3 = st.columns(3)
    c1.metric("Rows", f"{len(player):,}")
    c2.metric("Players", f"{player['PLAYER_ID'].nunique():,}" if "PLAYER_ID" in player else "—")
    c3.metric("Through", str(player["GAME_DATE"].max().date()) if "GAME_DATE" in player else "—")
    st.dataframe(player.tail(50), use_container_width=True, hide_index=True)
    st.download_button(
        "Download normalized player CSV",
        player.to_csv(index=False).encode("utf-8"),
        file_name=f"nba_player_games_{season.replace('-','_')}.csv",
        mime="text/csv",
    )
