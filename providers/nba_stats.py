from __future__ import annotations
import numpy as np
import pandas as pd


class NBAStatsProvider:
    """nba_api adapter for NBA/WNBA basic and matchup data."""
    def __init__(self, league_id="00"):
        self.league_id = str(league_id)

    def players(self, season):
        from nba_api.stats.endpoints import leaguedashplayerstats
        ep = leaguedashplayerstats.LeagueDashPlayerStats(
            season=str(season),
            season_type_all_star="Regular Season",
            league_id_nullable=self.league_id,
            per_mode_detailed="PerGame",
            timeout=8,
        )
        df = ep.get_data_frames()[0].copy()
        out = df.rename(columns={"TEAM_ABBREVIATION": "TEAM_ABBR"})
        keep = [c for c in ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBR"] if c in out.columns]
        return (
            out[keep]
            .dropna(subset=["PLAYER_ID", "PLAYER_NAME"])
            .drop_duplicates(subset=["PLAYER_ID"])
            .sort_values("PLAYER_NAME")
            .reset_index(drop=True)
        )

    def teams(self, season):
        p = self.players(season)
        return (
            p[["TEAM_ID", "TEAM_ABBR"]]
            .dropna()
            .drop_duplicates(subset=["TEAM_ID"])
            .sort_values("TEAM_ABBR")
            .reset_index(drop=True)
        )

    def player_game_log(self, player_id, season):
        from nba_api.stats.endpoints import playergamelog
        ep = playergamelog.PlayerGameLog(
            player_id=int(player_id),
            season=str(season),
            season_type_all_star="Regular Season",
            league_id_nullable=self.league_id,
            timeout=8,
        )
        x = ep.get_data_frames()[0].copy()
        x = x.rename(columns={"Game_ID": "GAME_ID", "Player_ID": "PLAYER_ID"})
        if "MATCHUP" in x.columns:
            x["TEAM_ABBR"] = x["MATCHUP"].astype(str).str.split().str[0]
            x["OPP_ABBR"] = x["MATCHUP"].astype(str).str.extract(
                r"(?:vs\.|@)\s*([A-Z]{2,4})", expand=False
            )
        x["PLAYER_ID"] = int(player_id)
        return x

    def league_team_game_logs(self, season):
        from nba_api.stats.endpoints import leaguegamelog
        ep = leaguegamelog.LeagueGameLog(
            counter=0,
            direction="DESC",
            league_id=self.league_id,
            player_or_team_abbreviation="T",
            season=str(season),
            season_type_all_star="Regular Season",
            sorter="DATE",
            timeout=8,
        )
        x = ep.get_data_frames()[0].copy()
        x = x.rename(columns={"TEAM_ABBREVIATION": "TEAM_ABBR"})
        if "MATCHUP" in x.columns:
            x["OPP_ABBR"] = x["MATCHUP"].astype(str).str.extract(
                r"(?:vs\.|@)\s*([A-Z]{2,4})", expand=False
            )
        return x.reset_index(drop=True)

    def team_game_log(self, team_id, season, league_logs=None):
        x = league_logs.copy() if isinstance(league_logs, pd.DataFrame) else self.league_team_game_logs(season)
        return x[pd.to_numeric(x["TEAM_ID"], errors="coerce") == int(team_id)].copy().reset_index(drop=True)

    def player_position(self, player_id):
        from nba_api.stats.endpoints import commonplayerinfo
        ep = commonplayerinfo.CommonPlayerInfo(
            player_id=int(player_id),
            league_id_nullable=self.league_id,
            timeout=8,
        )
        df = ep.get_data_frames()[0]
        raw = ""
        if not df.empty and "POSITION" in df.columns:
            raw = str(df.iloc[0]["POSITION"] or "")
        u = raw.upper()
        if u.startswith("G") or "GUARD" in u:
            broad = "G"
        elif u.startswith("F") or "FORWARD" in u:
            broad = "F"
        elif u.startswith("C") or "CENTER" in u:
            broad = "C"
        else:
            broad = None
        return {"raw": raw, "broad": broad}

    def position_totals(self, season, position, opponent_team_id=0):
        """
        Aggregate G/F/C production using LeagueDashPlayerStats with
        OpponentTeamID + PlayerPosition, then convert to per-36.
        """
        from nba_api.stats.endpoints import leaguedashplayerstats
        ep = leaguedashplayerstats.LeagueDashPlayerStats(
            season=str(season),
            season_type_all_star="Regular Season",
            league_id_nullable=self.league_id,
            opponent_team_id=int(opponent_team_id or 0),
            player_position_abbreviation_nullable=str(position),
            per_mode_detailed="Totals",
            timeout=8,
        )
        df = ep.get_data_frames()[0].copy()
        if df.empty:
            return {}

        numeric = ["GP", "MIN", "PTS", "REB", "AST", "FG3A", "FTA", "FGA", "OREB", "TOV"]
        for c in numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        total_min = float(df["MIN"].sum()) if "MIN" in df.columns else 0.0
        if total_min <= 0:
            return {}

        out = {
            "sample_min": total_min,
            "player_rows": int(len(df)),
            "gp_sum": float(df["GP"].sum()) if "GP" in df.columns else np.nan,
        }
        for src, key in [
            ("PTS", "PTS"), ("REB", "REB"), ("AST", "AST"),
            ("FG3A", "3PA"), ("FTA", "FTA"), ("FGA", "FGA"),
            ("OREB", "OREB"), ("TOV", "TOV"),
        ]:
            if src in df.columns:
                out[key] = float(df[src].sum()) / total_min * 36.0
        return out
