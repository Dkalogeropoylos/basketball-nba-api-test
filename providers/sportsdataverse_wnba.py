from __future__ import annotations

from io import BytesIO
from typing import Dict, Tuple
import requests
import numpy as np
import pandas as pd


BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"

DATASETS = {
    "player_box": {
        "tag": "espn_wnba_player_boxscores",
        "stem": "player_box_{season}",
    },
    "team_box": {
        "tag": "espn_wnba_team_boxscores",
        "stem": "team_box_{season}",
    },
}


class SportsDataverseWNBA:
    """
    WNBA historical provider backed by SportsDataverse GitHub release assets.

    Core model needs only two season files:
      - ESPN WNBA player boxscores
      - ESPN WNBA team boxscores

    No stats.nba.com request is required.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "BasketballPricingEngine/2.4 (+Streamlit)"
        })

    def _asset_urls(self, dataset: str, season: int):
        info = DATASETS[dataset]
        stem = info["stem"].format(season=int(season))
        tag = info["tag"]
        return [
            f"{BASE}/{tag}/{stem}.parquet",
            f"{BASE}/{tag}/{stem}.csv",
        ]

    def _download_table(self, dataset: str, season: int) -> Tuple[pd.DataFrame, str]:
        errors = []
        for url in self._asset_urls(dataset, season):
            try:
                r = self.session.get(url, timeout=self.timeout, allow_redirects=True)
                r.raise_for_status()
                if url.endswith(".parquet"):
                    df = pd.read_parquet(BytesIO(r.content))
                else:
                    df = pd.read_csv(BytesIO(r.content))
                return df, url
            except Exception as exc:
                errors.append(f"{url}: {exc}")
        raise RuntimeError(
            f"Could not load SportsDataverse {dataset} for {season}. "
            + " | ".join(errors)
        )

    def load_season(self, season: int) -> Dict[str, object]:
        player_raw, player_url = self._download_table("player_box", season)
        team_raw, team_url = self._download_table("team_box", season)

        player = self.normalize_player_box(player_raw)
        team = self.normalize_team_box(team_raw)

        # Model regular season only by default.
        if "SEASON_TYPE" in player.columns:
            reg = player[player["SEASON_TYPE"] == 2].copy()
            if not reg.empty:
                player = reg
        if "SEASON_TYPE" in team.columns:
            reg = team[team["SEASON_TYPE"] == 2].copy()
            if not reg.empty:
                team = reg

        # Derive scheduled game length from the player-minute accounting rather
        # than assuming OT only when one individual exceeds 40 minutes. Team
        # player minutes sum to 200 in regulation, 225 in 1OT, 250 in 2OT, etc.
        # Snapping to 5-minute increments is robust to small source rounding.
        _pm = player[["GAME_ID", "TEAM_ABBR", "MIN"]].copy()
        _pm["GAME_ID"] = _pm["GAME_ID"].astype(str)
        _pm["_MIN"] = pd.to_numeric(_pm["MIN"], errors="coerce").fillna(0.0)
        _team_len = (_pm.groupby(["GAME_ID", "TEAM_ABBR"])["_MIN"].sum() / 5.0).reset_index(name="_LEN")
        _game_len = _team_len.groupby("GAME_ID")["_LEN"].median()
        _ot_count = np.maximum(np.rint((_game_len - 40.0) / 5.0), 0).astype(int)
        _game_length = 40.0 + 5.0 * _ot_count
        _length_map = _game_length.to_dict()
        _ot_map = pd.Series(_ot_count, index=_game_len.index).to_dict()

        # Conservative fallback for malformed minute totals: the old >40 test
        # can still flag a one-OT game when the aggregate length is unavailable.
        _legacy_ot = set(
            player.loc[pd.to_numeric(player["MIN"], errors="coerce") > 40, "GAME_ID"].astype(str)
        )
        for _gid in _legacy_ot:
            if float(_length_map.get(_gid, 40.0)) <= 40.0:
                _length_map[_gid] = 45.0
                _ot_map[_gid] = 1

        for _df in (player, team):
            _gid = _df["GAME_ID"].astype(str)
            _df["GAME_LENGTH_MIN"] = _gid.map(_length_map).fillna(40.0).astype(float)
            _df["OT_COUNT"] = _gid.map(_ot_map).fillna(0).astype(int)
            _df["OT_FLAG"] = _df["OT_COUNT"] > 0

        return {
            "player": player,
            "team": team,
            "sources": {
                "player_box": player_url,
                "team_box": team_url,
            }
        }

    @staticmethod
    def _rename_existing(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        actual = {k: v for k, v in mapping.items() if k in df.columns}
        return df.rename(columns=actual)

    def normalize_player_box(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()

        mapping = {
            "game_id": "GAME_ID",
            "season": "SEASON",
            "season_type": "SEASON_TYPE",
            "game_date": "GAME_DATE",
            "athlete_id": "PLAYER_ID",
            "athlete_display_name": "PLAYER_NAME",
            "team_id": "TEAM_ID",
            "team_abbreviation": "TEAM_ABBR",
            "opponent_team_id": "OPP_ID",
            "opponent_team_abbreviation": "OPP_ABBR",
            "minutes": "MIN",
            "field_goals_made": "FGM",
            "field_goals_attempted": "FGA",
            "three_point_field_goals_made": "FG3M",
            "three_point_field_goals_attempted": "FG3A",
            "free_throws_made": "FTM",
            "free_throws_attempted": "FTA",
            "offensive_rebounds": "OREB",
            "defensive_rebounds": "DREB",
            "rebounds": "REB",
            "assists": "AST",
            "steals": "STL",
            "blocks": "BLK",
            "turnovers": "TOV",
            "fouls": "PF",
            "points": "PTS",
            "starter": "STARTER",
            "did_not_play": "DNP",
            "active": "ACTIVE",
            "athlete_position_name": "POSITION_NAME",
            "athlete_position_abbreviation": "POSITION_ABBR",
            "team_display_name": "TEAM_NAME",
            "opponent_team_display_name": "OPP_NAME",
            "home_away": "HOME_AWAY",
        }
        x = self._rename_existing(x, mapping)

        required_numeric = [
            "MIN","FGM","FGA","FG3M","FG3A","FTM","FTA",
            "OREB","DREB","REB","AST","STL","BLK","TOV","PF","PTS"
        ]
        for c in required_numeric:
            if c not in x.columns:
                x[c] = 0.0
            x[c] = pd.to_numeric(x[c], errors="coerce")

        if "GAME_DATE" in x.columns:
            x["GAME_DATE"] = pd.to_datetime(x["GAME_DATE"], errors="coerce")

        # Keep actual participants. DNP rows are useful for roster history but
        # must not enter rate samples.
        x = x[x["MIN"].fillna(0) > 0].copy()

        # Broad position for automatic matchup aggregation.
        x["POSITION_GROUP"] = x.get(
            "POSITION_ABBR",
            pd.Series(index=x.index, dtype="object")
        ).map(self.position_group)

        return x.sort_values(["GAME_DATE","GAME_ID","PLAYER_NAME"]).reset_index(drop=True)

    def normalize_team_box(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        mapping = {
            "game_id": "GAME_ID",
            "season": "SEASON",
            "season_type": "SEASON_TYPE",
            "game_date": "GAME_DATE",
            "team_id": "TEAM_ID",
            "team_abbreviation": "TEAM_ABBR",
            "team_display_name": "TEAM_NAME",
            "opponent_team_id": "OPP_ID",
            "opponent_team_abbreviation": "OPP_ABBR",
            "opponent_team_display_name": "OPP_NAME",
            "field_goals_made": "FGM",
            "field_goals_attempted": "FGA",
            "three_point_field_goals_made": "FG3M",
            "three_point_field_goals_attempted": "FG3A",
            "free_throws_made": "FTM",
            "free_throws_attempted": "FTA",
            "offensive_rebounds": "OREB",
            "defensive_rebounds": "DREB",
            "total_rebounds": "REB",
            "assists": "AST",
            "steals": "STL",
            "blocks": "BLK",
            "turnovers": "TOV",
            "total_turnovers": "TOTAL_TOV",
            "fouls": "PF",
            "team_score": "PTS",
            "team_home_away": "HOME_AWAY",
        }
        x = self._rename_existing(x, mapping)

        # Prefer total turnovers if ESPN supplies both individual and total.
        if "TOTAL_TOV" in x.columns:
            x["TOV"] = pd.to_numeric(x["TOTAL_TOV"], errors="coerce").fillna(
                pd.to_numeric(x.get("TOV"), errors="coerce")
            )

        numeric = [
            "FGM","FGA","FG3M","FG3A","FTM","FTA",
            "OREB","DREB","REB","AST","STL","BLK","TOV","PF","PTS"
        ]
        for c in numeric:
            if c not in x.columns:
                x[c] = 0.0
            x[c] = pd.to_numeric(x[c], errors="coerce")

        if "GAME_DATE" in x.columns:
            x["GAME_DATE"] = pd.to_datetime(x["GAME_DATE"], errors="coerce")

        return x.sort_values(["GAME_DATE","GAME_ID","TEAM_ABBR"]).reset_index(drop=True)

    @staticmethod
    def position_group(value):
        u = str(value or "").upper().strip()
        if not u or u == "NAN":
            return None
        # ESPN abbreviations can be PG/SG/G, SF/PF/F, C, G-F/F-C.
        if u in {"PG","SG","G"} or u.startswith("G"):
            return "G"
        if u in {"SF","PF","F"} or u.startswith("F"):
            return "F"
        if u == "C" or u.startswith("C"):
            return "C"
        return None

    @staticmethod
    def teams(team_df: pd.DataFrame) -> pd.DataFrame:
        cols = ["TEAM_ID","TEAM_ABBR","TEAM_NAME"]
        out = team_df[cols].dropna(subset=["TEAM_ID","TEAM_ABBR"]).copy()
        return (
            out.drop_duplicates(subset=["TEAM_ID"])
            .sort_values("TEAM_NAME")
            .reset_index(drop=True)
        )

    @staticmethod
    def current_player_pool(player_df: pd.DataFrame) -> pd.DataFrame:
        """
        Use each player's latest observed game to assign the current team.
        Handles in-season trades better than season-wide first occurrence.
        """
        x = player_df.sort_values("GAME_DATE").copy()
        latest = x.groupby("PLAYER_ID", dropna=False).tail(1)
        cols = [
            "PLAYER_ID","PLAYER_NAME","TEAM_ID","TEAM_ABBR","TEAM_NAME",
            "POSITION_ABBR","POSITION_GROUP"
        ]
        existing = [c for c in cols if c in latest.columns]
        return (
            latest[existing]
            .dropna(subset=["PLAYER_ID","PLAYER_NAME","TEAM_ABBR"])
            .drop_duplicates(subset=["PLAYER_ID"])
            .sort_values("PLAYER_NAME")
            .reset_index(drop=True)
        )

    @staticmethod
    def position_environment(
        player_df: pd.DataFrame,
        opponent_abbr: str,
        position_group: str,
    ) -> Tuple[dict, dict]:
        """
        Aggregate position production per 36 against the selected opponent and
        compare it with the same position's league baseline.

        This uses the same player-box database, so there are no extra API calls.
        """
        x = player_df[
            player_df["POSITION_GROUP"].astype(str) == str(position_group)
        ].copy()

        vs = x[
            x["OPP_ABBR"].astype(str).str.upper()
            == str(opponent_abbr).upper()
        ].copy()

        return (
            SportsDataverseWNBA._per36(vs),
            SportsDataverseWNBA._per36(x),
        )

    @staticmethod
    def _per36(df: pd.DataFrame) -> dict:
        if df is None or df.empty:
            return {}
        mins = float(pd.to_numeric(df["MIN"], errors="coerce").fillna(0).sum())
        if mins <= 0:
            return {}
        out = {
            "sample_min": mins,
            "player_game_rows": int(len(df)),
        }
        for source, key in [
            ("PTS","PTS"),("REB","REB"),("AST","AST"),
            ("FG3A","3PA"),("FTA","FTA"),("FGA","FGA"),
            ("OREB","OREB"),("TOV","TOV")
        ]:
            out[key] = (
                float(pd.to_numeric(df[source], errors="coerce").fillna(0).sum())
                / mins * 36.0
            )
        return out
