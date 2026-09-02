from __future__ import annotations
import requests
import pandas as pd

BASE = "https://api.balldontlie.io/wnba/v1"


class BDLWNBA:
    def __init__(self, api_key: str, timeout: int = 20):
        if not api_key:
            raise ValueError("BALLDONTLIE API key missing.")
        self.api_key=api_key
        self.timeout=timeout

    def _get(self, path, params=None):
        r=requests.get(
            f"{BASE}/{path.lstrip('/')}",
            params=params or {},
            headers={"Authorization":self.api_key},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def _paged(self, path, params=None, max_pages=10):
        params=dict(params or {})
        params.setdefault("per_page",100)
        rows=[]
        cursor=None
        for _ in range(max_pages):
            if cursor is not None:
                params["cursor"]=cursor
            payload=self._get(path,params)
            rows.extend(payload.get("data",[]))
            cursor=payload.get("meta",{}).get("next_cursor")
            if cursor is None:
                break
        return rows


    def teams_catalog(self):
        """Free-tier WNBA team catalog: one request, no pagination."""
        payload = self._get("teams")
        rows = []
        for t in payload.get("data", []):
            rows.append({
                "BDL_TEAM_ID": t.get("id"),
                "FULL_NAME": t.get("full_name"),
                "CITY": t.get("city"),
                "NAME": t.get("name"),
                "BDL_ABBR": t.get("abbreviation"),
            })
        return pd.DataFrame(rows).sort_values("FULL_NAME").reset_index(drop=True)

    def players(self, search=None):
        params={"search":search} if search else {}
        rows=self._paged("players",params,max_pages=8)
        out=[]
        for p in rows:
            t=p.get("team") or {}
            out.append({
                "PLAYER_ID":p.get("id"),
                "PLAYER_NAME":f"{p.get('first_name','')} {p.get('last_name','')}".strip(),
                "TEAM_ID":t.get("id"),
                "TEAM_ABBR":t.get("abbreviation"),
                "POSITION":p.get("position_abbreviation") or p.get("position"),
            })
        return pd.DataFrame(out)

    def games(self, season:int):
        rows=self._paged("games",{"seasons[]":season},max_pages=8)
        return rows

    def player_game_log(self, player_id:int, season:int):
        stats=self._paged("player_stats",{"player_ids[]":player_id,"seasons[]":season},max_pages=5)
        if not stats:
            return pd.DataFrame()

        game_ids=sorted({s.get("game",{}).get("id") for s in stats if s.get("game",{}).get("id") is not None})
        # Fetch season games once; simpler and cache-friendly at app layer.
        games={g["id"]:g for g in self.games(season)}
        rows=[]
        for s in stats:
            player=s.get("player") or {}
            team=s.get("team") or {}
            game=s.get("game") or {}
            gid=game.get("id")
            g=games.get(gid,{})
            ht=g.get("home_team") or {}
            vt=g.get("visitor_team") or {}
            tid=team.get("id")
            if tid==ht.get("id"):
                opp=vt
            elif tid==vt.get("id"):
                opp=ht
            else:
                opp={}
            rows.append({
                "GAME_ID":gid,
                "GAME_DATE":game.get("date") or g.get("date"),
                "PLAYER_ID":player.get("id"),
                "PLAYER_NAME":f"{player.get('first_name','')} {player.get('last_name','')}".strip(),
                "TEAM_ID":tid,
                "TEAM_ABBR":team.get("abbreviation"),
                "OPP_ID":opp.get("id"),
                "OPP_ABBR":opp.get("abbreviation"),
                "MIN":s.get("min"),
                "FGM":s.get("fgm"),"FGA":s.get("fga"),
                "FG3M":s.get("fg3m"),"FG3A":s.get("fg3a"),
                "FTM":s.get("ftm"),"FTA":s.get("fta"),
                "OREB":s.get("oreb",0),"DREB":s.get("dreb",0),"REB":s.get("reb"),
                "AST":s.get("ast"),"STL":s.get("stl"),"BLK":s.get("blk"),
                "TOV":s.get("turnover",s.get("turnovers")),
                "PF":s.get("pf",s.get("fouls")),
                "PTS":s.get("pts"),
            })
        return pd.DataFrame(rows)

    def team_game_log(self, team_id:int, season:int):
        stats=self._paged("team_stats",{"team_ids[]":team_id,"seasons[]":season},max_pages=5)
        if not stats:
            return pd.DataFrame()
        games={g["id"]:g for g in self.games(season)}
        rows=[]
        for s in stats:
            team=s.get("team") or {}
            game=s.get("game") or {}
            gid=game.get("id")
            g=games.get(gid,{})
            ht=g.get("home_team") or {}
            vt=g.get("visitor_team") or {}
            tid=team.get("id")
            opp=vt if tid==ht.get("id") else ht if tid==vt.get("id") else {}
            rows.append({
                "GAME_ID":gid,"GAME_DATE":game.get("date") or g.get("date"),
                "TEAM_ID":tid,"TEAM_ABBR":team.get("abbreviation"),
                "OPP_ID":opp.get("id"),"OPP_ABBR":opp.get("abbreviation"),
                "FGM":s.get("fgm"),"FGA":s.get("fga"),
                "FG3M":s.get("fg3m"),"FG3A":s.get("fg3a"),
                "FTM":s.get("ftm"),"FTA":s.get("fta"),
                "OREB":s.get("oreb"),"DREB":s.get("dreb"),"REB":s.get("reb"),
                "AST":s.get("ast"),"STL":s.get("stl"),"BLK":s.get("blk"),
                "TOV":s.get("turnovers",s.get("turnover")),
                "PF":s.get("fouls",s.get("pf")),
                "PTS":s.get("pts"),
            })
        return pd.DataFrame(rows)

    def player_season_advanced(self, season:int, player_id:int, measure_type="advanced"):
        rows=self._paged(
            "player_season_advanced_stats",
            {"season":season,"player_ids[]":player_id,"measure_type":measure_type,"per_mode":"per_game"},
            max_pages=2
        )
        return rows

    def team_season_advanced(self, season:int, team_id:int, measure_type="advanced"):
        rows=self._paged(
            "team_season_advanced_stats",
            {"season":season,"team_ids[]":team_id,"measure_type":measure_type,"per_mode":"per_game"},
            max_pages=2
        )
        return rows
