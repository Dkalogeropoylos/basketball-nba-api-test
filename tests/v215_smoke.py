import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inspect
import numpy as np
import pandas as pd

from core.availability import availability_similarity_weight_maps, confidence_by_stat
from core.redistribution import learn_redistribution_matrix
from core.player_model import simulate_player
from core.team_model import simulate_game

# ------------------------------------------------------------------
# Synthetic team history with two current OUT players: a guard creator
# and a forward rebounder. Each historical game can only receive one
# similarity score per stat.
# ------------------------------------------------------------------
dates = pd.date_range('2026-05-01', periods=8, freq='3D')
team_rows=[]
player_rows=[]
# gid0 establishes both OUT players as roster members.
patterns = {
    '0': {'G_OUT':1,'F_OUT':1},
    '1': {'G_OUT':0,'F_OUT':0},  # exact 2/2
    '2': {'G_OUT':0,'F_OUT':0},  # exact 2/2
    '3': {'G_OUT':0,'F_OUT':1},  # only guard absent
    '4': {'G_OUT':1,'F_OUT':0},  # only forward absent
    '5': {'G_OUT':1,'F_OUT':1},
    '6': {'G_OUT':0,'F_OUT':0},  # exact 2/2
    '7': {'G_OUT':0,'F_OUT':0},  # exact 2/2 => four exact games
}
players = [
    (1,'FOCAL_F','F'), (2,'G_OUT','G'), (3,'F_OUT','F'), (4,'G_ACTIVE','G'), (5,'C_ACTIVE','C')
]
for i,d in enumerate(dates):
    gid=str(i)
    team_rows.append({'GAME_ID':gid,'GAME_DATE':d,'TEAM_ABBR':'AAA','OPP_ABBR':'BBB',
                      'PTS':84,'FGA':66,'FGM':30,'FG3A':24,'FG3M':9,'FTA':20,'FTM':15,
                      'OREB':8,'DREB':25,'REB':33,'AST':20,'STL':7,'BLK':4,'TOV':13,'PF':19})
    for pid,name,pos in players:
        if name in patterns[gid] and patterns[gid][name] == 0:
            continue
        # Make the guard the creation-volume absence and forward the rebounding-volume absence.
        ast = 8 if name=='G_OUT' else (1 if name=='F_OUT' else 2)
        reb = 10 if name=='F_OUT' else (2 if name=='G_OUT' else 5)
        mins = 30 if name in {'G_OUT','F_OUT','FOCAL_F'} else 24
        player_rows.append({'GAME_ID':gid,'GAME_DATE':d,'TEAM_ABBR':'AAA','OPP_ABBR':'BBB',
            'PLAYER_ID':pid,'PLAYER_NAME':name,'POSITION_GROUP':pos,'POSITION_ABBR':pos,
            'MIN':mins,'PTS':10,'FGA':8,'FGM':4,'FG3A':3 if pos!='C' else 0,'FG3M':1,
            'FTA':2,'FTM':2,'REB':reb,'OREB':2 if pos!='G' else 0,'DREB':max(reb-2,0),
            'AST':ast,'TOV':1,'PF':2,'STL':1,'BLK':1 if pos=='C' else 0})

team_log=pd.DataFrame(team_rows)
player_db=pd.DataFrame(player_rows)
pool=pd.DataFrame([{'PLAYER_ID':pid,'PLAYER_NAME':name,'TEAM_ABBR':'AAA','POSITION_GROUP':pos,'POSITION_ABBR':pos}
                   for pid,name,pos in players])

maps,audit,scores = availability_similarity_weight_maps(
    player_db, team_log, 'AAA', ['G_OUT','F_OUT'], ['AST','REB'],
    current_pool=pool, focal_player='FOCAL_F', k=6.0, maturity_games=5.0,
)

# Single-score architecture: exactly one score per game/stat; no nested groups.
assert set(scores['AST']) == set(team_log.GAME_ID.astype(str))
assert set(scores['REB']) == set(team_log.GAME_ID.astype(str))
assert len(scores['AST']) == len(team_log)
assert all(0.0 <= x <= 1.0 for x in scores['AST'].values())

# Both absent is most similar; stat relevance changes which 1/2 state is closer.
assert abs(scores['AST']['1'] - 1.0) < 1e-9
assert abs(scores['REB']['1'] - 1.0) < 1e-9
# Guard absence matters more for focal-F assists; forward absence matters more for rebounds.
assert scores['AST']['3'] > scores['AST']['4']
assert scores['REB']['4'] > scores['REB']['3']

conf = confidence_by_stat(audit)
# Four exact games are useful but not mature/full-strength. They are NOT zeroed.
assert 0.0 < conf['AST'] < 0.5
assert 0.0 < conf['REB'] < 0.5
# Normalized inner weights preserve outer buckets by having mean ~1 over the state frame.
assert abs(np.mean(list(maps['AST'].values())) - 1.0) < 1e-8
assert abs(np.mean(list(maps['REB'].values())) - 1.0) < 1e-8

# ------------------------------------------------------------------
# Four common games in the replacement matrix now contribute with heavy
# shrinkage instead of empirical=0. Five remains the maturity point.
# ------------------------------------------------------------------
short_team = team_log.iloc[:4].copy()
short_pdb = player_db[player_db.GAME_ID.astype(str).isin(short_team.GAME_ID.astype(str))].copy()
mat,mat_audit = learn_redistribution_matrix(short_pdb, short_team, pool, 'AAA')
row = mat_audit[(mat_audit.Focal=='FOCAL_F') & (mat_audit.Replacement=='G_ACTIVE')].iloc[0]
assert int(row['games']) == 4
assert 0.0 < float(row['confidence']) < 0.25
assert abs(float(row['maturity']) - 0.8) < 1e-9

# Defaults are now 50k for both player and coupled-team simulation engines.
assert inspect.signature(simulate_player).parameters['n'].default == 50_000
assert inspect.signature(simulate_game).parameters['n'].default == 50_000

print('v2.15 smoke: PASS')
print(audit[['Stat','Evidence mass','Maturity','State confidence','Relevance']].to_string(index=False))
print('4-game replacement confidence', float(row['confidence']))
