import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from core.minutes_engine import project_team_minutes
from core.matchup import team_matchup_modifiers, fit_opponent_elasticities
from core.team_model import h2h_profile_blend, team_location_modifiers

# 1) Opponent effect is explicit/stat-specific and can materially change shot mix,
# while shooting efficiency remains more conservative.
opp_profile = {
    'rates': {
        'league': {'3P_SHARE':0.36,'FTA':0.25,'FGA_LIVE':0.95,'TOV':0.16,
                   'OREB_PER_MISS':0.23,'AST_PER_MAKE':0.62,'PF':0.24,
                   '3P_PCT':0.34,'2P_PCT':0.51},
        'opponent': {'3P_SHARE':0.42,'FTA':0.22,'FGA_LIVE':0.96,'TOV':0.15,
                     'OREB_PER_MISS':0.21,'AST_PER_MAKE':0.65,'PF':0.26,
                     '3P_PCT':0.36,'2P_PCT':0.53,'games':35},
    },
    'modifiers': {'FGA_LIVE':1,'3P_SHARE':1,'FTA':1,'TOV':1,'OREB_PER_MISS':1,
                  'AST_PER_MAKE':1,'PF':1,'BLK':1,'3P_PCT':1,'2P_PCT':1}
}
own = {'three_share':0.30,'fta_pp':0.30,'fga_live':0.94,'tov_pp':0.17,
       'oreb_per_miss':0.25,'assist_per_make':0.60,'pf_pp':0.23,
       'three_pct':0.31,'two_pct':0.50}
elas = {'3P_SHARE':0.60,'FTA':0.45,'FGA_LIVE':0.15,'TOV':0.30,
        'OREB_PER_MISS':0.30,'AST_PER_MAKE':0.25,'PF':0.25,
        '3P_PCT':0.20,'2P_PCT':0.20}
mods = team_matchup_modifiers(opp_profile, own, elas)
assert mods['3P_SHARE'] > 1.05
assert mods['FTA'] < 1.0
assert 1.0 < mods['3P_PCT'] < 1.06

# 2) DNP-aware rotation preserves star minutes; OUT redistribution is role/position aware.
dates = pd.date_range('2026-05-01', periods=12, freq='3D')
players = [
    (1,'STAR_F','F',34),(2,'STAR_G','G',34),(3,'BIG_C','C',30),
    (4,'G2','G',25),(5,'F2','F',22),(6,'C2','C',18),
    (7,'G3','G',15),(8,'F3','F',12),(9,'G4','G',10),
]
team_rows=[]; player_rows=[]
for i,d in enumerate(dates,1):
    gid=str(i)
    team_rows += [
        {'GAME_ID':gid,'GAME_DATE':d,'TEAM_ABBR':'AAA','OPP_ABBR':'BBB','PTS':85,'FGA':66,'FGM':31,'FG3A':24,'FG3M':9,'FTA':22,'FTM':17,'OREB':8,'DREB':25,'REB':33,'AST':21,'STL':7,'BLK':4,'TOV':13,'PF':19,'OT_FLAG':False,'IS_HOME': i%2==0},
        {'GAME_ID':gid,'GAME_DATE':d,'TEAM_ABBR':'BBB','OPP_ABBR':'AAA','PTS':82,'FGA':65,'FGM':30,'FG3A':23,'FG3M':8,'FTA':20,'FTM':15,'OREB':7,'DREB':24,'REB':31,'AST':19,'STL':6,'BLK':3,'TOV':14,'PF':20,'OT_FLAG':False,'IS_HOME': i%2!=0},
    ]
    for pid,name,pos,mins in players:
        player_rows.append({'GAME_ID':gid,'GAME_DATE':d,'TEAM_ABBR':'AAA','OPP_ABBR':'BBB',
            'PLAYER_ID':pid,'PLAYER_NAME':name,'POSITION_GROUP':pos,'POSITION_ABBR':pos,
            'MIN':mins,'STARTER': mins>=30,'PTS':10,'FGA':8,'FGM':4,'FG3A':3,'FG3M':1,
            'FTA':2,'FTM':2,'TOV':1,'OREB':1 if pos!='G' else 0,'DREB':3,'REB':4,
            'AST':2,'PF':2,'STL':1,'BLK':1 if pos=='C' else 0})
team_db=pd.DataFrame(team_rows); player_db=pd.DataFrame(player_rows)
pool=pd.DataFrame([{'PLAYER_ID':pid,'PLAYER_NAME':name,'TEAM_ABBR':'AAA','POSITION_GROUP':pos,'POSITION_ABBR':pos}
                   for pid,name,pos,mins in players])
neutral=project_team_minutes(player_db,team_db,pool,'AAA','AAA',{'injuries':{}})
assert abs(neutral['Projected Min'].sum()-200) < 1e-6
assert float(neutral.loc[neutral['Player']=='STAR_F','Projected Min'].iloc[0]) > 31
out=project_team_minutes(player_db,team_db,pool,'AAA','AAA',{'injuries':{'STAR_F':{'status':'OUT'}}})
assert abs(out['Projected Min'].sum()-200) < 1e-6
assert float(out.loc[out['Player']=='STAR_F','Projected Min'].iloc[0]) == 0
impact=out.attrs['out_redistribution_impact']
assert not impact.empty and 'Top replacements' in impact.columns
matrix=out.attrs['redistribution_matrix_audit']
star=matrix[matrix['Focal']=='STAR_F'].sort_values('Weight', ascending=False)
# A forward OUT should not default to a center-guard equivalence when empirical evidence is uninformative.
assert float(star[star['Replacement']=='F2']['role_prior'].iloc[0]) > float(star[star['Replacement']=='STAR_G']['role_prior'].iloc[0])

# 3) Location is data-driven and small. With identical home/away feature rates it must be neutral.
loc_mod, loc_audit = team_location_modifiers(team_db[team_db.TEAM_ABBR=='AAA'], True, league_team_logs=team_db)
assert abs(loc_mod['3P_SHARE'] - 1.0) < 0.02
assert abs(loc_mod['3P_PCT'] - 1.0) < 0.031

# 4) H2H remains a small disjoint layer; never >10%.
rows=[]
for i in range(5):
    rows.append({'GAME_ID':f'h{i}','GAME_DATE':pd.Timestamp('2026-06-01')+pd.Timedelta(days=i),
                 'TEAM_ABBR':'AAA','OPP_ABBR':'BBB','PTS':85,'FGA':66,'FGM':31,'FG3A':30,'FG3M':10,
                 'FTA':22,'FTM':17,'OREB':8,'DREB':25,'REB':33,'AST':21,'STL':7,'BLK':4,'TOV':13,'PF':19})
h2h_db=pd.DataFrame(rows)
base={'fga_live':0.95,'three_share':0.30,'fta_pp':0.25,'tov_pp':0.16,'oreb_per_miss':0.23,
      'assist_per_make':0.62,'pf_pp':0.24,'dreb_capture':0.90,'stl_per_opp_tov':0.50,
      'blk_rate_pp':0.05}
_, audit=h2h_profile_blend(h2h_db,'AAA','BBB',base,rotation_similarity=1.0)
assert audit['Applied H2H weight'].max() <= 0.1000001

print('v2.14 smoke: PASS')
print('opponent mods', {k:round(v,3) for k,v in mods.items() if k in ['3P_SHARE','FTA','3P_PCT']})
print('neutral star', float(neutral.loc[neutral['Player']=='STAR_F','Projected Min'].iloc[0]))
print('OUT replacements', impact.iloc[0].to_dict())
