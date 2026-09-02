import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from core.buckets import WeightConfig
from core.player_model import build_player_profile
from core.matchup import player_h2h_modifiers
from core.shooting_efficiency import fit_shooting_efficiency_models, predict_shooting_efficiency_modifiers

# ------------------------------------------------------------------
# 1) Current-opponent H2H must be disjoint from player opportunity profile.
# ------------------------------------------------------------------
rows=[]
for i in range(18):
    opp = 'BBB' if i in (14,17) else ('C'+str(i%4))
    ast = 10 if opp == 'BBB' else 4
    rows.append({
        'GAME_ID':str(i), 'GAME_DATE':pd.Timestamp('2026-01-01')+pd.Timedelta(days=i),
        'OPP_ABBR':opp, 'MIN':30, 'FGA':12, 'FGM':5, 'FG3A':4, 'FG3M':1,
        'FTA':3, 'FTM':2, 'REB':5, 'AST':ast, 'PTS':13,
    })
log=pd.DataFrame(rows)
prof_full,_=build_player_profile(log, WeightConfig.stable())
prof_dis,aud=build_player_profile(log, WeightConfig.stable(), exclude_opponent_abbr='BBB')
assert prof_dis['ast_pm'] < prof_full['ast_pm'], (prof_dis['ast_pm'],prof_full['ast_pm'])
assert int(aud['H2H_excluded'].sum()) == 2, aud
mods,ha=player_h2h_modifiers(log,'BBB',prof_dis,30,rotation_similarity=1.0,
                             prior_minutes_by_stat={'AST':120.0,'REB':np.inf,'2PA':np.inf,'3PA':np.inf})
assert mods['AST'] > 1.0, (mods,ha)
assert float(ha.loc[ha['Stat']=='AST','Posterior H2H weight'].iloc[0]) > 0

# ------------------------------------------------------------------
# 2) Attempt-weighted shooting model should learn a real defense effect.
# ------------------------------------------------------------------
rng=np.random.default_rng(7)
teams=[f'T{i}' for i in range(8)]
off3={t:0.30+0.012*i for i,t in enumerate(teams)}
def3={t:0.90+0.03*(i-3.5) for i,t in enumerate(teams)}
off2={t:0.47+0.012*i for i,t in enumerate(teams)}
def2={t:0.92+0.025*(i-3.5) for i,t in enumerate(teams)}
logs=[]
gid=0
for d in range(85):
    order=np.roll(np.array(teams), d%8)
    for j in range(0,8,2):
        a,b=str(order[j]),str(order[j+1])
        for team,opp in ((a,b),(b,a)):
            a3=int(rng.integers(20,31)); a2=int(rng.integers(35,48))
            p3=float(np.clip(off3[team]*def3[opp],0.20,0.48))
            p2=float(np.clip(off2[team]*def2[opp],0.35,0.68))
            m3=int(rng.binomial(a3,p3)); m2=int(rng.binomial(a2,p2))
            logs.append({
                'GAME_DATE':pd.Timestamp('2026-01-01')+pd.Timedelta(days=d),
                'GAME_ID':str(gid), 'TEAM_ABBR':team,'OPP_ABBR':opp,
                'FG3A':a3,'FG3M':m3,'FGA':a3+a2,'FGM':m3+m2,
                'FTA':20,'OREB':8,'TOV':13,'AST':20,'PF':18,
            })
        gid+=1
team_logs=pd.DataFrame(logs)
models,ma=fit_shooting_efficiency_models(team_logs)
assert models['3P_PCT'].active, ma
# Strong defense vs weak offense should produce a directional modifier.
own_profile={'three_pct':off3['T7'],'two_pct':off2['T7']}
mods,pa=predict_shooting_efficiency_modifiers(team_logs,'T7','T0',own_profile,models)
assert np.isfinite(mods['3P_PCT']) and mods['3P_PCT'] > 0
assert not np.isclose(mods['3P_PCT'],1.0), (mods,pa)

print('v2.17.2 disjoint H2H + shooting smoke: PASS')
print(ma.round(4).to_string(index=False))
print(pa.round(4).to_string(index=False))
