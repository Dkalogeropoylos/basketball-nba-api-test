import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from core.buckets import WeightConfig
from core.structural_calibration import fit_structural_rate_models, predict_structural_modifiers
from core.team_model import h2h_profile_blend

rng = np.random.default_rng(17)
teams = [f"T{i}" for i in range(10)]
off = {t: rng.normal(0, 0.12) for t in teams}
def3 = {t: rng.normal(0, 0.20) for t in teams}
def_tov = {t: rng.normal(0, 0.16) for t in teams}
def_fta = {t: rng.normal(0, 0.18) for t in teams}
def_ast = {t: rng.normal(0, 0.15) for t in teams}
rows=[]
gid=0
start=pd.Timestamp('2026-05-01')
# 45 rounds -> enough walk-forward rows.
for r in range(45):
    perm = teams[r%10:] + teams[:r%10]
    # fixed pair pattern, rotated each round
    pairs=[(perm[i],perm[i+1]) for i in range(0,10,2)]
    date=start+pd.Timedelta(days=r)
    for a,b in pairs:
        gid += 1
        for team,opp in [(a,b),(b,a)]:
            poss=80.0
            tov_p=1/(1+np.exp(-(-1.80 + 0.45*off[team] + 0.90*def_tov[opp] + rng.normal(0,0.05))))
            fta_pp=np.exp(-1.45 + 0.35*off[team] + 0.75*def_fta[opp] + rng.normal(0,0.04))
            share=1/(1+np.exp(-(-0.55 + 0.55*off[team] + 1.10*def3[opp] + rng.normal(0,0.05))))
            astpm=1/(1+np.exp(-(0.35 + 0.50*off[team] + 0.90*def_ast[opp] + rng.normal(0,0.04))))
            tov=max(4,int(round(poss*tov_p)))
            fta=max(2,int(round(poss*fta_pp)))
            # choose FGA so possession identity is roughly coherent
            fga=max(45,int(round(poss - tov - 0.44*fta + 9)))
            fg3a=max(5,min(fga-5,int(round(fga*share))))
            fg3m=int(round(fg3a*0.34))
            fg2a=fga-fg3a
            fg2m=int(round(fg2a*0.51))
            fgm=fg3m+fg2m
            ast=max(1,min(fgm,int(round(fgm*astpm))))
            oreb=9
            dreb=25
            rows.append({
                'GAME_ID':str(gid),'GAME_DATE':date,'TEAM_ID':team,'TEAM_ABBR':team,
                'OPP_ID':opp,'OPP_ABBR':opp,'FGM':fgm,'FGA':fga,'FG3M':fg3m,'FG3A':fg3a,
                'FTM':int(round(fta*0.78)),'FTA':fta,'OREB':oreb,'DREB':dreb,'REB':oreb+dreb,
                'AST':ast,'STL':7,'BLK':4,'TOV':tov,'PF':18,'PTS':2*(fgm-fg3m)+3*fg3m+int(round(fta*0.78)),
            })

df=pd.DataFrame(rows)
models,audit=fit_structural_rate_models(df)
assert len(models)==6
assert audit['Active'].any(), audit
mods,cur=predict_structural_modifiers(df,'T0','T1',models,WeightConfig.stable(),h2h_rotation_similarity=0.8)
assert set(mods)=={'3P_SHARE','FTA','TOV','OREB','DREB','AST'}
assert all(np.isfinite(v) and v>0 for v in mods.values())

# H2H skip must leave the four v2.17 rates untouched.
base={'fga_live':0.9,'three_share':0.35,'fta_pp':0.25,'tov_pp':0.15,'oreb_per_miss':0.25,
      'assist_per_make':0.62,'pf_pp':0.22,'dreb_capture':0.94,'stl_per_opp_tov':0.55,'blk_rate_pp':0.05}
out,aud=h2h_profile_blend(df,'T0','T1',base,rotation_similarity=1.0,
                          skip_features={'three_share','fta_pp','tov_pp','oreb_per_miss','dreb_capture','assist_per_make'})
for k in ['three_share','fta_pp','tov_pp','oreb_per_miss','dreb_capture','assist_per_make']:
    assert abs(out[k]-base[k])<1e-12, (k,out[k],base[k])
print('v2.18 structural compatibility smoke: PASS')
print(audit[['Feature','Active','Rows','Later holdout RMSE','Own-only holdout RMSE']])
print(cur[['Feature','Model active','Applied modifier']])
