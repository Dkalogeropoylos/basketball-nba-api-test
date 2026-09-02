import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from core.buckets import WeightConfig
from core.exposure import regulation_equivalent_factor, regulation_equivalent_minutes_row
from core.pace_engine import _weighted_team_pace
from core.structural_calibration import _single_game_features, fit_structural_rate_models

# ------------------------------------------------------------------
# 1) OT exposure: a 45-minute game with 12.5% more raw events should not
#    masquerade as a faster 40-minute pace observation.
# ------------------------------------------------------------------
rows=[]
base = dict(FGA=70.0, OREB=10.0, TOV=12.0, FTA=18.1818181818)
for i in range(5):
    scale = 1.125 if i == 4 else 1.0
    rows.append({
        'GAME_ID':str(i),'GAME_DATE':pd.Timestamp('2026-01-01')+pd.Timedelta(days=i),
        'TEAM_ABBR':'AAA','OPP_ABBR':'BBB',
        **{k:v*scale for k,v in base.items()},
        'OT_FLAG': i == 4,
        'OT_COUNT': 1 if i == 4 else 0,
        'GAME_LENGTH_MIN': 45.0 if i == 4 else 40.0,
    })
pace_df=pd.DataFrame(rows)
raw_last = pace_df.iloc[-1]['FGA']-pace_df.iloc[-1]['OREB']+pace_df.iloc[-1]['TOV']+.44*pace_df.iloc[-1]['FTA']
assert abs(raw_last-90.0) < 1e-6
assert abs(float(regulation_equivalent_factor(pace_df).iloc[-1]) - 40/45) < 1e-9
assert abs(_weighted_team_pace(pace_df, WeightConfig.stable()) - 80.0) < 1e-6

# Historical rotation minute is normalized, not arbitrarily downweighted.
r = pd.Series({'MIN':36.0,'OT_FLAG':True,'OT_COUNT':1,'GAME_LENGTH_MIN':45.0})
assert abs(regulation_equivalent_minutes_row(r)-32.0) < 1e-9

# ------------------------------------------------------------------
# 2) DREB capture is constructed from opponent missed-FG opportunities.
# ------------------------------------------------------------------
g = pd.DataFrame([
    {'GAME_DATE':'2026-01-01','GAME_ID':'g1','TEAM_ABBR':'AAA','OPP_ABBR':'BBB',
     'FGA':68,'FGM':31,'FG3A':24,'FTA':20,'TOV':13,'AST':21,'OREB':8,'DREB':27},
    {'GAME_DATE':'2026-01-01','GAME_ID':'g1','TEAM_ABBR':'BBB','OPP_ABBR':'AAA',
     'FGA':70,'FGM':30,'FG3A':25,'FTA':18,'TOV':14,'AST':19,'OREB':10,'DREB':25},
])
f=_single_game_features(g)
a=float(f.loc[f['TEAM_ABBR'].eq('AAA'),'DREB_CAPTURE'].iloc[0])
# BBB missed 40 shots and recovered 10: 30 AAA DREB chances; 27/30 = .90.
assert abs(a-.90) < 1e-12

# ------------------------------------------------------------------
# 3) Production structural model exposes DREB_CAPTURE as a calibrated feature.
#    Use the v2.18 synthetic league fixture by building a small repeated league.
# ------------------------------------------------------------------
rng=np.random.default_rng(2181)
teams=[f'T{i}' for i in range(10)]
out=[]; gid=0
for day in range(75):
    rot=teams[day%10:]+teams[:day%10]
    for j in range(0,10,2):
        aa,bb=rot[j],rot[j+1]
        gid += 1
        # Build paired rows so DREB capture is always physically valid.
        vals={}
        for t,o in ((aa,bb),(bb,aa)):
            poss=82
            tov=int(round(13 + rng.normal(0,1)))
            fta=int(round(20 + rng.normal(0,2)))
            fga=int(round(poss - tov - .44*fta + 9))
            a3=max(12,min(fga-12,int(round(fga*(.38+rng.normal(0,.025))))))
            m3=int(round(a3*.34)); m2=int(round((fga-a3)*.51)); fgm=m3+m2
            miss=max(fga-fgm,1)
            oreb=max(2,min(miss-2,int(round(miss*(.24+rng.normal(0,.02))))))
            vals[t]=dict(FGA=fga,FGM=fgm,FG3A=a3,FG3M=m3,FTA=fta,FTM=int(round(fta*.79)),
                         TOV=tov,OREB=oreb,AST=max(1,int(round(fgm*.65))),STL=7,BLK=4,PF=18)
        for t,o in ((aa,bb),(bb,aa)):
            opp=vals[o]
            chances=max(opp['FGA']-opp['FGM']-opp['OREB'],1)
            # Team-specific but noisy capture skill.
            skill=.86 + .015*(int(t[1:])%4)
            dreb=max(1,min(chances,int(round(chances*(skill+rng.normal(0,.015))))))
            v=vals[t]
            out.append({'GAME_DATE':pd.Timestamp('2026-02-01')+pd.Timedelta(days=day),'GAME_ID':str(gid),
                        'TEAM_ABBR':t,'OPP_ABBR':o,'DREB':dreb,'REB':dreb+v['OREB'],
                        'PTS':2*(v['FGM']-v['FG3M'])+3*v['FG3M']+v['FTM'], **v})
league=pd.DataFrame(out)
models,audit=fit_structural_rate_models(league)
assert 'DREB_CAPTURE' in models
assert 'DREB_CAPTURE' in set(audit['Feature'])

print('v2.18.1 OT exposure + DREB residual-H2H smoke: PASS')
