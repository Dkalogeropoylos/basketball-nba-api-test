import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from core.matchup import player_h2h_modifiers, fit_player_h2h_residual_calibration
from core.player_model import PlayerContext, simulate_player

# ------------------------------------------------------------------
# 1) A weak/losing holdout must shrink H2H continuously, not hard-delete it.
# Full EB residual: (16 + 10) / (12 + 10) = 1.1818. With predictive weight
# 0.25 the applied modifier must sit strictly between neutral and full EB.
# ------------------------------------------------------------------
log = pd.DataFrame([
    {'GAME_ID':'h1','GAME_DATE':'2026-01-01','OPP_ABBR':'BBB','MIN':30,'FGA':10,'FG3A':3,'FTA':4,'REB':4,'AST':8},
    {'GAME_ID':'h2','GAME_DATE':'2026-02-01','OPP_ABBR':'BBB','MIN':30,'FGA':10,'FG3A':3,'FTA':4,'REB':4,'AST':8},
])
profile = {'two_pa_pm':7/30,'three_pa_pm':3/30,'fta_pm':4/30,'reb_pm':4/30,'ast_pm':.20,
           'three_pct':.35,'two_pct':.50,'ft_pct':.80}
rh = pd.DataFrame([
    {'GAME_ID':'h1','MIN':30,
     '2PA_observed':7,'2PA_expected_events':7,
     '3PA_observed':3,'3PA_expected_events':3,
     'FTA_observed':4,'FTA_expected_events':3,
     'REB_observed':4,'REB_expected_events':4,
     'AST_observed':8,'AST_expected_events':6},
    {'GAME_ID':'h2','MIN':30,
     '2PA_observed':7,'2PA_expected_events':7,
     '3PA_observed':3,'3PA_expected_events':3,
     'FTA_observed':4,'FTA_expected_events':3,
     'REB_observed':4,'REB_expected_events':4,
     'AST_observed':8,'AST_expected_events':6},
])
cal = {st:{'active':True,'prior_events_k':10.0,'minute_tau':np.inf,'predictive_model_weight':0.25}
       for st in ('2PA','3PA','FTA','REB','AST')}
mods,aud = player_h2h_modifiers(
    log,'BBB',profile,31.5,rotation_similarity=1.0,
    residual_history=rh,residual_calibration_by_stat=cal,
    current_opponent_modifiers={'2PA':1,'3PA':1,'FTA':1,'REB':1,'AST':1},
)
full_ast=(16+10)/(12+10)
assert 1.0 < mods['AST'] < full_ast, (mods, aud)
assert 1.0 < mods['FTA'] < (8+10)/(6+10), (mods, aud)
r=aud[aud['Stat'].eq('AST')].iloc[0]
assert abs(float(r['Predictive H2H model weight'])-.25)<1e-12
assert float(r['Applied H2H weight']) > 0
assert abs(float(r['Full EB residual modifier before model averaging'])-full_ast)<1e-12

# ------------------------------------------------------------------
# 2) FTA H2H is wired into the player simulator rather than being audit-only.
# ------------------------------------------------------------------
ctx0=PlayerContext(projected_minutes=30,minutes_sd=0,pace_multiplier=1,
                   h2h_fta=1.0)
ctx1=PlayerContext(projected_minutes=30,minutes_sd=0,pace_multiplier=1,
                   h2h_fta=1.20)
s0=simulate_player(profile,ctx0,n=120000,seed=2182)
s1=simulate_player(profile,ctx1,n=120000,seed=2182)
assert s1['FTA'].mean() > s0['FTA'].mean()*1.15, (s0['FTA'].mean(),s1['FTA'].mean())

# ------------------------------------------------------------------
# 3) End-to-end calibration exposes a finite candidate plus a continuous
# predictive model weight for all primitive opportunity stats, including FTA.
# ------------------------------------------------------------------
rng=np.random.default_rng(2182)
teams=['A','B','C','D']
player_rows=[]; team_rows=[]; gid=0
for d in range(36):
    pairs=[('A','B'),('C','D')] if d%2==0 else [('A','C'),('B','D')]
    date=pd.Timestamp('2026-03-01')+pd.Timedelta(days=d)
    for a,b in pairs:
        for team,opp in ((a,b),(b,a)):
            team_rows.append({'GAME_DATE':date,'GAME_ID':str(gid),'TEAM_ABBR':team,'OPP_ABBR':opp,
                'FGA':70,'FGM':31,'FG3A':25,'FG3M':9,'FTA':20,'OREB':8,'TOV':13,
                'AST':21,'REB':36,'PTS':82,'PF':18,'BLK':4})
            for j in range(5):
                mins=27.0+j*.8
                a3=int(rng.poisson(mins*(.07+.01*j)))
                a2=int(rng.poisson(mins*(.14+.015*j)))
                fta=int(rng.poisson(mins*(.05+.008*j)))
                ast=int(rng.poisson(mins*(.08+.012*j)))
                player_rows.append({'GAME_DATE':date,'GAME_ID':str(gid),'TEAM_ABBR':team,'OPP_ABBR':opp,
                    'PLAYER_ID':f'{team}{j}','PLAYER_NAME':f'{team}{j}','POSITION_GROUP':['G','G','F','F','C'][j],
                    'MIN':mins,'FGA':a2+a3,'FGM':max(int((a2+a3)*.45),0),'FG3A':a3,'FG3M':max(int(a3*.34),0),
                    'FTA':fta,'FTM':max(int(fta*.80),0),'REB':int(rng.poisson(mins*.14)),'AST':ast,'PTS':12,'OREB':1,'TOV':2})
        gid += 1
cal_fit,cal_audit=fit_player_h2h_residual_calibration(pd.DataFrame(player_rows),pd.DataFrame(team_rows))
assert {'Predictive H2H model weight','Held-out NLL gain'}.issubset(cal_audit.columns)
for st in ('2PA','3PA','FTA','REB','AST'):
    assert st in cal_fit
    if cal_fit[st]['active']:
        assert np.isfinite(float(cal_fit[st]['prior_events_k']))
        w=float(cal_fit[st]['predictive_model_weight'])
        assert 0 < w < 1

print('v2.18.2 continuous residual player H2H + FTA wiring smoke: PASS')
