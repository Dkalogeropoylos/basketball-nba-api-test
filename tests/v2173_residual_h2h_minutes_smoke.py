import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from core.matchup import player_h2h_modifiers
from core.minutes_engine import resolve_minutes_sd

# ------------------------------------------------------------------
# 1) Residualized H2H: if observed H2H equals the historical no-H2H
# expectation, the explicit H2H layer must be neutral even when the
# generic opponent environment is positive.
# ------------------------------------------------------------------
log = pd.DataFrame([
    {'GAME_ID':'h1','GAME_DATE':'2026-01-01','OPP_ABBR':'BBB','MIN':30,'FGA':10,'FG3A':3,'REB':4,'AST':6},
    {'GAME_ID':'h2','GAME_DATE':'2026-02-01','OPP_ABBR':'BBB','MIN':30,'FGA':10,'FG3A':3,'REB':4,'AST':6},
])
profile = {'two_pa_pm':7/30, 'three_pa_pm':3/30, 'reb_pm':4/30, 'ast_pm':0.20}
# Historical no-H2H AST expectation was 0.20 * 1.10 = 0.22/min => 6.6/game.
# Use fractional expected events; observed total 12 vs expected 13.2, so the
# residual should be slightly below 1, not a second +10% opponent boost.
rh = pd.DataFrame([
    {'GAME_ID':'h1','MIN':30,'AST_observed':6.6,'AST_expected_events':6.6,
     'REB_observed':4,'REB_expected_events':4,'2PA_observed':7,'2PA_expected_events':7,
     '3PA_observed':3,'3PA_expected_events':3},
    {'GAME_ID':'h2','MIN':30,'AST_observed':6.6,'AST_expected_events':6.6,
     'REB_observed':4,'REB_expected_events':4,'2PA_observed':7,'2PA_expected_events':7,
     '3PA_observed':3,'3PA_expected_events':3},
])
cal = {st:{'active':True,'prior_events_k':10.0,'minute_tau':np.inf} for st in ('2PA','3PA','REB','AST')}
mods,aud = player_h2h_modifiers(
    log,'BBB',profile,31.5,rotation_similarity=1.0,
    residual_history=rh,residual_calibration_by_stat=cal,
    current_opponent_modifiers={'AST':1.10,'REB':1.05,'2PA':1.03,'3PA':1.02},
)
assert abs(mods['AST'] - 1.0) < 1e-9, (mods, aud)
row = aud[aud['Stat'].eq('AST')].iloc[0]
assert abs(float(row['Current no-H2H expected rate/min']) - 0.22) < 1e-9
assert np.isinf(float(row['Learned minute relevance tau']))

# A genuine pair residual should survive, but only as the residual itself.
rh2 = rh.copy()
rh2['AST_observed'] = [8.0, 8.0]
mods2,aud2 = player_h2h_modifiers(
    log,'BBB',profile,31.5,rotation_similarity=1.0,
    residual_history=rh2,residual_calibration_by_stat=cal,
    current_opponent_modifiers={'AST':1.10},
)
assert 1.0 < mods2['AST'] < (16.0/13.2), mods2

# ------------------------------------------------------------------
# 2) Minute relevance tau is learned/configurable: tau=inf means no
# minute-distance penalty; finite tau reduces effective evidence.
# ------------------------------------------------------------------
rh3 = pd.DataFrame([
    {'GAME_ID':'h1','MIN':27.0,'AST_observed':8.0,'AST_expected_events':6.0,
     'REB_observed':4,'REB_expected_events':4,'2PA_observed':7,'2PA_expected_events':7,'3PA_observed':3,'3PA_expected_events':3},
])
cal_inf = {st:{'active':True,'prior_events_k':10.0,'minute_tau':np.inf} for st in ('2PA','3PA','REB','AST')}
cal_10 = {st:{'active':True,'prior_events_k':10.0,'minute_tau':10.0} for st in ('2PA','3PA','REB','AST')}
_,a_inf = player_h2h_modifiers(log.iloc[:1],'BBB',profile,31.5,1.0,
    residual_history=rh3,residual_calibration_by_stat=cal_inf,current_opponent_modifiers={'AST':1.0})
_,a_10 = player_h2h_modifiers(log.iloc[:1],'BBB',profile,31.5,1.0,
    residual_history=rh3,residual_calibration_by_stat=cal_10,current_opponent_modifiers={'AST':1.0})
w_inf=float(a_inf.loc[a_inf['Stat'].eq('AST'),'Posterior H2H weight'].iloc[0])
w_10=float(a_10.loc[a_10['Stat'].eq('AST'),'Posterior H2H weight'].iloc[0])
assert w_inf > w_10, (w_inf,w_10)

# ------------------------------------------------------------------
# 3) Trader minute override changes the central mean only; Monte Carlo SD
# remains the historical role-conditioned SD rather than being hard-capped 2.25.
# ------------------------------------------------------------------
assert abs(resolve_minutes_sd(3.10, ratio=1.40, source='TRADER') - 3.10) < 1e-9
assert abs(resolve_minutes_sd(3.10, ratio=1.40, source='METADATA') - 3.10) < 1e-9
assert resolve_minutes_sd(3.10, ratio=1.40, source='AUTO') > 3.10
assert resolve_minutes_sd(3.10, ratio=1.40, source='OUT') == 0.0

print('v2.17.3 residual H2H + conditional minutes SD smoke: PASS')
print(aud2.round(4).to_string(index=False))

# ------------------------------------------------------------------
# 4) The new walk-forward residual calibration itself runs end-to-end and
# exposes blocked held-out validation plus a data-selected tau (possibly inf).
# ------------------------------------------------------------------
from core.matchup import fit_player_h2h_residual_calibration
rng = np.random.default_rng(11)
teams = ['A','B','C','D']
player_rows=[]; team_rows=[]; gid=0
for d in range(24):
    pairs = [('A','B'),('C','D')] if d % 2 == 0 else [('A','C'),('B','D')]
    date = pd.Timestamp('2026-03-01') + pd.Timedelta(days=d)
    for a,b in pairs:
        for team,opp in ((a,b),(b,a)):
            fga=70; a3=25; fgm=31
            team_rows.append({'GAME_DATE':date,'GAME_ID':str(gid),'TEAM_ABBR':team,'OPP_ABBR':opp,
                'FGA':fga,'FGM':fgm,'FG3A':a3,'FG3M':9,'FTA':19,'OREB':8,'TOV':13,
                'AST':21,'REB':36,'PTS':82,'PF':18,'BLK':4})
            for j in range(4):
                mins=28.0+j*.5
                ast=int(rng.poisson(mins*(.10+.015*j)))
                a3p=int(rng.poisson(mins*.09)); a2p=int(rng.poisson(mins*.18))
                player_rows.append({'GAME_DATE':date,'GAME_ID':str(gid),'TEAM_ABBR':team,'OPP_ABBR':opp,
                    'PLAYER_ID':f'{team}{j}','PLAYER_NAME':f'{team}{j}','POSITION_GROUP':['G','G','F','C'][j],
                    'MIN':mins,'FGA':a2p+a3p,'FGM':max(int((a2p+a3p)*.45),0),'FG3A':a3p,'FG3M':max(int(a3p*.34),0),
                    'FTA':2,'FTM':2,'REB':int(rng.poisson(mins*.15)),'AST':ast,'PTS':12,'OREB':1,'TOV':2})
        gid += 1
cal_fit, cal_audit = fit_player_h2h_residual_calibration(pd.DataFrame(player_rows), pd.DataFrame(team_rows))
assert {'Train events','Holdout events','Held-out NLL gain','Minute relevance tau'}.issubset(cal_audit.columns)
for st in ('2PA','3PA','REB','AST'):
    assert st in cal_fit
    tau=float(cal_fit[st]['minute_tau'])
    assert np.isinf(tau) or tau >= 4.0

print('v2.17.3 residual calibration end-to-end: PASS')
