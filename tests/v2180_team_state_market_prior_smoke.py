import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from core.buckets import WeightConfig
from core.structural_calibration import fit_structural_rate_models, predict_structural_modifiers
from core.market_prior import fit_market_margin_calibration, apply_market_margin_prior

# ------------------------------------------------------------------
# 1) Synthetic league with a genuine non-negative opponent TOV/3P signal.
# The v2.18 model must never learn a negative opponent beta.
# ------------------------------------------------------------------
rng = np.random.default_rng(2180)
teams = [f"T{i}" for i in range(12)]
off = {t: rng.normal(0, .13) for t in teams}
def_tov = {t: rng.normal(0, .20) for t in teams}
def_3 = {t: rng.normal(0, .18) for t in teams}
def_fta = {t: rng.normal(0, .15) for t in teams}
def_oreb = {t: rng.normal(0, .15) for t in teams}
def_ast = {t: rng.normal(0, .14) for t in teams}
rows=[]; gid=0; start=pd.Timestamp('2026-01-01')
for r in range(70):
    perm = teams[r % len(teams):] + teams[:r % len(teams)]
    pairs=[(perm[i], perm[i+1]) for i in range(0, len(teams), 2)]
    date=start+pd.Timedelta(days=r)
    for a,b in pairs:
        gid += 1
        for team,opp in ((a,b),(b,a)):
            poss=82.0
            tov_p=1/(1+np.exp(-(-1.72 + .35*off[team] + .85*def_tov[opp] + rng.normal(0,.04))))
            fta_pp=np.exp(-1.45 + .25*off[team] + .55*def_fta[opp] + rng.normal(0,.04))
            share=1/(1+np.exp(-(-.52 + .45*off[team] + .85*def_3[opp] + rng.normal(0,.04))))
            oreb_p=1/(1+np.exp(-(-1.05 + .25*off[team] + .55*def_oreb[opp] + rng.normal(0,.04))))
            astpm=1/(1+np.exp(-(.35 + .35*off[team] + .65*def_ast[opp] + rng.normal(0,.04))))
            tov=max(4,int(round(poss*tov_p)))
            fta=max(2,int(round(poss*fta_pp)))
            fga=max(48,int(round(poss - tov - .44*fta + 9.5)))
            a3=max(5,min(fga-5,int(round(fga*share))))
            a2=fga-a3
            m3=int(round(a3*.34)); m2=int(round(a2*.51)); fgm=m3+m2
            misses=max(fga-fgm,1)
            oreb=max(1,min(misses-1,int(round(misses*oreb_p))))
            ast=max(1,min(fgm,int(round(fgm*astpm))))
            rows.append({
                'GAME_ID':str(gid),'GAME_DATE':date,'TEAM_ABBR':team,'OPP_ABBR':opp,
                'FGM':fgm,'FGA':fga,'FG3M':m3,'FG3A':a3,'FTM':int(round(fta*.79)),'FTA':fta,
                'OREB':oreb,'DREB':25,'REB':oreb+25,'AST':ast,'STL':7,'BLK':4,'TOV':tov,'PF':18,
                'PTS':2*m2+3*m3+int(round(fta*.79)),
            })

df=pd.DataFrame(rows)
models,audit=fit_structural_rate_models(df)
assert set(models) == {'3P_SHARE','FTA','TOV','OREB_PER_MISS','DREB_CAPTURE','AST_PER_MAKE'}
assert all(m.opponent_beta >= 0 for m in models.values())
mods,cur=predict_structural_modifiers(df,'T0','T1',models,WeightConfig.stable(),h2h_rotation_similarity=.8)
assert set(mods)=={'3P_SHARE','FTA','TOV','OREB','DREB','AST'}
assert {'Prediction without H2H','Effective H2H weight','H2H raw-unit delta'}.issubset(cur.columns)
assert np.all(cur['Effective H2H weight'].fillna(0).to_numpy() >= 0)

# ------------------------------------------------------------------
# 2) Market spread receives zero weight unless a historical later holdout
# demonstrates forecast-combination improvement.
# ------------------------------------------------------------------
n=80
model_margin=rng.normal(0,7,n)
true_margin=model_margin + rng.normal(0,5,n)
# Market is an independently informative second forecast.
market_margin=true_margin + rng.normal(0,2.5,n)
cal_df=pd.DataFrame({
    'GAME_DATE':pd.date_range('2025-01-01', periods=n, freq='D'),
    'MODEL_HOME_MARGIN':model_margin,
    'MARKET_HOME_SPREAD':-market_margin,
    'ACTUAL_HOME_MARGIN':true_margin,
})
cal=fit_market_margin_calibration(cal_df)
assert 0 <= cal.weight <= 1
assert cal.rows == n

# Coherent paired simulation. Reweighting must preserve row-wise identities.
sz=20000
home_pts=rng.poisson(88,sz)
away_pts=rng.poisson(82,sz)
h=pd.DataFrame({'PTS':home_pts,'FGA':rng.poisson(70,sz),'3PA':rng.poisson(27,sz)})
a=pd.DataFrame({'PTS':away_pts,'FGA':rng.poisson(68,sz),'3PA':rng.poisson(26,sz)})
if cal.active:
    h2,a2,aud=apply_market_margin_prior(h,a,-10.0,cal,seed=7)
    assert len(h2)==len(h) and len(a2)==len(a)
    assert aud['active']
    before=aud['model_margin_before']; target=aud['target_blended_margin']; after=aud['margin_after']
    assert abs(after-target) < max(0.35, abs(target-before)*0.15)
else:
    h2,a2,aud=apply_market_margin_prior(h,a,-10.0,cal,seed=7)
    assert not aud['active']

# No calibration: guaranteed no-op.
h3,a3,aud3=apply_market_margin_prior(h,a,-10.0,None,seed=8)
assert h3 is h and a3 is a and not aud3['active']

print('v2.18 team-state + calibrated market-prior smoke: PASS')
print(audit[['Feature','Active','Opponent beta','H2H prior K','Later holdout RMSE','Own-only holdout RMSE']])
