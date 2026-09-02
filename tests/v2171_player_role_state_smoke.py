import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from core.buckets import WeightConfig
from core.player_model import build_player_profile


def make_log(ast_old=4, ast_mid=4, ast_l5=4, fga_old=10, fga_mid=10, fga_l5=10):
    rows=[]
    for i in range(26):
        if i < 16:
            ast, fga = ast_old, fga_old
        elif i < 21:
            ast, fga = ast_mid, fga_mid
        else:
            ast, fga = ast_l5, fga_l5
        rows.append({
            'GAME_ID':str(i+1), 'GAME_DATE':pd.Timestamp('2026-01-01')+pd.Timedelta(days=i),
            'MIN':30, 'FGA':fga, 'FGM':max(int(round(fga*0.45)),1), 'FG3A':4, 'FG3M':1,
            'FTA':3, 'FTM':2, 'REB':5, 'AST':ast, 'PTS':14,
        })
    return pd.DataFrame(rows)

# Stable history should remain almost unchanged.
stable = make_log()
prof_s, _ = build_player_profile(stable, WeightConfig.stable())
rs_s = prof_s['_role_state_audit'].set_index('Opportunity')
assert abs(float(rs_s.loc['AST/min','Applied modifier']) - 1.0) < 0.08, rs_s

# Sustained creation role change: mid and L5 both higher than old. The adaptive
# state must lift AST/min beyond the fixed 55/20/25 bucket baseline, but it must
# not alter shooting percentages.
changed = make_log(ast_old=3, ast_mid=5, ast_l5=7)
prof_c, audit_c = build_player_profile(changed, WeightConfig.stable())
fixed_ast = 0.55*(3/30) + 0.20*(5/30) + 0.25*(7/30)
assert prof_c['ast_pm'] > fixed_ast * 1.02, (prof_c['ast_pm'], fixed_ast, prof_c['_role_state_audit'])
assert bool(prof_c['_role_state_audit'].set_index('Opportunity').loc['AST/min','Active'])
assert 0.25 < prof_c['three_pct'] < 0.45

# Sustained shot-volume role change should lift 2PA/min too, without touching
# trader minutes (minutes are not part of this module).
shots = make_log(fga_old=10, fga_mid=13, fga_l5=16)
prof_f, _ = build_player_profile(shots, WeightConfig.stable())
fixed_2pa = 0.55*((10-4)/30) + 0.20*((13-4)/30) + 0.25*((16-4)/30)
assert prof_f['two_pa_pm'] > fixed_2pa * 1.01, (prof_f['two_pa_pm'], fixed_2pa)

print('v2.17.1 adaptive player role-state smoke: PASS')
