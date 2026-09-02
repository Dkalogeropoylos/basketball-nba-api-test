import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from core.player_model import PlayerContext, simulate_player
from core.matchup import player_h2h_modifiers

# 1) 2PA volume must respond to opp_2pa, not opp_pts.
profile = {
    'three_pa_pm':0.10,'two_pa_pm':0.30,'fta_pm':0.08,'reb_pm':0.12,'ast_pm':0.10,
    'three_pct':0.34,'two_pct':0.50,'ft_pct':0.78,
}
base = simulate_player(profile, PlayerContext(projected_minutes=30, minutes_sd=0, opp_pts=1.20, opp_2pa=1.00), n=60000, seed=101)
low2 = simulate_player(profile, PlayerContext(projected_minutes=30, minutes_sd=0, opp_pts=1.20, opp_2pa=0.90), n=60000, seed=101)
assert low2['PTS'].mean() < base['PTS'].mean(), (low2['PTS'].mean(), base['PTS'].mean())

# 2) H2H is tiny even with a strong raw ratio.
log = pd.DataFrame([
    {'GAME_ID':'1','GAME_DATE':'2026-01-01','OPP_ABBR':'BBB','MIN':30,'FGA':18,'FG3A':6,'REB':8,'AST':8},
    {'GAME_ID':'2','GAME_DATE':'2026-02-01','OPP_ABBR':'BBB','MIN':29,'FGA':17,'FG3A':5,'REB':7,'AST':7},
])
prof = {'two_pa_pm':0.25,'three_pa_pm':0.10,'reb_pm':0.12,'ast_pm':0.10}
mods,aud = player_h2h_modifiers(log,'BBB',prof,30,rotation_similarity=1.0,max_weight=0.05)
assert all(0.97 <= v <= 1.03 for v in mods.values()), mods
assert float(aud['Applied H2H weight'].max()) <= 0.05 + 1e-12
print('v2.16.1 stabilization smoke: PASS', mods)
