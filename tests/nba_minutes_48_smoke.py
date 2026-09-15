import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from core.player_model import PlayerContext, simulate_player


PROFILE = {
    "three_pa_pm": 0.25,
    "two_pa_pm": 0.30,
    "fta_pm": 0.12,
    "reb_pm": 0.18,
    "ast_pm": 0.16,
    "three_pct": 0.37,
    "two_pct": 0.54,
    "ft_pct": 0.82,
}

# High-minute NBA star: Monte Carlo tails may approach 48 but must never exceed it.
ctx = PlayerContext(projected_minutes=45.0, minutes_sd=4.5)
sim = simulate_player(PROFILE, ctx, n=50_000, seed=20260908)
assert float(sim["MIN"].max()) <= 48.0 + 1e-12, sim["MIN"].max()
assert float(sim["MIN"].min()) >= 37.0 - 1e-12, sim["MIN"].min()

# Regulation ceiling remains respected even with a pathological manual projection.
ctx2 = PlayerContext(projected_minutes=48.0, minutes_sd=5.5)
sim2 = simulate_player(PROFILE, ctx2, n=20_000, seed=20260909)
assert float(sim2["MIN"].max()) <= 48.0 + 1e-12, sim2["MIN"].max()

# Guard the final displayed/stress interval in the team minutes engine too.
text = (ROOT / "core" / "minutes_engine.py").read_text(encoding="utf-8")
assert 'out["Projected Min"] - 1.20 * out["Minutes SD"], 0, 48' in text
assert 'out["Projected Min"] + 1.20 * out["Minutes SD"], 0, 48' in text

print("NBA 48-minute exposure smoke: PASS")
