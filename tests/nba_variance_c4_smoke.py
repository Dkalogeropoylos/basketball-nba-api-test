from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.team_model import TeamContext, simulate_game
from backtest.variance_experiment import NBA_FTA_LOG_SIGMA_EARLY_HALF_2024_25


profile = {
    "tov_pp": 0.14,
    "fta_pp": 0.215,
    "three_share": 0.42,
    "three_pct": 0.36,
    "two_pct": 0.55,
    "ft_pct": 0.79,
    "oreb_per_miss": 0.27,
    "assist_per_make": 0.62,
    "pf_pp": 0.19,
    "dreb_capture": 0.78,
    "stl_per_opp_tov": 0.48,
    "blk_rate_pp": 0.048,
    "league_two_pa_pp": 0.50,
}
ctx = TeamContext(projected_possessions=99.0, possessions_sd=4.0)

n = 100_000
base, _ = simulate_game(profile, profile, ctx, ctx, n=n, seed=123)
chain, _ = simulate_game(profile, profile, ctx, ctx, n=n, seed=123, fga_process="rebound_chain")
fta_wide, _ = simulate_game(
    profile, profile, ctx, ctx, n=n, seed=123,
    fta_log_sigma=NBA_FTA_LOG_SIGMA_EARLY_HALF_2024_25,
)

# B1 should preserve the FGA center but remove a large amount of the extra full-Poisson spread.
mean_gap = abs(float(chain["FGA"].mean()) - float(base["FGA"].mean()))
assert mean_gap < 0.35, mean_gap
assert float(chain["FGA"].std(ddof=0)) < 0.80 * float(base["FGA"].std(ddof=0))

# The rebound-chain obeys the possession identity much more tightly because every OREB creates one continuation attempt.
base_identity = base["FGA"] - base["OREB"] + base["TOV"] + 0.44 * base["FTA"] - base["POSS"]
chain_identity = chain["FGA"] - chain["OREB"] + chain["TOV"] + 0.44 * chain["FTA"] - chain["POSS"]
assert float(chain_identity.std(ddof=0)) < 0.35 * float(base_identity.std(ddof=0))

# B2 is mean-preserving to Monte Carlo tolerance but increases FTA spread.
fta_mean_gap = abs(float(fta_wide["FTA"].mean()) - float(base["FTA"].mean()))
assert fta_mean_gap < 0.20, fta_mean_gap
assert float(fta_wide["FTA"].std(ddof=0)) > 1.12 * float(base["FTA"].std(ddof=0))

print("NBA Phase C4 variance smoke: PASS")
print({
    "baseline_FGA_mean": round(float(base['FGA'].mean()), 4),
    "chain_FGA_mean": round(float(chain['FGA'].mean()), 4),
    "baseline_FGA_sd": round(float(base['FGA'].std(ddof=0)), 4),
    "chain_FGA_sd": round(float(chain['FGA'].std(ddof=0)), 4),
    "baseline_FTA_mean": round(float(base['FTA'].mean()), 4),
    "wide_FTA_mean": round(float(fta_wide['FTA'].mean()), 4),
    "baseline_FTA_sd": round(float(base['FTA'].std(ddof=0)), 4),
    "wide_FTA_sd": round(float(fta_wide['FTA'].std(ddof=0)), 4),
})
