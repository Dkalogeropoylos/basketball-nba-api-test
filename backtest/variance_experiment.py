from __future__ import annotations

from dataclasses import dataclass


# Frozen C3 baseline values.
BASELINE_FTA_LOG_SIGMA = 0.12

# Phase C4 development calibration.
# IMPORTANT: fitted ONLY on the FIRST HALF of the 2024-25 eligible validation games
# (497 games / 994 team observations), leaving the final 498 games untouched for A/B.
#
# Method-of-moments approximation for a mean-preserving Poisson-lognormal rate shock:
#   lambda = mu * exp(sigma*z - sigma^2/2), z~N(0,1)
#   Var contribution from the lognormal rate mixture ~= mu^2 * (exp(sigma^2)-1)
#
# We hold all other C3 variance components fixed and solve
#   exp(s1^2) = exp(s0^2) + (V_obs - V_mc) / E[mu^2]
# using ONLY the early-half C3 baseline detail rows for TEAM FTA.
#
# Early-half audit:
#   games                 = 497
#   team rows             = 994
#   baseline sigma s0     = 0.120000
#   TEAM FTA bias         = +0.278051 (projection - actual)
#   residual variance     = 42.683152
#   mean C3 MC variance   = 29.142369
#   mean projection^2     = 471.672857
#   fitted sigma s1       = 0.205681
#
# This value is a DEVELOPMENT parameter. It must be judged only on the untouched
# late-half 498 games and then frozen before any 2025-26 final test.
NBA_FTA_LOG_SIGMA_EARLY_HALF_2024_25 = 0.20568078391234368


@dataclass(frozen=True)
class VarianceVariant:
    key: str
    label: str
    fga_process: str
    fta_log_sigma: float
    description: str


VARIANTS = {
    "baseline": VarianceVariant(
        key="baseline",
        label="A — Frozen C3 baseline",
        fga_process="poisson",
        fta_log_sigma=BASELINE_FTA_LOG_SIGMA,
        description="Original full Poisson FGA redraw + original FTA log-rate sigma 0.12.",
    ),
    "fga_chain": VarianceVariant(
        key="fga_chain",
        label="B1 — FGA rebound-chain only",
        fga_process="rebound_chain",
        fta_log_sigma=BASELINE_FTA_LOG_SIGMA,
        description="Replaces only the final Poisson(FGA_mean) redraw with a possession/miss/OREB continuation chain.",
    ),
    "fta_dispersion": VarianceVariant(
        key="fta_dispersion",
        label="B2 — FTA dispersion only",
        fga_process="poisson",
        fta_log_sigma=NBA_FTA_LOG_SIGMA_EARLY_HALF_2024_25,
        description="Keeps FGA baseline; increases only mean-preserving Poisson-lognormal foul-intensity dispersion using early-half calibration.",
    ),
    "combined": VarianceVariant(
        key="combined",
        label="B3 — Combined FGA chain + FTA dispersion",
        fga_process="rebound_chain",
        fta_log_sigma=NBA_FTA_LOG_SIGMA_EARLY_HALF_2024_25,
        description="Applies both independently motivated variance changes. Use only after B1/B2 are understood.",
    ),
}


def late_half_start_index(eligible_games: int) -> int:
    """Chronological split: first floor(N/2) games calibrate, last games evaluate."""
    return int(eligible_games) // 2
