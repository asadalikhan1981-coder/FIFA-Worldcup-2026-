"""Dixon-Coles bivariate-Poisson match model: ratings -> goal rates -> W/D/L.

This is the *control's* match engine, built to the handover spec:

    dR_eff_i = (R_i - R_j)/2 + home_i
    lambda_i = BASE * exp(dR_eff_i / SCALE)

with the Dixon-Coles low-score correction tau(x, y; lambda_i, lambda_j, rho)
applied to the independent-Poisson score grid. The correction lifts the
probability mass on 0-0/1-1 and trims 1-0/0-1, fixing the well-known tendency
of the independent Poisson to under-predict draws.

Defaults are the handover's baseline numbers (BASE=1.35, SCALE=215,
home bump=55 Elo, rho=-0.05). These are a *specified prior*, not values tuned
on any test set.

Dixon, M.J. & Coles, S.G. (1997), "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market", *Applied Statistics* 46(2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson


@dataclass(frozen=True)
class MatchModelParams:
    base: float = 1.35
    scale: float = 215.0
    home_bump: float = 55.0  # Elo points added to the home side when not neutral
    rho: float = -0.05       # Dixon-Coles low-score dependence
    max_goals: int = 10


def goal_rates(
    r_i: float, r_j: float, neutral: bool, p: MatchModelParams
) -> tuple[float, float]:
    """Expected goals (lambda_i, lambda_j) for team i (listed first) vs j."""
    home_i = 0.0 if neutral else p.home_bump
    dr_i = (r_i - r_j) / 2.0 + home_i
    dr_j = (r_j - r_i) / 2.0  # the second team never gets the host bump
    lam_i = p.base * np.exp(dr_i / p.scale)
    lam_j = p.base * np.exp(dr_j / p.scale)
    return lam_i, lam_j


def _dc_correction(rho: float, lam_i: float, lam_j: float, n: int) -> np.ndarray:
    """tau matrix of shape (n, n) for scorelines x=0..n-1, y=0..n-1."""
    tau = np.ones((n, n))
    tau[0, 0] = 1.0 - lam_i * lam_j * rho
    tau[0, 1] = 1.0 + lam_i * rho
    tau[1, 0] = 1.0 + lam_j * rho
    tau[1, 1] = 1.0 - rho
    return tau


def score_matrix(lam_i: float, lam_j: float, p: MatchModelParams) -> np.ndarray:
    """Joint scoreline probability matrix P[x, y], normalised to sum to 1."""
    n = p.max_goals + 1
    px = poisson.pmf(np.arange(n), lam_i)
    py = poisson.pmf(np.arange(n), lam_j)
    grid = np.outer(px, py) * _dc_correction(p.rho, lam_i, lam_j, n)
    grid = np.clip(grid, 0.0, None)  # tau can dip <0 for extreme rho; guard it
    return grid / grid.sum()


def outcome_probs(
    r_i: float, r_j: float, neutral: bool, p: MatchModelParams | None = None
) -> np.ndarray:
    """Return [P(i win), P(draw), P(j win)] for team i (first) vs team j."""
    p = p or MatchModelParams()
    lam_i, lam_j = goal_rates(r_i, r_j, neutral, p)
    grid = score_matrix(lam_i, lam_j, p)
    home = np.tril(grid, -1).sum()   # x > y
    draw = np.trace(grid)            # x == y
    away = np.triu(grid, 1).sum()    # x < y
    return np.array([home, draw, away])
