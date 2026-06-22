"""Volatility-aware strength model: each team gets a *level* and a *swing*.

This is our candidate *novel method*. It breaks a core assumption shared by
Elo / FIFA / Opta / Dixon-Coles: that a team is one number and every team is
equally consistent. Here each team t has:

    mu_t    -- level (mean latent performance), like an Elo rating
    sigma_t -- swing (match-to-match volatility), a *persistent team trait*

A match between i (listed first / home) and j produces latent performances
P_i ~ N(mu_i + h, sigma_i^2), P_j ~ N(mu_j, sigma_j^2). The margin
M = P_i - P_j ~ N(mu_i - mu_j + h, sigma_i^2 + sigma_j^2). With a symmetric
draw band of half-width delta this gives an *ordered probit with a
team-specific scale* s = sqrt(sigma_i^2 + sigma_j^2):

    P(i win)  = Phi((Delta - delta) / s)
    P(j win)  = Phi((-Delta - delta) / s)      Delta = mu_i - mu_j + h
    P(draw)   = 1 - P(i win) - P(j win)

The novelty is the *team-specific scale*: upset-prone (high-swing) teams widen
the outcome distribution, which is precisely the tail behaviour the standard
homoskedastic model gets wrong and the futures market misprices.

NOT TrueSkill/Glicko: their sigma is *estimation uncertainty* (how unsure we
are of the rating; it shrinks with more games). Ours is *performance
volatility* (how streaky the team genuinely is), a structural property used to
shape the outcome distribution.

The control is the same model with all sigma forced equal (``pool_volatility``):
turning the swing off recovers a homoskedastic ordered probit. So the
volatility contribution is measured by a clean nested A/B.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr  # standard normal CDF, fast

from .. import elo

INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _phi(x: np.ndarray) -> np.ndarray:
    return INV_SQRT_2PI * np.exp(-0.5 * x * x)


@dataclass
class FitResult:
    mu: dict[str, float]
    sigma: dict[str, float]
    delta: float
    home: float
    teams: list[str] = field(default_factory=list)

    def get_mu(self, team: str) -> float:
        return self.mu.get(team, 0.0)

    def get_sigma(self, team: str) -> float:
        return self.sigma.get(team, 1.0)


@dataclass
class TrainView:
    """Vectorised training arrays for one cutoff date."""
    idx_i: np.ndarray
    idx_j: np.ndarray
    home_flag: np.ndarray   # +1 if first team has home advantage, else 0
    outcome: np.ndarray     # 0 = i win, 1 = draw, 2 = j win
    weight: np.ndarray
    teams: list[str]


def build_train_view(
    matches: pd.DataFrame,
    cutoff: pd.Timestamp,
    window_years: float = 8.0,
    halflife_days: float = 730.0,
) -> TrainView:
    """Select pre-cutoff matches, time-decay them, and index the teams."""
    start = cutoff - pd.Timedelta(days=int(window_years * 365.25))
    df = matches[(matches["date"] < cutoff) & (matches["date"] >= start)]
    df = df.sort_values("date", kind="stable")

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    index = {t: k for k, t in enumerate(teams)}

    idx_i = df["home_team"].map(index).to_numpy()
    idx_j = df["away_team"].map(index).to_numpy()
    neutral = df["neutral"].to_numpy()
    home_flag = np.where(neutral, 0.0, 1.0)

    hs, as_ = df["home_score"].to_numpy(), df["away_score"].to_numpy()
    outcome = np.where(hs > as_, 0, np.where(hs == as_, 1, 2))

    age_days = (cutoff - df["date"]).dt.days.to_numpy()
    decay = 0.5 ** (age_days / halflife_days)
    imp = np.array([elo.importance_weight(t) for t in df["tournament"]])
    weight = decay * np.sqrt(imp / 20.0)  # mild upweight of competitive games

    return TrainView(idx_i, idx_j, home_flag, outcome, weight, teams)


def _probs(delta: float, home: float, mu: np.ndarray, gamma: np.ndarray,
           v: TrainView):
    Delta = mu[v.idx_i] - mu[v.idx_j] + home * v.home_flag
    s = np.sqrt(np.exp(2 * gamma[v.idx_i]) + np.exp(2 * gamma[v.idx_j]))
    a = (Delta - delta) / s
    b = (-Delta - delta) / s
    pwin = ndtr(a)
    plose = ndtr(b)
    pdraw = np.clip(1.0 - pwin - plose, 1e-12, None)
    return Delta, s, a, b, pwin, plose, pdraw


def _objective(theta, v: TrainView, n: int, lam_mu: float, lam_gamma: float,
               pool_volatility: bool):
    mu = theta[:n]
    if pool_volatility:
        gamma = np.zeros(n)
    else:
        gamma = theta[n:2 * n]
    delta = np.log1p(np.exp(theta[-2]))      # softplus -> delta > 0
    home = theta[-1]

    Delta, s, a, b, pwin, plose, pdraw = _probs(delta, home, mu, gamma, v)
    pa, pb = _phi(a), _phi(b)

    p_obs = np.where(v.outcome == 0, pwin, np.where(v.outcome == 1, pdraw, plose))
    p_obs = np.clip(p_obs, 1e-12, None)
    nll = -np.sum(v.weight * np.log(p_obs))
    pen = lam_mu * np.sum(mu * mu) + lam_gamma * np.sum(gamma * gamma)
    val = nll + pen

    # d log p / d{Delta, delta, s} for the realised outcome
    dD = np.empty_like(p_obs)
    dDe = np.empty_like(p_obs)
    dS = np.empty_like(p_obs)
    win, draw, lose = v.outcome == 0, v.outcome == 1, v.outcome == 2

    dD[win] = (pa[win] / s[win]) / pwin[win]
    dDe[win] = (-pa[win] / s[win]) / pwin[win]
    dS[win] = (-pa[win] * a[win] / s[win]) / pwin[win]

    dD[lose] = (-pb[lose] / s[lose]) / plose[lose]
    dDe[lose] = (-pb[lose] / s[lose]) / plose[lose]
    dS[lose] = (-pb[lose] * b[lose] / s[lose]) / plose[lose]

    dD[draw] = ((pb[draw] - pa[draw]) / s[draw]) / pdraw[draw]
    dDe[draw] = ((pa[draw] + pb[draw]) / s[draw]) / pdraw[draw]
    dS[draw] = ((pa[draw] * a[draw] + pb[draw] * b[draw]) / s[draw]) / pdraw[draw]

    w = v.weight
    gD = -w * dD            # d nll / d Delta_m
    gDe = -w * dDe          # d nll / d delta_m
    gS = -w * dS            # d nll / d s_m

    grad = np.zeros_like(theta)
    # mu: dDelta/dmu_i = +1, dmu_j = -1
    np.add.at(grad, v.idx_i, gD)
    np.add.at(grad, v.idx_j, -gD)
    grad[:n] += 2 * lam_mu * mu
    # gamma via s: ds/dgamma_t = v_t / s, v_t = exp(2 gamma_t)
    if not pool_volatility:
        vi = np.exp(2 * gamma[v.idx_i]) / s
        vj = np.exp(2 * gamma[v.idx_j]) / s
        np.add.at(grad, n + v.idx_i, gS * vi)
        np.add.at(grad, n + v.idx_j, gS * vj)
        grad[n:2 * n] += 2 * lam_gamma * gamma
    # delta (softplus) and home
    ddelta_draw = 1.0 / (1.0 + np.exp(-theta[-2]))   # sigmoid
    grad[-2] = np.sum(gDe) * ddelta_draw
    grad[-1] = np.sum(gD * v.home_flag)
    return val, grad


def fit(
    matches: pd.DataFrame,
    cutoff: pd.Timestamp,
    pool_volatility: bool = False,
    lam_mu: float = 1.0,
    lam_gamma: float = 8.0,
    window_years: float = 8.0,
    halflife_days: float = 730.0,
) -> FitResult:
    """Fit level (and swing, unless pooled) by penalised maximum likelihood.

    ``lam_gamma`` is the key knob: large -> swings shrink to equal (homoskedastic);
    it is the partial-pooling strength that pays the variance tax. ``pool_volatility``
    forces all swings equal (the control).
    """
    v = build_train_view(matches, cutoff, window_years, halflife_days)
    n = len(v.teams)
    # theta = [mu(n), gamma(n), delta_raw, home]
    theta0 = np.zeros(2 * n + 2)
    theta0[-2] = -1.0   # softplus(-1) ~ 0.31 starting draw band
    theta0[-1] = 0.3    # small positive home edge on the latent scale

    res = minimize(
        _objective, theta0, args=(v, n, lam_mu, lam_gamma, pool_volatility),
        jac=True, method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
    )
    theta = res.x
    mu = theta[:n]
    gamma = np.zeros(n) if pool_volatility else theta[n:2 * n]
    delta = float(np.log1p(np.exp(theta[-2])))
    home = float(theta[-1])
    return FitResult(
        mu={t: float(mu[k]) for k, t in enumerate(v.teams)},
        sigma={t: float(np.exp(gamma[k])) for k, t in enumerate(v.teams)},
        delta=delta, home=home, teams=v.teams,
    )


def predict(
    fit_res: FitResult, team_i: str, team_j: str, neutral: bool = True
) -> np.ndarray:
    """[P(i win), P(draw), P(j win)] for team i (first) vs j."""
    mu_i, mu_j = fit_res.get_mu(team_i), fit_res.get_mu(team_j)
    si, sj = fit_res.get_sigma(team_i), fit_res.get_sigma(team_j)
    Delta = mu_i - mu_j + (0.0 if neutral else fit_res.home)
    s = np.sqrt(si * si + sj * sj)
    pwin = ndtr((Delta - fit_res.delta) / s)
    plose = ndtr((-Delta - fit_res.delta) / s)
    pdraw = max(1.0 - pwin - plose, 1e-12)
    out = np.array([pwin, pdraw, plose])
    return out / out.sum()
