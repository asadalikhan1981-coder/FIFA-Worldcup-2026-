"""Proper scoring rules for probabilistic match-outcome forecasts.

Convention used throughout the project: a forecast for a single match is a
length-3 probability vector ordered ``[home_win, draw, away_win]`` and the
observed outcome is the matching one-hot vector. For neutral-venue knockout
games "home" simply means the team listed first.

All functions are vectorised: pass ``probs`` of shape ``(n, 3)`` and either an
integer outcome array of shape ``(n,)`` (0=home, 1=draw, 2=away) or a one-hot
array of shape ``(n, 3)``.

References
----------
Ranked Probability Score for football, as advocated by Constantinou & Fenton
(2012), "Solving the problem of inadequate scoring rules for assessing
probabilistic football forecast models", *Journal of Quantitative Analysis in
Sports*.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-15


def _as_onehot(outcomes: np.ndarray, n_classes: int = 3) -> np.ndarray:
    outcomes = np.asarray(outcomes)
    if outcomes.ndim == 2:
        return outcomes.astype(float)
    oh = np.zeros((outcomes.shape[0], n_classes))
    oh[np.arange(outcomes.shape[0]), outcomes.astype(int)] = 1.0
    return oh


def ranked_probability_score(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Per-match RPS for ordered categories. Lower is better, range [0, 1].

    RPS = 1/(r-1) * sum_{i=1}^{r-1} (CP_i - CO_i)^2  with r=3 categories.

    The ordering [home, draw, away] is meaningful: predicting an away win when
    the home team wins is penalised more than predicting a draw, because a draw
    is "closer" on the result spectrum. This is exactly why RPS is preferred
    over Brier for football.
    """
    probs = np.asarray(probs, dtype=float)
    obs = _as_onehot(outcomes, probs.shape[1])
    cp = np.cumsum(probs, axis=1)
    co = np.cumsum(obs, axis=1)
    r = probs.shape[1]
    return np.sum((cp[:, :-1] - co[:, :-1]) ** 2, axis=1) / (r - 1)


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Per-match multi-class Brier score = sum_k (p_k - o_k)^2. Lower is better."""
    probs = np.asarray(probs, dtype=float)
    obs = _as_onehot(outcomes, probs.shape[1])
    return np.sum((probs - obs) ** 2, axis=1)


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Per-match log loss = -sum_k o_k log p_k. Lower is better."""
    probs = np.asarray(probs, dtype=float)
    obs = _as_onehot(outcomes, probs.shape[1])
    p = np.clip(probs, EPS, 1.0)
    return -np.sum(obs * np.log(p), axis=1)


def summary(probs: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    """Mean of each scoring rule over all matches, plus sample size."""
    return {
        "n": int(np.asarray(outcomes).shape[0]),
        "rps": float(np.mean(ranked_probability_score(probs, outcomes))),
        "brier": float(np.mean(brier_score(probs, outcomes))),
        "log_loss": float(np.mean(log_loss(probs, outcomes))),
    }


def calibration_table(
    event_probs: np.ndarray, event_observed: np.ndarray, n_bins: int = 10
) -> dict[str, np.ndarray]:
    """Reliability data for a *binary* event (e.g. "home team wins").

    Returns bin centres, mean predicted probability, observed frequency and the
    count in each bin. Feed (predicted P(event), 0/1 observed) for any event you
    want a calibration plot of.
    """
    event_probs = np.asarray(event_probs, dtype=float)
    event_observed = np.asarray(event_observed, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(event_probs, edges[1:-1]), 0, n_bins - 1)
    mean_pred = np.full(n_bins, np.nan)
    obs_freq = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = idx == b
        counts[b] = int(mask.sum())
        if counts[b]:
            mean_pred[b] = event_probs[mask].mean()
            obs_freq[b] = event_observed[mask].mean()
    return {
        "bin_centre": (edges[:-1] + edges[1:]) / 2,
        "mean_pred": mean_pred,
        "obs_freq": obs_freq,
        "count": counts,
    }
