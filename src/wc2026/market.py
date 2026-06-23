"""Turn bookmaker 1X2 decimal odds into probabilities.

A bookmaker's quoted prices imply probabilities that sum to more than 1: the
excess is the *overround* (vig), the book's margin. To compare a model against
the market we must strip that margin back out. Two standard methods:

* ``devig_proportional`` - normalise the inverse odds so they sum to 1. Simple
  and the usual default; it removes the margin evenly across outcomes.
* ``devig_shin`` - Shin (1992) backs out a fraction ``z`` of insider money and
  removes the margin *unevenly*. Because longshots are over-bet, it shifts
  probability toward the favourite relative to the proportional method. A
  robustness check on the proportional baseline.

Convention matches the rest of the repo: odds/probabilities are ordered
``[home, draw, away]``.

Shin, H.S. (1992), "Prices of State Contingent Claims with Insider Traders, and
the Favourite-Longshot Bias", *Economic Journal* 102.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def implied_inverse(o_home: float, o_draw: float, o_away: float) -> np.ndarray:
    """Raw inverse odds [1/o_home, 1/o_draw, 1/o_away]; sums to 1 + overround."""
    return np.array([1.0 / o_home, 1.0 / o_draw, 1.0 / o_away])


def overround(o_home: float, o_draw: float, o_away: float) -> float:
    """Booksum minus 1 (the bookmaker's margin); ~0.03 for sharp closing lines."""
    return float(implied_inverse(o_home, o_draw, o_away).sum() - 1.0)


def devig_proportional(o_home: float, o_draw: float, o_away: float) -> np.ndarray:
    """De-vig by normalising inverse odds to sum to 1."""
    inv = implied_inverse(o_home, o_draw, o_away)
    return inv / inv.sum()


def devig_shin(o_home: float, o_draw: float, o_away: float) -> np.ndarray:
    """Shin (1992) de-vig. Falls back to proportional if the root solve fails."""
    pi = implied_inverse(o_home, o_draw, o_away)
    booksum = pi.sum()

    def p_of_z(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4 * (1 - z) * pi * pi / booksum) - z) / (2 * (1 - z))

    try:
        z = brentq(lambda z: p_of_z(z).sum() - 1.0, 1e-9, 0.2)
    except ValueError:
        return pi / booksum
    p = p_of_z(z)
    return p / p.sum()
