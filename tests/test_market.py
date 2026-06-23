import numpy as np

from wc2026 import market


def test_devig_sums_to_one():
    for fn in (market.devig_proportional, market.devig_shin):
        p = fn(2.0, 3.5, 4.0)
        assert np.isclose(p.sum(), 1.0)
        assert (p > 0).all()


def test_overround_positive_for_real_book():
    # A real line always implies > 1.0 of probability (the vig).
    assert market.overround(2.0, 3.5, 4.0) > 0


def test_fair_odds_recover_truth():
    # Vig-free odds (inverse of a true distribution) must return that distribution.
    true = np.array([0.5, 0.3, 0.2])
    odds = 1.0 / true
    assert np.allclose(market.devig_proportional(*odds), true)
    assert np.isclose(market.overround(*odds), 0.0, atol=1e-9)


def test_proportional_preserves_order():
    # Shorter price -> higher probability, and the favourite stays the favourite.
    p = market.devig_proportional(1.5, 4.0, 7.0)
    assert p[0] > p[1] > p[2]


def test_shin_lifts_favourite_above_proportional():
    # Favourite-longshot bias: longshots are over-bet, so Shin moves probability
    # toward the favourite relative to the naive proportional de-vig.
    prop = market.devig_proportional(1.3, 5.0, 9.0)
    shin = market.devig_shin(1.3, 5.0, 9.0)
    assert shin[0] >= prop[0] - 1e-9
    assert shin[2] <= prop[2] + 1e-9  # longshot trimmed
