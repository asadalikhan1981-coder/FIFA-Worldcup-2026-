import numpy as np

from wc2026 import metrics


def test_rps_perfect_is_zero():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(metrics.ranked_probability_score(probs, [0, 2]), 0.0)


def test_rps_orders_distance():
    # Home team wins. Predicting away (far) must score worse than draw (near).
    p_away = np.array([[0.0, 0.0, 1.0]])
    p_draw = np.array([[0.0, 1.0, 0.0]])
    rps_away = metrics.ranked_probability_score(p_away, [0])[0]
    rps_draw = metrics.ranked_probability_score(p_draw, [0])[0]
    assert rps_away > rps_draw


def test_rps_known_value():
    # Classic Constantinou example: pred [1,0,0], outcome draw -> RPS = 0.5
    p = np.array([[1.0, 0.0, 0.0]])
    assert np.isclose(metrics.ranked_probability_score(p, [1])[0], 0.5)


def test_brier_and_logloss_perfect():
    probs = np.array([[1.0, 0.0, 0.0]])
    assert np.isclose(metrics.brier_score(probs, [0])[0], 0.0)
    assert metrics.log_loss(probs, [0])[0] < 1e-10


def test_logloss_penalises_confident_wrong():
    confident_wrong = np.array([[1e-6, 1e-6, 1.0]])
    assert metrics.log_loss(confident_wrong, [0])[0] > 5.0


def test_calibration_table_recovers_frequency():
    # All predictions at 0.7, half observed positive -> obs_freq ~0.5 in that bin.
    probs = np.full(1000, 0.7)
    obs = np.array([1, 0] * 500)
    cal = metrics.calibration_table(probs, obs, n_bins=10)
    b = 6  # bin covering [0.6, 0.7)
    assert cal["count"][b] == 1000
    assert np.isclose(cal["obs_freq"][b], 0.5, atol=1e-9)
