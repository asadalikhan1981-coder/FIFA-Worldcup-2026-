import numpy as np

from wc2026 import elo
from wc2026.match_model import MatchModelParams, goal_rates, outcome_probs, score_matrix


def test_elo_expected_score_symmetry():
    assert np.isclose(elo.expected_score(1500, 1500, 0.0), 0.5)
    assert elo.expected_score(1700, 1500, 0.0) > 0.5


def test_elo_goal_diff_multiplier():
    assert elo.goal_diff_multiplier(1) == 1.0
    assert elo.goal_diff_multiplier(2) == 1.5
    assert elo.goal_diff_multiplier(3) == (11 + 3) / 8
    assert elo.goal_diff_multiplier(4) > elo.goal_diff_multiplier(3)


def test_elo_update_conserves_points():
    m = elo.EloModel(ratings={"A": 1600, "B": 1400})
    before = m.get("A") + m.get("B")
    m.update_match("A", "B", 2, 0, k=40, neutral=True)
    after = m.get("A") + m.get("B")
    assert np.isclose(before, after)  # zero-sum update
    assert m.get("A") > 1600  # favourite won, gains


def test_elo_importance_weights():
    assert elo.importance_weight("Friendly") == 20
    assert elo.importance_weight("FIFA World Cup") == 60
    assert elo.importance_weight("FIFA World Cup qualification") == 40


def test_score_matrix_normalised():
    p = MatchModelParams()
    grid = score_matrix(1.5, 1.1, p)
    assert np.isclose(grid.sum(), 1.0)


def test_outcome_probs_sum_to_one_and_favour_stronger():
    p = MatchModelParams()
    probs = outcome_probs(1800, 1500, neutral=True, p=p)
    assert np.isclose(probs.sum(), 1.0)
    assert probs[0] > probs[2]  # stronger first team more likely to win


def test_home_bump_helps_first_team():
    p = MatchModelParams()
    neutral = outcome_probs(1500, 1500, neutral=True, p=p)
    at_home = outcome_probs(1500, 1500, neutral=False, p=p)
    assert at_home[0] > neutral[0]


def test_dixon_coles_lifts_draw_vs_independent():
    # With rho<0 the draw probability should exceed the rho=0 (independent) case.
    base = MatchModelParams(rho=0.0)
    dc = MatchModelParams(rho=-0.05)
    assert outcome_probs(1500, 1500, True, dc)[1] > outcome_probs(1500, 1500, True, base)[1]
