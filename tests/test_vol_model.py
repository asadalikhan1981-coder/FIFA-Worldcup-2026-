import numpy as np
import pandas as pd

from wc2026.models import volatility_strength as vs


def _toy_matches(n=400, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{k}" for k in range(12)]
    dates = pd.date_range("2015-01-01", periods=n, freq="7D")
    home = rng.choice(teams, n)
    away = rng.choice(teams, n)
    keep = home != away
    hs = rng.poisson(1.4, n)
    as_ = rng.poisson(1.2, n)
    df = pd.DataFrame({
        "date": dates, "home_team": home, "away_team": away,
        "home_score": hs, "away_score": as_,
        "tournament": "Friendly", "neutral": rng.random(n) < 0.3,
    })
    return df[keep].reset_index(drop=True)


def test_gradient_matches_numerical():
    df = _toy_matches()
    v = vs.build_train_view(df, pd.Timestamp("2023-01-01"))
    n = len(v.teams)
    rng = np.random.default_rng(1)
    theta = rng.normal(scale=0.3, size=2 * n + 2)

    val, grad = vs._objective(theta, v, n, lam_mu=1.0, lam_gamma=8.0,
                              pool_volatility=False)
    # central finite differences on a random subset of coordinates
    eps = 1e-6
    for k in rng.choice(len(theta), size=15, replace=False):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += eps; tm[k] -= eps
        fp = vs._objective(tp, v, n, 1.0, 8.0, False)[0]
        fm = vs._objective(tm, v, n, 1.0, 8.0, False)[0]
        num = (fp - fm) / (2 * eps)
        assert abs(num - grad[k]) < 1e-3, f"grad mismatch at {k}: {num} vs {grad[k]}"


def test_fit_runs_and_predicts_valid_distribution():
    df = _toy_matches()
    fr = vs.fit(df, pd.Timestamp("2023-01-01"), pool_volatility=False)
    p = vs.predict(fr, "T0", "T1", neutral=True)
    assert np.isclose(p.sum(), 1.0)
    assert np.all(p >= 0)


def test_pooled_makes_sigma_constant():
    df = _toy_matches()
    fr = vs.fit(df, pd.Timestamp("2023-01-01"), pool_volatility=True)
    sigmas = np.array(list(fr.sigma.values()))
    assert np.allclose(sigmas, sigmas[0])  # swing OFF -> all equal


def test_higher_swing_widens_outcome():
    # Two even teams: more combined swing -> lower draw prob, fatter win/lose tails.
    base = vs.FitResult(mu={"A": 0.0, "B": 0.0}, sigma={"A": 0.5, "B": 0.5},
                        delta=0.3, home=0.0)
    wide = vs.FitResult(mu={"A": 0.0, "B": 0.0}, sigma={"A": 1.5, "B": 1.5},
                        delta=0.3, home=0.0)
    p_base = vs.predict(base, "A", "B")
    p_wide = vs.predict(wide, "A", "B")
    assert p_wide[1] < p_base[1]            # wider swing -> fewer draws
    assert p_wide[0] > p_base[0]            # fatter win tail
