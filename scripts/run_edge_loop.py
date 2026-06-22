"""Thorough edge loop: do any reachable NOVEL signals beat plain Elo out-of-sample?

Design (no leakage, no test-set tuning):
  - For every backtest tournament, compute each match's pre-tournament Elo gap and
    the novel features (late-game strength, recent form) strictly as of the
    tournament start date.
  - Pool all tournament matches. For each candidate feature set, run
    leave-one-tournament-out cross-validation: fit the ordered-probit
    coefficients on four editions, predict the held-out edition. Every prediction
    is genuinely out-of-sample.
  - Score pooled OOS predictions (RPS / Brier / log-loss) and test each candidate
    against the Elo-only baseline with a paired t on per-match RPS.

    python scripts/run_edge_loop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wc2026 import data, elo, features, metrics  # noqa: E402

EPS = 1e-12


def build_dataset(results, goalscorers):
    rows = []
    for name, spec in data.BACKTEST_TOURNAMENTS.items():
        cutoff = pd.Timestamp(spec["start"])
        model = elo.build_history(results, until=cutoff)
        feat = features.compute_features(results, goalscorers, cutoff)
        games = data.tournament_matches(results, spec)
        for g in games.itertuples(index=False):
            li = feat.loc[g.home_team] if g.home_team in feat.index else None
            lj = feat.loc[g.away_team] if g.away_team in feat.index else None
            rows.append({
                "edition": name,
                "elo": (model.get(g.home_team) - model.get(g.away_team)) / 200.0,
                "late": (0.0 if li is None else li["late_z"]) - (0.0 if lj is None else lj["late_z"]),
                "form": (0.0 if li is None else li["form_z"]) - (0.0 if lj is None else lj["form_z"]),
                "outcome": 0 if g.home_score > g.away_score else (1 if g.home_score == g.away_score else 2),
            })
    return pd.DataFrame(rows)


def _nll(theta, X, y, ridge=1e-3):
    k = X.shape[1]
    beta = theta[:k]
    delta = np.log1p(np.exp(theta[-1]))
    eta = X @ beta
    pW = ndtr(eta - delta)
    pL = ndtr(-eta - delta)
    pD = np.clip(1 - pW - pL, EPS, None)
    p = np.where(y == 0, pW, np.where(y == 1, pD, pL))
    return -np.sum(np.log(np.clip(p, EPS, None))) + ridge * np.sum(beta * beta)


def fit_predict(Xtr, ytr, Xte):
    k = Xtr.shape[1]
    theta0 = np.concatenate([np.zeros(k), [-1.0]])
    theta0[0] = 0.5  # sensible start for the elo coefficient
    res = minimize(_nll, theta0, args=(Xtr, ytr), method="L-BFGS-B",
                   options={"maxiter": 500})
    beta = res.x[:k]
    delta = np.log1p(np.exp(res.x[-1]))
    eta = Xte @ beta
    pW = ndtr(eta - delta); pL = ndtr(-eta - delta)
    pD = np.clip(1 - pW - pL, EPS, None)
    P = np.vstack([pW, pD, pL]).T
    return P / P.sum(axis=1, keepdims=True), beta


def loo_cv(df, feats):
    preds = np.zeros((len(df), 3))
    coefs = []
    for ed in df["edition"].unique():
        tr = df[df["edition"] != ed]
        te = df[df["edition"] == ed]
        P, beta = fit_predict(tr[feats].to_numpy(), tr["outcome"].to_numpy(),
                              te[feats].to_numpy())
        preds[df["edition"].to_numpy() == ed] = P
        coefs.append(beta)
    return preds, np.mean(coefs, axis=0)


def main():
    results = data.load_results()
    goalscorers = pd.read_csv("data/raw/goalscorers.csv")
    df = build_dataset(results, goalscorers)
    y = df["outcome"].to_numpy()

    candidates = {
        "Elo only (baseline)": ["elo"],
        "Elo + form": ["elo", "form"],
        "Elo + late-strength": ["elo", "late"],
        "Elo + form + late": ["elo", "form", "late"],
    }

    base_rps = None
    rows = []
    rps_by_model = {}
    for name, feats in candidates.items():
        P, beta = loo_cv(df, feats)
        rps = metrics.ranked_probability_score(P, y)
        rps_by_model[name] = rps
        s = metrics.summary(P, y)
        coef_str = ", ".join(f"{f}={b:+.2f}" for f, b in zip(feats, beta))
        rows.append({"model": name, "rps": s["rps"], "brier": s["brier"],
                     "log_loss": s["log_loss"], "coef": coef_str})

    base_rps = rps_by_model["Elo only (baseline)"]
    print("\n=== Edge loop: leave-one-tournament-out, OOS (lower RPS = better) ===")
    tbl = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}", "display.width", 140):
        print(tbl.to_string(index=False))

    print("\n--- vs Elo-only baseline (paired t on per-match RPS) ---")
    for name in candidates:
        if name == "Elo only (baseline)":
            continue
        d = rps_by_model[name] - base_rps
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se else float("nan")
        verdict = "BEATS baseline" if (d.mean() < 0 and abs(t) >= 2) else \
                  ("better but NS" if d.mean() < 0 else "no improvement")
        print(f"{name:<22} dRPS={d.mean():+.4f}  t={t:+.2f}  -> {verdict}")


if __name__ == "__main__":
    main()
