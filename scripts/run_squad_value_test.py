"""Does SQUAD MARKET VALUE (a forward-looking signal Elo lacks) beat Elo OOS?

Squad values at each tournament's kickoff come from the free, public
ericsanmiguel/football_elo dataset (Transfermarkt). We have 2018 + 2022 men's
World Cups to backtest on and 2026 for the live forecast.

Test (leave-one-tournament-out, no leakage): for each of WC2018 / WC2022 build
each match's pre-tournament Elo gap and squad-value gap, fit an ordered probit on
one tournament, predict the other. Compare Elo-only vs Elo+squad-value.

    python scripts/run_squad_value_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wc2026 import data, elo, metrics  # noqa: E402

EPS = 1e-12

# squad-file name -> martj42 results name
NAME_FIX = {
    "South Korea": "South Korea", "Korea Republic": "South Korea",
    "IR Iran": "Iran", "USA": "United States", "Costa Rica": "Costa Rica",
    "Cabo Verde": "Cape Verde", "Czechia": "Czech Republic",
    "Türkiye": "Turkey", "Turkiye": "Turkey", "Curacao": "Curaçao",
}


def squad_strength(year: int) -> dict[str, float]:
    """log(total squad market value), z-scored across the tournament field."""
    df = pd.read_csv(f"data/raw/squads/{year}.csv")
    df["team"] = df["team"].map(lambda t: NAME_FIX.get(t, t))
    tot = df.groupby("team")["value_at_kickoff"].sum()
    logv = np.log(tot.clip(lower=1.0))
    z = (logv - logv.mean()) / (logv.std(ddof=0) + 1e-9)
    return z.to_dict()


def build(results):
    specs = {"WC2018": data.BACKTEST_TOURNAMENTS["WC2018"],
             "WC2022": data.BACKTEST_TOURNAMENTS["WC2022"]}
    rows, unmatched = [], set()
    for name, spec in specs.items():
        cutoff = pd.Timestamp(spec["start"])
        year = int(name[-4:])
        model = elo.build_history(results, until=cutoff)
        sv = squad_strength(year)
        games = data.tournament_matches(results, spec)
        for g in games.itertuples(index=False):
            for t in (g.home_team, g.away_team):
                if t not in sv:
                    unmatched.add(t)
            rows.append({
                "edition": name,
                "elo": (model.get(g.home_team) - model.get(g.away_team)) / 200.0,
                "sqval": sv.get(g.home_team, 0.0) - sv.get(g.away_team, 0.0),
                "outcome": 0 if g.home_score > g.away_score else (1 if g.home_score == g.away_score else 2),
            })
    if unmatched:
        print("WARN unmatched squad teams (treated as average):", sorted(unmatched))
    return pd.DataFrame(rows)


def _nll(theta, X, y, ridge=1e-3):
    k = X.shape[1]; beta = theta[:k]; delta = np.log1p(np.exp(theta[-1]))
    eta = X @ beta
    pW = ndtr(eta - delta); pL = ndtr(-eta - delta)
    pD = np.clip(1 - pW - pL, EPS, None)
    p = np.where(y == 0, pW, np.where(y == 1, pD, pL))
    return -np.sum(np.log(np.clip(p, EPS, None))) + ridge * np.sum(beta * beta)


def fit_predict(Xtr, ytr, Xte):
    k = Xtr.shape[1]
    theta0 = np.concatenate([np.zeros(k), [-1.0]]); theta0[0] = 0.5
    r = minimize(_nll, theta0, args=(Xtr, ytr), method="L-BFGS-B", options={"maxiter": 500})
    beta = r.x[:k]; delta = np.log1p(np.exp(r.x[-1]))
    eta = Xte @ beta
    P = np.vstack([ndtr(eta - delta), np.clip(1 - ndtr(eta - delta) - ndtr(-eta - delta), EPS, None), ndtr(-eta - delta)]).T
    return P / P.sum(axis=1, keepdims=True), beta


def loo(df, feats):
    preds = np.zeros((len(df), 3)); coefs = []
    for ed in df["edition"].unique():
        tr, te = df[df.edition != ed], df[df.edition == ed]
        P, b = fit_predict(tr[feats].to_numpy(), tr["outcome"].to_numpy(), te[feats].to_numpy())
        preds[df.edition.to_numpy() == ed] = P; coefs.append(b)
    return preds, np.mean(coefs, axis=0)


def main():
    results = data.load_results()
    df = build(results)
    y = df["outcome"].to_numpy()
    print(f"\nBacktest matches with squad values: {len(df)} (WC2018 + WC2022)\n")

    P_elo, b_elo = loo(df, ["elo"])
    P_sv, b_sv = loo(df, ["elo", "sqval"])
    rps_elo = metrics.ranked_probability_score(P_elo, y)
    rps_sv = metrics.ranked_probability_score(P_sv, y)

    print("=== Out-of-sample (leave-one-tournament-out) ===")
    print(f"Elo only           RPS={rps_elo.mean():.4f}   coef: elo={b_elo[0]:+.2f}")
    print(f"Elo + squad value  RPS={rps_sv.mean():.4f}   coef: elo={b_sv[0]:+.2f}, squad={b_sv[1]:+.2f}")

    d = rps_sv - rps_elo
    se = d.std(ddof=1) / np.sqrt(len(d)); t = d.mean() / se
    print(f"\nΔRPS (squad value vs Elo) = {d.mean():+.4f}  (t={t:+.2f})")
    # per-tournament
    for ed in df["edition"].unique():
        m = df.edition.to_numpy() == ed
        print(f"   {ed}: Elo {rps_elo[m].mean():.4f} -> +squad {rps_sv[m].mean():.4f} "
              f"({rps_sv[m].mean()-rps_elo[m].mean():+.4f})")
    verdict = "BEATS Elo" if d.mean() < 0 else "does NOT beat Elo"
    sig = "significant" if abs(t) >= 2 else "not significant (small sample)"
    print(f"\nVERDICT: squad value {verdict} out-of-sample — {sig}.")


if __name__ == "__main__":
    main()
