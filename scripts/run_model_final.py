"""Definitive model: Elo + squad value + star-concentration + host advantage.

Features per team (computed strictly before each tournament, no leakage):
  - elo            : results-based strength
  - logtot_z       : log(total squad market value), z-scored across the field
  - share_z        : top-3 players' share of squad value, z-scored (star concentration)
Plus a calibrated, team-specific host bonus for 2026 (USA/Mexico full, Canada reduced).

Coefficients are fit on the 2018 + 2022 World Cups (the editions with squad data),
then applied to 2026. Backtest (leave-one-tournament-out) selected this feature
set: each addition lowered out-of-sample RPS (Elo 0.2152 -> +squad 0.2039 ->
+star 0.2000).

    python scripts/run_model_final.py [n_sims]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wc2026 import data, elo, simulate  # noqa: E402

rf = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("rf", "scripts/run_forecast.py"))
sys.modules["rf"] = rf; rf.__loader__.exec_module(rf)
sv = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("sv", "scripts/run_squad_value_test.py"))
sys.modules["sv"] = sv; sv.__loader__.exec_module(sv)

WC_START = pd.Timestamp("2026-06-11")
FEATS = ["elo", "sqval", "star"]
HOST_ETA = {"United States": 0.33, "Mexico": 0.33, "Canada": 0.20}


def squad_features(year):
    df = pd.read_csv(f"data/raw/squads/{year}.csv")
    df["team"] = df["team"].map(lambda t: sv.NAME_FIX.get(t, t))
    g = df.groupby("team")["value_at_kickoff"]
    tot = g.sum(); top3 = g.apply(lambda s: s.nlargest(3).sum())
    z = lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    return pd.DataFrame({"logtot_z": z(np.log(tot.clip(lower=1))),
                         "share_z": z(top3 / tot)})


def build_training(results):
    rows = []
    for name in ("WC2018", "WC2022"):
        spec = data.BACKTEST_TOURNAMENTS[name]; yr = int(name[-4:])
        m = elo.build_history(results, until=pd.Timestamp(spec["start"]))
        f = squad_features(yr)
        for x in data.tournament_matches(results, spec).itertuples(index=False):
            fi = f.loc[x.home_team] if x.home_team in f.index else None
            fj = f.loc[x.away_team] if x.away_team in f.index else None
            gv = lambda ff, c: (0.0 if ff is None else ff[c])
            rows.append(dict(
                elo=(m.get(x.home_team) - m.get(x.away_team)) / 200,
                sqval=gv(fi, "logtot_z") - gv(fj, "logtot_z"),
                star=gv(fi, "share_z") - gv(fj, "share_z"),
                outcome=0 if x.home_score > x.away_score else (1 if x.home_score == x.away_score else 2)))
    return pd.DataFrame(rows)


def fit(df, feats):
    X = df[feats].to_numpy(); y = df["outcome"].to_numpy()
    k = X.shape[1]; t0 = np.concatenate([np.zeros(k), [-1.0]]); t0[0] = 0.4
    r = minimize(sv._nll, t0, args=(X, y), method="L-BFGS-B", options={"maxiter": 500})
    return r.x[:k], np.log1p(np.exp(r.x[-1]))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    results = data.load_results()
    groups, bp, bgd, bgf, rem, wc = rf.derive_state(results)
    teams = sorted({t for g in groups.values() for t in g})

    beta, delta = fit(build_training(results), FEATS)
    print(f"coefficients: " + ", ".join(f"{f}={b:+.2f}" for f, b in zip(FEATS, beta)))

    model = elo.build_history(results, until=WC_START)
    f26 = squad_features(2026)
    R = np.array([model.get(t) for t in teams])
    sqv = np.array([f26.loc[t, "logtot_z"] if t in f26.index else 0.0 for t in teams])
    star = np.array([f26.loc[t, "share_z"] if t in f26.index else 0.0 for t in teams])
    hb = np.array([HOST_ETA.get(t, 0.0) for t in teams])

    nT = len(teams); pW = np.zeros((nT, nT)); pD = np.zeros((nT, nT)); pL = np.zeros((nT, nT))
    for a in range(nT):
        for b in range(nT):
            if a == b:
                continue
            eta = (beta[0] * (R[a] - R[b]) / 200 + beta[1] * (sqv[a] - sqv[b])
                   + beta[2] * (star[a] - star[b]) + hb[a] - hb[b])
            w = ndtr(eta - delta); l = ndtr(-eta - delta)
            pW[a, b] = w; pL[a, b] = l; pD[a, b] = max(1 - w - l, 1e-9)
    pw = simulate.Pairwise(teams, pW, pD, pL)

    df = simulate.simulate(groups, bp, bgd, bgf, rem, pw, n_sims=n, seed=2026)
    for c in simulate.ROUNDS:
        df[c] = (df[c] * 100).round(2)
    df = df.sort_values("win", ascending=False).reset_index(drop=True)
    print(f"\n=== FINAL forecast ({n:,} sims) ===")
    print(df.head(16).to_string(index=False))
    df.to_csv("results/forecast_star_experiment.csv", index=False)


if __name__ == "__main__":
    main()
