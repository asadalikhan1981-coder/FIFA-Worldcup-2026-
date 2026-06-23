"""FINAL model forecast: Elo + squad market value blend, applied to 2026.

This is the model the backtest selected (squad value beats Elo out-of-sample).
Coefficients are fit on 2018+2022; inputs are 2026 Elo + 2026 squad values.

    python scripts/run_final_forecast.py [n_sims]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.special import ndtr  # noqa: E402

from wc2026 import data, elo, simulate  # noqa: E402

rf = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("rf", "scripts/run_forecast.py"))
sys.modules["rf"] = rf; rf.__loader__.exec_module(rf)
sv = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("sv", "scripts/run_squad_value_test.py"))
sys.modules["sv"] = sv; sv.__loader__.exec_module(sv)
fv = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("fv", "scripts/run_forecast_v2.py"))
sys.modules["fv"] = fv; fv.__loader__.exec_module(fv)

# Host advantage (probit units). General home advantage is ~0.33; WC hosts have
# historically done even better (~0.57). Team-specific per the crowd reality:
#   USA    - full, plays in the US throughout
#   Mexico - full, de facto home ANYWHERE in North America (~75% pro-Mexico
#            crowds even on US soil; Mexico routinely plays "home" games in the US)
#   Canada - genuine home in the group stage, but ~neutral on US soil in the
#            knockouts (no comparable US fanbase), so a reduced central value
HOST_ETA = {"United States": 0.33, "Mexico": 0.33, "Canada": 0.20}


def build_pairwise(teams, R, sqz, beta, delta, scale=1.0):
    n = len(teams)
    hb = np.array([scale * HOST_ETA.get(t, 0.0) for t in teams])
    pW = np.zeros((n, n)); pD = np.zeros((n, n)); pL = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            eta = beta[0] * (R[a] - R[b]) / 200 + beta[1] * (sqz[a] - sqz[b]) + hb[a] - hb[b]
            w = ndtr(eta - delta); l = ndtr(-eta - delta)
            pW[a, b] = w; pL[a, b] = l; pD[a, b] = max(1 - w - l, 1e-9)
    return simulate.Pairwise(teams, pW, pD, pL)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    results = data.load_results()
    groups, bp, bgd, bgf, rem, wc = rf.derive_state(results)
    teams = sorted({t for g in groups.values() for t in g})

    bt = sv.build(results)
    beta, delta = fv.fit_blend(bt, ["elo", "sqval"])
    model = elo.build_history(results, until=fv.WC_START)
    R = np.array([model.get(t) for t in teams])
    sqz = np.array([sv.squad_strength(2026).get(t, 0.0) for t in teams])
    pw = build_pairwise(teams, R, sqz, beta, delta, scale)

    print(f"FINAL model (Elo+squad+host), {n:,} sims. "
          f"coef elo={beta[0]:+.2f} squad={beta[1]:+.2f} host(US/MX={0.33*scale:.2f}, CA={0.20*scale:.2f})")
    df = simulate.simulate(groups, bp, bgd, bgf, rem, pw, n_sims=n, seed=2026)
    for c in simulate.ROUNDS:
        df[c] = (df[c] * 100).round(2)
    df = df.sort_values("win", ascending=False).reset_index(drop=True)
    print(df.head(20).to_string(index=False))
    df.to_csv("results/forecast_final.csv", index=False)
    print("DONE_FINAL -> results/forecast_final.csv")


if __name__ == "__main__":
    main()
