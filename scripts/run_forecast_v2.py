"""2026 forecast from the BLENDED model: Elo + squad market value.

The blend coefficients are fit on the 2018 + 2022 World Cups (the only editions
with squad data), then applied to 2026 using current Elo + 2026 squad values.
Prints the new forecast next to the Elo-only one so the effect of squad value is
visible.

    python scripts/run_forecast_v2.py [n_sims]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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


def fit_blend(df, feats):
    from scipy.optimize import minimize
    X = df[feats].to_numpy(); y = df["outcome"].to_numpy()
    k = X.shape[1]; theta0 = np.concatenate([np.zeros(k), [-1.0]]); theta0[0] = 0.4
    r = minimize(sv._nll, theta0, args=(X, y), method="L-BFGS-B", options={"maxiter": 500})
    return r.x[:k], np.log1p(np.exp(r.x[-1]))


def pairwise_from_blend(teams, R, sqz, beta, delta, use_squad):
    n = len(teams); pW = np.zeros((n, n)); pD = np.zeros((n, n)); pL = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            eta = beta[0] * (R[a] - R[b]) / 200.0
            if use_squad:
                eta += beta[1] * (sqz[a] - sqz[b])
            w = ndtr(eta - delta); l = ndtr(-eta - delta)
            pW[a, b] = w; pL[a, b] = l; pD[a, b] = max(1 - w - l, 1e-9)
    return simulate.Pairwise(teams, pW, pD, pL)


def forecast(teams, pw, groups, bp, bgd, bgf, rem, n):
    df = simulate.simulate(groups, bp, bgd, bgf, rem, pw, n_sims=n, seed=2026)
    return df.set_index("team")["win"] * 100


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    results = data.load_results()
    groups, bp, bgd, bgf, rem, wc = rf.derive_state(results)
    teams = sorted({t for g in groups.values() for t in g})

    # fit blend on 2018+2022
    bt = sv.build(results)
    beta_elo, d_elo = fit_blend(bt, ["elo"])
    beta_sv, d_sv = fit_blend(bt, ["elo", "sqval"])
    print(f"blend coefficients (fit on 2018+2022): elo-only elo={beta_elo[0]:+.2f} | "
          f"+squad elo={beta_sv[0]:+.2f}, squad={beta_sv[1]:+.2f}")

    # 2026 inputs
    model = elo.build_history(results, until=WC_START)
    R = np.array([model.get(t) for t in teams])
    sq2026 = sv.squad_strength(2026)
    sqz = np.array([sq2026.get(t, 0.0) for t in teams])

    pw_elo = pairwise_from_blend(teams, R, sqz, beta_elo, d_elo, use_squad=False)
    pw_blend = pairwise_from_blend(teams, R, sqz, beta_sv, d_sv, use_squad=True)

    win_elo = forecast(teams, pw_elo, groups, bp, bgd, bgf, rem, n)
    win_blend = forecast(teams, pw_blend, groups, bp, bgd, bgf, rem, n)

    out = pd.DataFrame({"Elo_only%": win_elo, "Elo+squad%": win_blend})
    out["change"] = out["Elo+squad%"] - out["Elo_only%"]
    out = out.sort_values("Elo+squad%", ascending=False).round(1)
    print(f"\n=== 2026 win probability: Elo-only vs Elo+squad value ({n:,} sims) ===")
    print(out.head(16).to_string())
    out.to_csv("results/forecast_blend.csv")
    print("\nwrote results/forecast_blend.csv")


if __name__ == "__main__":
    main()
