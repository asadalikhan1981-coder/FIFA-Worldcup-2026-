"""Bracket funnel + model decomposition + final-four sensitivity.

Three model-derived views that back BRACKET.md (all reproducible; the external
Opta/market numbers in that file are dated snapshots, not produced here):

  1. FUNNEL    - most-likely team to reach each round (R16 -> winner), read from
                 the canonical 1M forecast (results/forecast_final.csv).
  2. DECOMP    - title-win % under Elo-only / +squad / +squad+host, isolating
                 what each ingredient of the final model contributes.
  3. SENSITIVITY - reach-semifinal % for the contenders as the host advantage and
                 the squad-value coefficient are varied. Shows which of the last
                 four are robust and which (the USA) hinge on an assumption.

    python scripts/run_bracket_analysis.py [n_sims]   # default 200,000

Decomposition/sensitivity use n_sims each (200k is plenty to rank the top teams);
the canonical headline forecast stays at 1M via scripts/run_final_forecast.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wc2026 import data, elo, simulate  # noqa: E402


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    rf = _load("rf", "scripts/run_forecast.py")
    sv = _load("sv", "scripts/run_squad_value_test.py")
    fv = _load("fv", "scripts/run_forecast_v2.py")
    rff = _load("rff", "scripts/run_final_forecast.py")

    results = data.load_results()
    groups, bp, bgd, bgf, rem, _ = rf.derive_state(results)
    teams = sorted({t for g in groups.values() for t in g})
    beta, delta = fv.fit_blend(sv.build(results), ["elo", "sqval"])
    model = elo.build_history(results, until=fv.WC_START)
    R = np.array([model.get(t) for t in teams])
    sqz = np.array([sv.squad_strength(2026).get(t, 0.0) for t in teams])
    print(f"coef elo={beta[0]:+.3f} squad={beta[1]:+.3f} delta={delta:.3f}; {n:,} sims/scenario")

    def run(beta_use, host_scale):
        pw = rff.build_pairwise(teams, R, sqz, beta_use, delta, host_scale)
        return simulate.simulate(groups, bp, bgd, bgf, rem, pw, n_sims=n, seed=2026).set_index("team")

    # 1) Funnel from the canonical 1M forecast (fall back to this run if absent).
    fc_path = Path("results/forecast_final.csv")
    fc = pd.read_csv(fc_path).set_index("team") if fc_path.exists() else run(beta, 1.0) * 100
    print("\n=== 1. BRACKET FUNNEL (most-likely team per round) ===")
    rounds = [("Round of 16", "reach_R16", 16), ("Quarter-finals", "reach_QF", 8),
              ("Semi-finals", "reach_SF", 4), ("Final", "reach_final", 2), ("Winner", "win", 1)]
    for label, col, k in rounds:
        names = list(fc.sort_values(col, ascending=False).head(k).index)
        print(f"  {label:15} {', '.join(names)}")

    # 2) Component decomposition (win %).
    full = run(beta, 1.0)
    dec = pd.DataFrame({
        "Elo_only":   (run(np.array([beta[0], 0.0]), 0.0)["win"] * 100),
        "plus_squad": (run(beta, 0.0)["win"] * 100),
        "FULL":       (full["win"] * 100),
    }).loc[full.sort_values("win", ascending=False).head(12).index].round(1)
    print("\n=== 2. COMPONENT DECOMPOSITION (title-win %) ===")
    print(dec.to_string())

    # 3) Final-four sensitivity (reach-semifinal %).
    contenders = full.sort_values("reach_SF", ascending=False).head(7).index.tolist()
    scen = {
        "host=0.00": run(beta, 0.0), "host=0.20": run(beta, 0.20 / 0.33),
        "host=0.33*": full, "host=0.50": run(beta, 0.50 / 0.33),
        "squad=0": run(np.array([beta[0], 0.0]), 1.0),
        "squad x1.5": run(np.array([beta[0], beta[1] * 1.5]), 1.0),
    }
    sens = pd.DataFrame({k: (v.loc[contenders, "reach_SF"] * 100).round(1) for k, v in scen.items()})
    print("\n=== 3. FINAL-FOUR SENSITIVITY (reach semi-final %; *=base) ===")
    print(sens.to_string())


if __name__ == "__main__":
    main()
