"""Produce the 2026 World Cup forecast: knockouts -> winner.

Conditions on real group results played so far (martj42 dataset) and simulates
the remainder with the volatility-aware (level + swing) model.

    python scripts/run_forecast.py [n_sims]
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wc2026 import data, simulate  # noqa: E402
from wc2026.models import volatility_strength as vs  # noqa: E402

WC_START = pd.Timestamp("2026-06-11")


def derive_state(results: pd.DataFrame):
    wc = results[(results.tournament == "FIFA World Cup")
                 & (results.date >= pd.Timestamp("2026-06-01"))].copy()

    # groups via connected components of the plays-against graph
    adj = collections.defaultdict(set)
    for r in wc.itertuples():
        adj[r.home_team].add(r.away_team); adj[r.away_team].add(r.home_team)
    seen, comps = set(), []
    for t in sorted(adj):
        if t in seen:
            continue
        comp, stack = set(), [t]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp.add(x); stack += list(adj[x] - seen)
        comps.append(sorted(comp))

    groups, base_pts, base_gd, base_gf, remaining = {}, {}, {}, {}, []
    for n, members in enumerate(comps):
        L = chr(65 + n)
        groups[L] = members
        kmap = {t: k for k, t in enumerate(members)}
        pts = [0] * len(members); gd = [0] * len(members); gf = [0] * len(members)
        played = wc[(wc.home_team.isin(members)) & (wc.home_score.notna())]
        for r in played.itertuples():
            h, a = int(r.home_score), int(r.away_score)
            ih, ia = kmap[r.home_team], kmap[r.away_team]
            gf[ih] += h; gf[ia] += a; gd[ih] += h - a; gd[ia] += a - h
            if h > a:
                pts[ih] += 3
            elif h < a:
                pts[ia] += 3
            else:
                pts[ih] += 1; pts[ia] += 1
        base_pts[L], base_gd[L], base_gf[L] = pts, gd, gf
        for r in wc[(wc.home_team.isin(members)) & (wc.home_score.isna())].itertuples():
            remaining.append((L, r.home_team, r.away_team))
    return groups, base_pts, base_gd, base_gf, remaining, wc


def main() -> None:
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    results = data.load_results()
    groups, base_pts, base_gd, base_gf, remaining, wc = derive_state(results)

    played = int(wc.home_score.notna().sum())
    print(f"Conditioning on {played}/{len(wc)} group games played "
          f"(through {wc[wc.home_score.notna()].date.max().date()}); "
          f"simulating {len(remaining)} remaining + knockouts, {n_sims:,} runs.\n")

    # fit the model at tournament start (no peeking into in-tournament results)
    fit = vs.fit(results, WC_START, pool_volatility=False)
    teams = sorted({t for g in groups.values() for t in g})
    pw = simulate.Pairwise.from_model(fit, teams)

    df = simulate.simulate(groups, base_pts, base_gd, base_gf, remaining,
                           pw, n_sims=n_sims)

    out = df.copy()
    for c in simulate.ROUNDS:
        out[c] = (out[c] * 100).round(1)
    pd.set_option("display.width", 120)
    print("=== 2026 World Cup forecast (probabilities %, level+swing model) ===")
    print(out.head(16).to_string(index=False))

    champ = df.iloc[0]
    print(f"\nMost likely winner: {champ['team']} ({champ['win']*100:.1f}%)")
    top4 = ", ".join(f"{r['team']} {r['win']*100:.1f}%" for _, r in df.head(4).iterrows())
    print(f"Top contenders: {top4}")

    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/forecast_2026.csv", index=False)
    print("\nWrote results/forecast_2026.csv")


if __name__ == "__main__":
    main()
