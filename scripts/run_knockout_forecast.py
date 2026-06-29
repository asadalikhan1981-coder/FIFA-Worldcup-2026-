"""In-tournament knockout forecast, conditioned on the REAL Round-of-32 bracket.

By the second round the group stage is over and the bracket is set, so the
group-stage simulator (scripts/run_final_forecast.py, which draws *hypothetical*
brackets) is superseded. This script reads the actual 16 R32 fixtures from the
martj42 data, locks any knockout games already played, and Monte-Carlos the fixed
single-elimination tree forward to the title with the FINAL model
(Elo + squad value + host advantage).

    python scripts/run_knockout_forecast.py [n_sims]   # default 1,000,000

Bracket-tree caveat: the data gives the R32 *matchups* but not the official
R16+ slot tree, so we assume consecutive R32 fixtures (in schedule order) feed
each R16 tie. The R32 matchups themselves are real; only the later pairing order
is assumed.
"""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wc2026 import data, elo, simulate  # noqa: E402

ROUNDS = ["reach_R16", "reach_QF", "reach_SF", "reach_final", "win"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def knockout_fixtures(results: pd.DataFrame):
    """Return the knockout games in schedule order as (home, away, played_winner).

    A WC-2026 game is 'knockout' once either side has already played its 3 group
    games — robust to date ordering.
    """
    wc = results[(results.tournament == "FIFA World Cup")
                 & (results.date >= pd.Timestamp("2026-06-01"))].sort_values("date")
    seen = Counter()
    fixtures = []
    for g in wc.itertuples():
        ko = seen[g.home_team] >= 3 or seen[g.away_team] >= 3
        seen[g.home_team] += 1; seen[g.away_team] += 1
        if not ko:
            continue
        winner = None
        if pd.notna(g.home_score):
            winner = g.home_team if g.home_score >= g.away_score else g.away_team
        fixtures.append((g.home_team, g.away_team, winner))
    return fixtures


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    rf = _load("rf", "scripts/run_forecast.py")
    sv = _load("sv", "scripts/run_squad_value_test.py")
    fv = _load("fv", "scripts/run_forecast_v2.py")
    rff = _load("rff", "scripts/run_final_forecast.py")

    results = data.load_results()
    fixtures = knockout_fixtures(results)
    if len(fixtures) != 16:
        print(f"WARNING: expected 16 R32 fixtures, found {len(fixtures)}.")
    teams = sorted({t for f in fixtures for t in f[:2]})

    beta, delta = fv.fit_blend(sv.build(results), ["elo", "sqval"])
    model = elo.build_history(results, until=fv.WC_START)
    R = np.array([model.get(t) for t in teams])
    sqz = np.array([sv.squad_strength(2026).get(t, 0.0) for t in teams])
    pw = rff.build_pairwise(teams, R, sqz, beta, delta, scale=1.0)
    idx = pw.idx

    played = sum(f[2] is not None for f in fixtures)
    print(f"Knockout forecast: real R32 bracket, {played}/16 games played, {n:,} sims.")
    print(f"coef elo={beta[0]:+.2f} squad={beta[1]:+.2f} host(US/MX=0.33, CA=0.20)\n")

    # index-space bracket; locked winner index or None
    bracket = [(idx[a], idx[b], idx[w] if w else None) for a, b, w in fixtures]
    rng = np.random.default_rng(2026)
    counts = {t: dict.fromkeys(ROUNDS, 0) for t in teams}

    for _ in range(n):
        advancing = []
        for a, b, locked in bracket:
            advancing.append(locked if locked is not None
                             else (a if rng.random() < pw.pWin_nd[a, b] else b))
        for t in advancing:
            counts[teams[t]]["reach_R16"] += 1
        for rname in ROUNDS[1:]:
            advancing = [(_a if rng.random() < pw.pWin_nd[_a, _b] else _b)
                         for _a, _b in zip(advancing[::2], advancing[1::2])]
            for t in advancing:
                counts[teams[t]][rname] += 1

    df = pd.DataFrame([{"team": t, **{r: counts[t][r] / n * 100 for r in ROUNDS}} for t in teams])
    df = df.sort_values("win", ascending=False).reset_index(drop=True).round(2)
    print(df.to_string(index=False))
    df.to_csv("results/forecast_knockout.csv", index=False)
    print("\nDONE -> results/forecast_knockout.csv")


if __name__ == "__main__":
    main()
