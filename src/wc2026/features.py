"""Novel, results-reachable team features the standard Elo/Poisson models ignore.

The supercomputers see only the final score. The goal-timing data lets us build
signals about *how* a team gets its results:

- ``late_strength`` : recency-weighted net goals scored minus conceded in the
  last 15 minutes (>=75'). A proxy for fitness / squad depth / closing
  mentality — qualities that matter most in knockout football (late winners,
  extra time) and are invisible to a final-score model.
- ``form`` : recency-weighted average goal difference. Recent form *beyond* what
  the slow-moving Elo has absorbed.

Both are computed strictly from matches before a cutoff date and z-scored across
teams, so they slot into an out-of-sample backtest without leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _recency_weight(dates: pd.Series, cutoff: pd.Timestamp, halflife_days: float):
    age = (cutoff - dates).dt.days.to_numpy()
    return 0.5 ** (age / halflife_days)


def late_goal_table(results: pd.DataFrame, goalscorers: pd.DataFrame) -> pd.DataFrame:
    """Per-team, per-match late (>=75') goals for and against."""
    g = goalscorers.copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g[g["minute"].notna()]
    g["late"] = (g["minute"] >= 75).astype(int)
    # late goals credited to the scoring team per match
    key = ["date", "home_team", "away_team", "team"]
    late_for = g.groupby(key, as_index=False)["late"].sum().rename(columns={"late": "late_gf"})

    rows = []
    for r in results.itertuples(index=False):
        rows.append((r.date, r.home_team, r.away_team, r.home_team))
        rows.append((r.date, r.home_team, r.away_team, r.away_team))
    base = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "team"])
    m = base.merge(late_for, on=key, how="left").fillna({"late_gf": 0})

    # opponent's late goals = late against
    opp = m.copy()
    opp["opp"] = np.where(opp["team"] == opp["home_team"], opp["away_team"], opp["home_team"])
    ga = late_for.rename(columns={"team": "opp", "late_gf": "late_ga"})
    m = opp.merge(ga, on=["date", "home_team", "away_team", "opp"], how="left").fillna({"late_ga": 0})
    m["late_net"] = m["late_gf"] - m["late_ga"]
    return m[["date", "team", "late_net"]]


def compute_features(
    results: pd.DataFrame,
    goalscorers: pd.DataFrame,
    cutoff: pd.Timestamp,
    window_years: float = 8.0,
    halflife_days: float = 730.0,
    min_matches: int = 5,
) -> pd.DataFrame:
    """Return per-team z-scored features as of ``cutoff`` (no leakage)."""
    start = cutoff - pd.Timedelta(days=int(window_years * 365.25))
    res = results[(results["date"] < cutoff) & (results["date"] >= start)].copy()

    # form = recency-weighted avg goal difference (from each team's perspective)
    rows = []
    for r in res.itertuples(index=False):
        rows.append((r.date, r.home_team, r.home_score - r.away_score))
        rows.append((r.date, r.away_team, r.away_score - r.home_score))
    gd = pd.DataFrame(rows, columns=["date", "team", "gd"])
    gd["w"] = _recency_weight(gd["date"], cutoff, halflife_days)

    late = late_goal_table(res, goalscorers)
    late = late[(late["date"] < cutoff) & (late["date"] >= start)].copy()
    late["w"] = _recency_weight(late["date"], cutoff, halflife_days)

    teams = sorted(set(gd["team"]))
    out = []
    for t in teams:
        sub = gd[gd["team"] == t]
        if len(sub) < min_matches:
            continue
        form = np.average(sub["gd"], weights=sub["w"])
        ls = late[late["team"] == t]
        late_str = np.average(ls["late_net"], weights=ls["w"]) if len(ls) else 0.0
        out.append((t, form, late_str, len(sub)))
    df = pd.DataFrame(out, columns=["team", "form_raw", "late_raw", "n"])
    # z-score across teams
    for col in ["form_raw", "late_raw"]:
        z = (df[col] - df[col].mean()) / (df[col].std(ddof=0) + 1e-9)
        df[col.replace("_raw", "_z")] = z
    return df.set_index("team")
