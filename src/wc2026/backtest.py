"""Out-of-sample match-outcome backtest for the baseline (control) model.

For each tournament we:
  1. Freeze Elo priors using only matches strictly before the start date.
  2. Walk the tournament's games in date order. For each game, predict
     [home, draw, away] from the *current* ratings BEFORE seeing the result,
     then apply the shrunk (K=8) in-tournament update.

Step 2 is genuinely sequential and out-of-sample: a game is always scored with
information available before kick-off, and within-tournament results only ever
inform *later* games. Knockout ties settled on penalties count as draws (the
regulation/ET result), which is the correct label for a W/D/L forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, elo, metrics
from .match_model import MatchModelParams, outcome_probs


def outcome_label(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def backtest_tournament(
    results: pd.DataFrame,
    spec: dict,
    params: MatchModelParams | None = None,
    in_tournament_update: bool = True,
) -> pd.DataFrame:
    """Return a per-match frame of predictions and outcomes for one edition."""
    params = params or MatchModelParams()
    start = pd.Timestamp(spec["start"])
    model = elo.build_history(results, until=start)
    games = data.tournament_matches(results, spec)

    rows = []
    for g in games.itertuples(index=False):
        r_i, r_j = model.get(g.home_team), model.get(g.away_team)
        probs = outcome_probs(r_i, r_j, neutral=bool(g.neutral), p=params)
        y = outcome_label(g.home_score, g.away_score)
        rows.append({
            "date": g.date, "home": g.home_team, "away": g.away_team,
            "r_home": r_i, "r_away": r_j,
            "p_home": probs[0], "p_draw": probs[1], "p_away": probs[2],
            "home_score": g.home_score, "away_score": g.away_score,
            "outcome": y,
        })
        if in_tournament_update:
            elo.shrunk_update(
                model, g.home_team, g.away_team,
                g.home_score, g.away_score, neutral=bool(g.neutral),
            )
    return pd.DataFrame(rows)


def score_frame(df: pd.DataFrame) -> dict[str, float]:
    probs = df[["p_home", "p_draw", "p_away"]].to_numpy()
    return metrics.summary(probs, df["outcome"].to_numpy())


def run_all(
    results: pd.DataFrame,
    tournaments: dict | None = None,
    params: MatchModelParams | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest every configured tournament; return (per-match, per-tournament)."""
    tournaments = tournaments or data.BACKTEST_TOURNAMENTS
    all_matches, summaries = [], []
    for name, spec in tournaments.items():
        df = backtest_tournament(results, spec, params=params)
        if df.empty:
            continue
        df.insert(0, "edition", name)
        all_matches.append(df)
        s = score_frame(df)
        s["edition"] = name
        summaries.append(s)
    per_match = pd.concat(all_matches, ignore_index=True)
    # Pooled "ALL" row scored over every match together.
    pooled = score_frame(per_match)
    pooled["edition"] = "ALL"
    summaries.append(pooled)
    per_tourn = pd.DataFrame(summaries)[["edition", "n", "rps", "brier", "log_loss"]]
    return per_match, per_tourn
