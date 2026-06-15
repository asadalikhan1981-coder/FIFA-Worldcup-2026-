"""Data adapters. The container has no egress except GitHub, so everything here
reads from local files in ``data/raw/`` (gathered on the user's laptop) or, for
the one reachable source, the martj42 international-results dataset on GitHub.

Keeping I/O behind these functions means the models never care where data came
from: drop in a Transfermarkt squad-value CSV or an odds export and the rest of
the pipeline is unchanged.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)


def load_results(path: str | Path | None = None) -> pd.DataFrame:
    """Load the international match-results table.

    Falls back to fetching from GitHub if the local copy is absent. Returns a
    frame with parsed dates and a real boolean ``neutral`` column, sorted by
    date.
    """
    path = Path(path) if path else RAW / "results.csv"
    src = str(path) if path.exists() else RESULTS_URL
    df = pd.read_csv(src)
    df["date"] = pd.to_datetime(df["date"])
    # The CSV stores neutral as the strings TRUE/FALSE.
    df["neutral"] = df["neutral"].map(
        lambda v: str(v).strip().lower() in ("true", "1", "yes")
    )
    return df.sort_values("date", kind="stable").reset_index(drop=True)


# Backtest tournaments. ``start`` freezes the priors (only matches strictly
# before it train the Elo); ``tournament`` + date window selects the games to
# score. These are the out-of-sample evaluation sets named in the handover.
BACKTEST_TOURNAMENTS = {
    "WC2018": dict(tournament="FIFA World Cup", start="2018-06-14", end="2018-07-16"),
    "WC2022": dict(tournament="FIFA World Cup", start="2022-11-20", end="2022-12-19"),
    "EURO2016": dict(tournament="UEFA Euro", start="2016-06-10", end="2016-07-11"),
    "EURO2020": dict(tournament="UEFA Euro", start="2021-06-11", end="2021-07-12"),
    "EURO2024": dict(tournament="UEFA Euro", start="2024-06-14", end="2024-07-15"),
}


def tournament_matches(results: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Select the games of one tournament edition, in date order."""
    start = pd.Timestamp(spec["start"])
    end = pd.Timestamp(spec["end"])
    mask = (
        (results["tournament"] == spec["tournament"])
        & (results["date"] >= start)
        & (results["date"] <= end)
    )
    return results[mask].sort_values("date", kind="stable").reset_index(drop=True)
