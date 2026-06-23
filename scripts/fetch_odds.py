"""Fetch historical World Cup match odds and write the committed odds tables.

The "beat the bookmakers" benchmark needs a bookmaker line for past World Cups.
Two free, public sources are combined, each the best freely available for its
tournament:

* **WC-2022 — Pinnacle CLOSING** 1X2 odds for all 64 matches, from
  ``eatpizzanot/soccer-dataset`` (``source = API-Football-closing``). Pinnacle is
  the sharpest book, so its de-vigged closing line is the hardest honest bar.

* **WC-2010 / 2014 / 2018 — AVERAGE pre-match** 1X2 odds from
  ``mrthlinh/FIFA-World-Cup-Prediction`` (``database_matches.csv``, average of
  several bookmakers). 2010 and 2014 are full (64 games each); the 2018 capture
  was frozen mid-tournament, so only its 48 group games are present. Average
  pre-match odds are a *softer* bar than a closing line — flagged in the output
  (``bookmaker`` column) and in MARKET.md.

Each line is re-oriented to the martj42 home/away ordering used everywhere else
in this repo. Outputs are small and committed, so the backtest reproduces
without the ~58 MB upstream downloads:

    data/odds/wc2022_pinnacle_closing.csv
    data/odds/wc2018_average_prematch.csv
    data/odds/wc2014_average_prematch.csv
    data/odds/wc2010_average_prematch.csv

    python scripts/fetch_odds.py

No clean *free* WC *closing* line before 2022 was found (the-odds-api historical
is paid; OddsPortal needs scraping), hence the average-odds fallbacks. Drop a
better file with the same columns here and run_market_backtest.py picks it up.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wc2026 import data  # noqa: E402

ODDS_DIR = Path(__file__).resolve().parents[1] / "data" / "odds"
COLUMNS = ["date", "home", "away", "home_score", "away_score",
           "odds_home", "odds_draw", "odds_away", "bookmaker", "source"]

# --- WC-2022: Pinnacle closing, eatpizzanot/soccer-dataset ---------------- #
EATP = "https://raw.githubusercontent.com/eatpizzanot/soccer-dataset/main/csv/"
WC_LEAGUE_ID = 78  # FIFA World Cup, per the upstream leagues table
# --- WC pre-2022: average pre-match, mrthlinh/FIFA-World-Cup-Prediction ---- #
MRTH = ("https://raw.githubusercontent.com/mrthlinh/FIFA-World-Cup-Prediction/"
        "master/data/database_matches.csv")
# Backtest windows for the average-odds editions (martj42 tournament dates).
AVG_EDITIONS = {  # year -> committed filename stem
    2010: "wc2010_average_prematch",
    2014: "wc2014_average_prematch",
    2018: "wc2018_average_prematch",
}

NAME_FIX = {"USA": "United States", "IR Iran": "Iran", "Korea Republic": "South Korea",
            "Bosnia & Herzegovina": "Bosnia and Herzegovina"}


# Backtest windows. 2018/2022 reuse the canonical specs; 2010/2014 are local.
SPECS = dict(data.BACKTEST_TOURNAMENTS)
SPECS["WC2010"] = dict(tournament="FIFA World Cup", start="2010-06-11", end="2010-07-11")
SPECS["WC2014"] = dict(tournament="FIFA World Cup", start="2014-06-12", end="2014-07-13")


def _orient(records: dict, results: pd.DataFrame, tournament: str,
            bookmaker: str, source: str) -> tuple[pd.DataFrame, list]:
    """Align a {(date, frozenset(teams)): (home_named, o_h, o_d, o_a)} mapping to
    the martj42 home/away ordering for the given backtest tournament."""
    games = data.tournament_matches(results, SPECS[tournament])
    games["d"] = games["date"].dt.date
    rows, missing = [], []
    for g in games.itertuples():
        key = (g.d, frozenset((g.home_team, g.away_team)))
        if key not in records:
            missing.append((g.d, g.home_team, g.away_team))
            continue
        oh, o_home, o_draw, o_away = records[key]
        odds_home, odds_draw, odds_away = (
            (o_home, o_draw, o_away) if oh == g.home_team else (o_away, o_draw, o_home)
        )
        rows.append(dict(
            date=g.date.date(), home=g.home_team, away=g.away_team,
            home_score=int(g.home_score), away_score=int(g.away_score),
            odds_home=odds_home, odds_draw=odds_draw, odds_away=odds_away,
            bookmaker=bookmaker, source=source,
        ))
    return pd.DataFrame(rows, columns=COLUMNS).sort_values("date"), missing


def fetch_wc2022(results: pd.DataFrame) -> None:
    print("WC-2022: downloading eatpizzanot fixtures/odds/teams (~58 MB) ...", flush=True)
    teams = pd.read_csv(EATP + "teams.csv")
    fixtures = pd.read_csv(EATP + "fixtures.csv")
    odds = pd.read_csv(EATP + "odds.csv")
    tn = dict(zip(teams["id"], teams["name"]))
    wc = fixtures[fixtures["league_id"] == WC_LEAGUE_ID].copy()
    wc["date"] = pd.to_datetime(wc["date"])
    wc22 = wc[wc["date"].dt.year == 2022].copy()
    wc22["ohome"] = wc22["home_team_id"].map(tn).replace(NAME_FIX)
    wc22["oaway"] = wc22["away_team_id"].map(tn).replace(NAME_FIX)
    pin = odds[odds["bookmaker"] == "Pinnacle"][["fixture_id", "home_win", "draw", "away_win"]]
    o = wc22.merge(pin, left_on="id", right_on="fixture_id", how="inner")
    o["d"] = o["date"].dt.date
    records = {(r.d, frozenset((r.ohome, r.oaway))): (r.ohome, r.home_win, r.draw, r.away_win)
               for r in o.itertuples()}
    out, missing = _orient(records, results, "WC2022", "Pinnacle", "API-Football-closing")
    if missing:
        print(f"  WARN {len(missing)} WC2022 games had no line:", missing)
    out.to_csv(ODDS_DIR / "wc2022_pinnacle_closing.csv", index=False)
    print(f"  wrote wc2022_pinnacle_closing.csv  rows={len(out)} (Pinnacle closing)")


def fetch_wc_average(df: pd.DataFrame, results: pd.DataFrame, year: int, stem: str) -> None:
    df = df.copy()
    df["dt"] = pd.to_datetime(df["match_date"], errors="coerce")
    w = df[(df["league"] == "World: FIFA World Cup") & (df["dt"].dt.year == year)]
    w = w.drop_duplicates(["team_1", "team_2", "dt"]).copy()  # exact dupes upstream
    w["t1"] = w["team_1"].replace(NAME_FIX)
    w["t2"] = w["team_2"].replace(NAME_FIX)
    w["d"] = w["dt"].dt.date
    records = {(r.d, frozenset((r.t1, r.t2))): (r.t1, r.avg_odds_win_1, r.avg_odds_draw, r.avg_odds_win_2)
               for r in w.itertuples()}
    out, missing = _orient(records, results, f"WC{year}", "Average", "mrthlinh/database_matches")
    note = f"{len(missing)} games absent upstream" if missing else "full tournament"
    print(f"  wrote {stem}.csv  rows={len(out)} (average pre-match; {note})")
    out.to_csv(ODDS_DIR / f"{stem}.csv", index=False)


def main() -> None:
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    results = data.load_results()
    fetch_wc2022(results)
    print("WC pre-2022: downloading mrthlinh database_matches.csv ...", flush=True)
    df = pd.read_csv(MRTH)
    df["league"] = df["league"].astype(str)
    for year, stem in AVG_EDITIONS.items():
        fetch_wc_average(df, results, year, stem)


if __name__ == "__main__":
    main()
