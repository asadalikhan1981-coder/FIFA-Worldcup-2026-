"""Fetch historical World Cup closing odds and write the committed odds table.

The "beat the bookmakers" benchmark needs a bookmaker line for past World Cups.
The free, public ``eatpizzanot/soccer-dataset`` carries **Pinnacle closing**
1X2 odds (``source = API-Football-closing``) for every WC-2022 match. Pinnacle
is the sharpest book, so its de-vigged closing line is the hardest honest bar.

This script downloads that dataset's ``fixtures``/``odds``/``teams`` tables,
keeps the FIFA World Cup (league_id 78) games, orients each line to the
martj42 home/away ordering used everywhere else in this repo, and writes the
small, self-contained ``data/odds/wc2022_pinnacle_closing.csv`` (committed, so
the backtest reproduces without the ~58 MB upstream download).

    python scripts/fetch_odds.py

NOTE: 2018 is intentionally absent. The upstream set has no closing odds for
WC-2018, and no clean *free* source was found. The market backtest is therefore
WC-2022 only; see README. Drop a 2018 file with the same columns here to extend
it. The squad-value blend evaluated on 2022 is fit on 2018, so the test stays
out-of-sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wc2026 import data  # noqa: E402

BASE = "https://raw.githubusercontent.com/eatpizzanot/soccer-dataset/main/csv/"
WC_LEAGUE_ID = 78  # FIFA World Cup, per the upstream leagues table
# upstream team name -> martj42 results.csv name
NAME_FIX = {"USA": "United States", "IR Iran": "Iran"}
OUT = Path(__file__).resolve().parents[1] / "data" / "odds" / "wc2022_pinnacle_closing.csv"


def main() -> None:
    print("downloading upstream fixtures/odds/teams (~58 MB) ...", flush=True)
    teams = pd.read_csv(BASE + "teams.csv")
    fixtures = pd.read_csv(BASE + "fixtures.csv")
    odds = pd.read_csv(BASE + "odds.csv")

    tn = dict(zip(teams["id"], teams["name"]))
    wc = fixtures[fixtures["league_id"] == WC_LEAGUE_ID].copy()
    wc["date"] = pd.to_datetime(wc["date"])
    wc22 = wc[wc["date"].dt.year == 2022].copy()
    wc22["ohome"] = wc22["home_team_id"].map(tn).replace(NAME_FIX)
    wc22["oaway"] = wc22["away_team_id"].map(tn).replace(NAME_FIX)

    pin = odds[odds["bookmaker"] == "Pinnacle"][
        ["fixture_id", "home_win", "draw", "away_win", "source"]
    ]
    o = wc22.merge(pin, left_on="id", right_on="fixture_id", how="inner")
    o["d"] = o["date"].dt.date
    upstream = {
        (r.d, frozenset((r.ohome, r.oaway))): (r.ohome, r.home_win, r.draw, r.away_win, r.source)
        for r in o.itertuples()
    }

    # Align every line to the martj42 home/away ordering used by the model.
    results = data.load_results()
    games = data.tournament_matches(results, data.BACKTEST_TOURNAMENTS["WC2022"])
    games["d"] = games["date"].dt.date

    rows, missing = [], []
    for g in games.itertuples():
        key = (g.d, frozenset((g.home_team, g.away_team)))
        if key not in upstream:
            missing.append((g.d, g.home_team, g.away_team))
            continue
        oh, h, draw, a, src = upstream[key]
        odds_home, odds_draw, odds_away = (h, draw, a) if oh == g.home_team else (a, draw, h)
        rows.append(dict(
            date=g.date.date(), home=g.home_team, away=g.away_team,
            home_score=int(g.home_score), away_score=int(g.away_score),
            odds_home=odds_home, odds_draw=odds_draw, odds_away=odds_away,
            bookmaker="Pinnacle", source=src,
        ))

    if missing:
        print(f"WARN {len(missing)} WC2022 games had no Pinnacle line:", missing)
    out = pd.DataFrame(rows).sort_values("date")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(OUT.parents[2])}  rows={len(out)}")


if __name__ == "__main__":
    main()
