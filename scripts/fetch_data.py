"""Fetch the always-available backbone datasets into data/raw/.

Run on a machine with network access. The other datasets (odds, squad value,
xG) are gathered manually — see data/README.md.

    python scripts/fetch_data.py
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
M = "https://raw.githubusercontent.com/martj42/international_results/master/"
FE = "https://raw.githubusercontent.com/ericsanmiguel/football_elo/main/data/squads/"
FILES = {
    "results.csv": M + "results.csv",
    "shootouts.csv": M + "shootouts.csv",
    "goalscorers.csv": M + "goalscorers.csv",
    # free, public squad market values at tournament kickoff (Transfermarkt)
    "squads/2018.csv": FE + "2018.csv",
    "squads/2022.csv": FE + "2022.csv",
    "squads/2026.csv": FE + "2026.csv",
    # free historical FIFA ranking points
    "fifa_ranking.csv": "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/ranking_fifa_historical.csv",
}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "squads").mkdir(exist_ok=True)
    for name, url in FILES.items():
        dest = RAW / name
        print(f"fetching {name} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"ok ({dest.stat().st_size:,} bytes)")
        except Exception as exc:  # pragma: no cover
            print(f"FAILED: {exc}")


if __name__ == "__main__":
    main()
