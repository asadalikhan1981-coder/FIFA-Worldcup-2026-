"""Fetch the always-available backbone datasets into data/raw/.

Run on a machine with network access. The other datasets (odds, squad value,
xG) are gathered manually — see data/README.md.

    python scripts/fetch_data.py
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = {
    "results.csv": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
    "shootouts.csv": "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv",
    "goalscorers.csv": "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv",
}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
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
