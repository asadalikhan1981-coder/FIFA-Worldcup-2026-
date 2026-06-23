# Historical World Cup closing odds

`wc2022_pinnacle_closing.csv` — **Pinnacle closing** 1X2 decimal odds for all 64
matches of the 2022 FIFA World Cup, used by `scripts/run_market_backtest.py` to
benchmark the model against the sharpest book's efficient-market price.

Unlike `data/raw/` (gitignored, re-fetched each session), this small derived
file is **committed** so the market backtest reproduces without the ~58 MB
upstream download.

## Columns

| column | meaning |
|---|---|
| `date`, `home`, `away` | match, with teams named per martj42/international_results |
| `home_score`, `away_score` | regulation/ET result (penalty shootouts count as draws) |
| `odds_home`, `odds_draw`, `odds_away` | Pinnacle closing decimal odds, **oriented to the `home`/`away` columns above** |
| `bookmaker` | `Pinnacle` |
| `source` | `API-Football-closing` (upstream provenance tag) |

De-vigging (removing the overround) happens at scoring time in
`src/wc2026/market.py`; this file stores the raw quoted odds.

## Source & reproduction

Pulled from the free, public **`eatpizzanot/soccer-dataset`** (CC-licensed),
which carries API-Football closing 1X2 prices. Regenerate with:

```bash
python scripts/fetch_odds.py
```

That script downloads the upstream `fixtures`/`odds`/`teams` tables, keeps the
FIFA World Cup games, and re-orients every line to this repo's home/away
convention.

## Why 2022 only

The upstream set has **no closing odds for WC-2018**, and no clean *free* 2018
source was found (the-odds-api's historical endpoint is paid; OddsPortal needs
scraping). The market backtest is therefore WC-2022 only. This stays
out-of-sample: the squad-value blend scored on 2022 is fit on 2018. To extend to
a second tournament, drop a `wc2018_*_closing.csv` here with the same columns and
the backtest picks it up.
