# Historical World Cup match odds

Committed 1X2 odds used by `scripts/run_market_backtest.py` to benchmark the
model against the market. Unlike `data/raw/` (gitignored, re-fetched each
session), these small derived files are **committed** so the backtest reproduces
without the large upstream downloads. Regenerate both with:

```bash
python scripts/fetch_odds.py
```

| File | Tournament | Odds | n | Bar |
|---|---|---|--:|---|
| `wc2022_pinnacle_closing.csv` | WC-2022 (all) | **Pinnacle closing** | 64 | hard (sharpest book, closing price) |
| `wc2018_average_prematch.csv` | WC-2018 (group stage) | **average pre-match** | 48 | softer (averaged books, not closing) |

## Columns (both files)

| column | meaning |
|---|---|
| `date`, `home`, `away` | match, with teams named per martj42/international_results |
| `home_score`, `away_score` | regulation/ET result (penalty shootouts count as draws) |
| `odds_home`, `odds_draw`, `odds_away` | decimal odds, **oriented to the `home`/`away` columns above** |
| `bookmaker` | `Pinnacle` (2022) or `Average` (2018) |
| `source` | upstream provenance tag |

De-vigging (removing the overround) happens at scoring time in
`src/wc2026/market.py`; these files store the raw quoted odds.

## Sources

- **WC-2022** — [`eatpizzanot/soccer-dataset`](https://github.com/eatpizzanot/soccer-dataset)
  (CC-licensed), API-Football **closing** 1X2 prices, Pinnacle.
- **WC-2018** — [`mrthlinh/FIFA-World-Cup-Prediction`](https://github.com/mrthlinh/FIFA-World-Cup-Prediction)
  (`data/database_matches.csv`), **average** of several bookmakers' pre-match 1X2.

## Coverage notes (honest)

The two bars differ in sharpness — read them separately, not pooled. Average
pre-match odds (2018) are **softer** than a closing line (2022), so beating 2018
is the weaker result.

**Why no WC-2018 closing line, and no knockouts?** The sharp/closing free source
(eatpizzanot) has no 2018 odds at all; `the-odds-api`'s historical endpoint is
paid and OddsPortal needs scraping. The free 2018 fallback used here was frozen
**mid-tournament**, so only the 48 group games are present. To improve this, drop
a `wc2018_*_closing.csv` (or any tournament) here with the same columns — the
backtest fits leave-one-tournament-out and picks up whatever is present.
