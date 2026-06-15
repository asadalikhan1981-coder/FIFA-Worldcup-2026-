# Data shopping list (run on your laptop, where the network is open)

Drop files into `data/raw/`. Everything here is gitignored — it's reproducible
and may be licensed. The pipeline reads local files first and only falls back to
GitHub for the one always-available source.

## 1. International results — REQUIRED, already wired (auto-fetched)
`results.csv` from
`https://raw.githubusercontent.com/martj42/international_results/master/results.csv`
Columns: `date, home_team, away_team, home_score, away_score, tournament, city,
country, neutral`. The backtest fetches this automatically if absent. Also grab
`shootouts.csv` from the same repo for the knockout sim.

## 2. Closing odds — the real benchmark (the "hard bar")
This is the highest-value thing to collect, and the only piece that needs a paid
key or a manual export.

**Match closing odds** (1X2) for each backtested tournament, schema:
```
date, home_team, away_team, home_odds, draw_odds, away_odds   # decimal odds
```
**Tournament-winner futures** (the favourite-longshot exploit), schema:
```
tournament, asof_date, team, win_odds        # decimal; capture OPEN and CLOSE
```
Sources: the-odds-api.com (set `THE_ODDS_API_KEY`), oddsportal.com history,
football-data.co.uk. Name the files `odds_matches_<edition>.csv` and
`odds_futures_<edition>.csv`.

## 3. Squad market value / availability — the novel input
For the projected-XI strength signal. Two granularities, coarse is enough to start:
```
# coarse (per team, per tournament) — easiest, do this first
edition, team, squad_value_eur, n_key_absentees
```
```
# fine (per player) — better, if you can get it
edition, team, player, position, market_value_eur, available (bool)
```
Source: Transfermarkt (squad value at tournament start; mark injured/suspended
absentees). Name `squad_value_<edition>.csv`.

## 4. xG / player data — optional, only if we go state-space later
Per-team or per-player xG/xGA from FBref or Understat. Not needed for the
current control or the market-edge thesis; collect only if we pivot method.
Name `xg_<edition>.csv`.

---

### Naming so the adapters find them
`<edition>` uses the keys in `src/wc2026/data.py::BACKTEST_TOURNAMENTS`
(`WC2018`, `WC2022`, `EURO2016`, `EURO2020`, `EURO2024`). When these files
exist, the corresponding model/benchmark switches on automatically — no code
change needed.
