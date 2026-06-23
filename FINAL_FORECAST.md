# 2026 World Cup — FINAL forecast

*One-shot, locked. As of 21 June 2026 (40 of 72 group games played), 1,000,000
simulations. Reproduce: `python scripts/run_final_forecast.py 1000000`.*

## The forecast — Round of 16 → Winner

| # | Team | Reach R16 | QF | SF | Final | **Win** |
|--:|---|--:|--:|--:|--:|--:|
| 1 | 🇫🇷 France | 71% | 46% | 30% | 18% | **11.1%** |
| 2 | 🇪🇸 Spain | 70% | 45% | 28% | 17% | **10.1%** |
| 3 | 🏴 England | 70% | 44% | 27% | 16% | **9.5%** |
| 4 | 🇦🇷 Argentina | 66% | 39% | 22% | 12% | **6.5%** |
| 5 | 🇧🇷 Brazil | 65% | 38% | 21% | 12% | **6.4%** |
| 6 | 🇩🇪 Germany | 67% | 38% | 21% | 11% | **6.1%** |
| 7 | 🇺🇸 USA *(host)* | 66% | 38% | 21% | 11% | **6.0%** |
| 8 | 🇵🇹 Portugal | 57% | 34% | 20% | 11% | **5.9%** |
| 9 | 🇳🇱 Netherlands | 64% | 36% | 19% | 10% | **5.2%** |
| 10 | 🇲🇽 Mexico *(host)* | 61% | 32% | 16% | 8% | **3.7%** |
| 11 | 🇳🇴 Norway | 57% | 31% | 16% | 8% | **3.6%** |
| 12 | 🇲🇦 Morocco | 56% | 29% | 15% | 7% | **3.1%** |

**Headline:** A tight three-way lead — **France (11%), Spain (10%), England (9.5%)** — then a chasing pack around 6%, in which **the USA is a genuine host dark horse**.

## How it works (plain English)

Three ingredients, each earning its place:

1. **Strength (Elo)** — the standard "who's good based on results" rating. The baseline everyone uses.
2. **Squad market value** — what the players are actually worth *today*. This is the differentiator: a forward-looking talent signal that results-based ratings lag. **It is the only thing we tested that beat the standard model out-of-sample.** It's why France/England/Germany sit above Argentina here, where a pure-results model had them too low.
3. **Host advantage** — calibrated from data (World Cup hosts win 61% of home games, +0.91 goal difference). Applied team-by-team: **USA and Mexico get the full boost** (Mexico is de facto home anywhere in North America — crowds are ~75% pro-Mexico even on US soil); **Canada gets a reduced boost** (home in its group, ~neutral on US soil later).

## How we got here

We tried and **rejected** several "clever" ideas because they failed an honest out-of-sample backtest: a volatility/"swing" method, goal-timing (late-game strength), and recent form — **none beat plain Elo**. The lesson: edges come from *information*, not fancier maths. We then searched public data and found one signal that *does* win: **squad market value**.

## Honest status

- ✅ **Beats the standard Elo model out-of-sample** (RPS 0.2039 vs 0.2152) — squad value even beats Elo on its own, and the result is robust to stress-testing. This is real and new.
- ⚠️ **Not yet statistically bulletproof** — only 2 World Cups (128 matches) have free public squad data, so it's directionally strong but not significant (t = −1.49).
- ➖ **Tested against the bookmakers — matches, doesn't beat.** On all 64 WC-2022 matches vs **Pinnacle closing** odds (the sharpest book), the model scores **RPS 0.2102 vs the market's 0.2079** — statistically level (t = 0.35), and squad value removes ~87% of plain Elo's deficit to the line. No edge over near-efficient closing odds (as expected), but it *matches* them. See [MARKET.md](MARKET.md).
- Simulation count (1M) buys *precision*, not credibility — the numbers are converged.

## Data — all free & public

Match results & 2026 fixtures (martj42/international_results), squad market values
at kickoff (ericsanmiguel/football_elo, Transfermarkt), FIFA rankings
(Dato-Futbol). No paid feeds, no scraping, no betting tips. Reproduce everything
with `python scripts/fetch_data.py` then `python scripts/run_final_forecast.py`.
