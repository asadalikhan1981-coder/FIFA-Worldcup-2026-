# 2026 World Cup — predicted knockout bracket & model comparison

*Model: Elo + squad market value + host advantage. 1,000,000 simulations,
conditioned on group results through 21 June 2026. Reproduce the model tables
with `python scripts/run_bracket_analysis.py`; the headline forecast is
`scripts/run_final_forecast.py`.*

The model is probabilistic — it plays the tournament a million times — so this is
the **most-likely team to reach each round**, not a single fixed bracket (exact
Round-of-16 slots lock on 27 June). % = chance of reaching that round.

## 1. The knockout path — Round of 16 → Winner

**🏆 Winner — 🇫🇷 France (11.1%)**

**Final** · 🇫🇷 France 18.3% · 🇪🇸 Spain 16.9%

**Semi-finals — the last 4** · 🇫🇷 France 29.5% · 🇪🇸 Spain 27.7% · 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England 26.9% · 🇦🇷 Argentina 21.8%

**Quarter-finals — the last 8** · + 🇩🇪 Germany 38.2% · 🇧🇷 Brazil 37.9% · 🇺🇸 USA 37.7% · 🇳🇱 Netherlands 35.9%

**Round of 16 — the 16 knockout teams** (by chance of reaching it):

| # | Team | R16 | # | Team | R16 |
|--:|---|--:|--:|---|--:|
| 1 | 🇫🇷 France | 71% | 9 | 🇲🇽 Mexico | 61% |
| 2 | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 70% | 10 | 🇳🇴 Norway | 57% |
| 3 | 🇪🇸 Spain | 70% | 11 | 🇵🇹 Portugal | 57% |
| 4 | 🇩🇪 Germany | 67% | 12 | 🇲🇦 Morocco | 56% |
| 5 | 🇦🇷 Argentina | 66% | 13 | 🇨🇦 Canada | 54% |
| 6 | 🇺🇸 USA | 66% | 14 | 🇨🇭 Switzerland | 51% |
| 7 | 🇧🇷 Brazil | 65% | 15 | 🇨🇴 Colombia | 51% |
| 8 | 🇳🇱 Netherlands | 64% | 16 | 🇨🇮 Ivory Coast | 50% |

## 2. My model vs other models (title-win %)

| Team | **My model** | Opta* | Market† |
|---|--:|--:|--:|
| 🇫🇷 France | **11.1** | 13.0 | 20.4 |
| 🇪🇸 Spain | **10.1** | 16.1 | 16.7 |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | **9.5** | 11.2 | 14.3 |
| 🇦🇷 Argentina | **6.5** | 10.4 | 12.5 |
| 🇧🇷 Brazil | **6.4** | 6.6 | 7.1 |
| 🇩🇪 Germany | **6.1** | 5.1 | 7.7 |
| 🇺🇸 USA | **6.0** | 1.2 | 2.9 |
| 🇵🇹 Portugal | **6.0** | 7.0 | 7.7 |
| 🇳🇱 Netherlands | **5.2** | 3.6 | 5.9 |
| 🇲🇽 Mexico | **3.7** | 1.0 | 2.2 |

\* **Opta supercomputer**, *pre-tournament* (1 Jun 2026, 25k sims, before any
games) — [theanalyst.com](https://theanalyst.com/articles/who-will-win-2026-fifa-world-cup-predictions-opta-supercomputer).
† **FanDuel** outright, 22 Jun 2026, raw implied from American odds. Raw market is
inflated by the book's overround (these 10 already sum to ~97%), so de-vigged the
favourites are ~30% lower — e.g. France ~15%, not 20%.

Some of the gap to Opta is *timing* (Opta is pre-tournament; mine and the market
are conditioned on the group games played). The USA gap is structural, not timing.

## 3. What's different in my model

Decomposing the final model shows what each ingredient adds (title-win %, same sim):

| Team | Elo only *(standard pipeline)* | + squad value | + host *(FULL)* |
|---|--:|--:|--:|
| 🇫🇷 France | 4.2 | 11.7 | 11.1 |
| 🇪🇸 Spain | 5.0 | 11.0 | 10.1 |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 4.0 | 10.2 | 9.5 |
| 🇺🇸 USA | 2.7 | 2.0 | **5.9** |
| 🇲🇽 Mexico | 3.4 | 1.1 | **3.8** |

1. **Squad value creates the top tier.** Plain Elo — what most public base models
   reduce to — is *flat*: every contender sits at 4–5%. The squad-value signal
   (the only thing that beat Elo out-of-sample) lifts France/Spain/England to
   ~10–11% and gives the model a spine.
2. **Host advantage makes the USA a dark horse.** Squad value alone would *bury*
   the hosts (USA 2.0%, Mexico 1.1%) — their squads are cheap. The data-calibrated
   host bump rescues them to 5.9% / 3.8%. This is the model's biggest disagreement
   with everyone else: **USA 6.0% vs Opta 1.2% and market ~2%** — the live,
   falsifiable bet.
3. **Flatter tails than the market.** My favourites top out ~10–11%; Opta and the
   bookmakers pile onto Spain/France at 13–20%. The model spreads probability
   across a deeper field — consistent with the project's thesis that the market
   over-concentrates on favourites.

## 4. Sensitivity of the final four

How robust are the last four to the two assumptions that matter — **host
advantage** (probit units; base 0.33) and the **squad-value coefficient** (base
0.35)? Reach-semifinal %, 200k sims each:

| Team | host=0 | host=0.20 | **host=0.33** | host=0.50 | squad=0 | squad×1.5 |
|---|--:|--:|--:|--:|--:|--:|
| 🇫🇷 France | 30.7 | 30.1 | **29.6** | 28.7 | 13.9 | 36.6 |
| 🇪🇸 Spain | 29.1 | 28.3 | **27.6** | 26.8 | 15.4 | 32.6 |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 28.3 | 27.5 | **27.0** | 25.9 | 13.3 | 32.7 |
| 🇦🇷 Argentina | 22.9 | 22.3 | **21.7** | 20.8 | 14.7 | 23.6 |
| 🇺🇸 USA *(challenger)* | 11.0 | 16.4 | **20.7** | 27.1 | 21.8 | 18.5 |

- **The top three are rock-solid.** France/Spain/England barely move with the host
  bump (±2 pts) — they're carried by squad value, not host effects. Turning squad
  value *off* collapses the whole field toward ~13–15% (the flat Elo world),
  confirming squad value is the load-bearing signal.
- **The fourth semifinal spot is the contested one.** Argentina holds it at base
  settings (21.7%), but **the USA is the swing case**: invisible at host=0 (11%),
  level with the Argentina/Brazil/Germany pack at the base 0.33 (20.7%), and
  **into the final four at host=0.50 (27.1%)**. The whole "USA dark horse" story
  lives or dies on the host-advantage assumption — exactly the parameter the
  bookmaker test (see [MARKET.md](MARKET.md)) can't yet settle.
- Squad value actually *hurts* the USA (squad=0 → USA SF 21.8%): Transfermarkt
  underprices their talent relative to their Elo+host, so the squad term docks them.
