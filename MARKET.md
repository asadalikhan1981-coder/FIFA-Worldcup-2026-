# Versus the market — the hard benchmark

The README states the project's honest position up front: **we do not expect to
beat bookmaker closing odds at the match level** — closing lines are
near-efficient and almost nothing public beats them there. This file is where
that claim is *tested with numbers* rather than asserted, plus a live read of
how the model disagrees with the 2026 outright market.

---

## 1. Match level — does the model beat the line? (Depends how sharp the line is.)

**Benchmark:** de-vigged 1X2 odds for **four** World Cups, each the best freely
available for its edition (see [`data/odds/README.md`](data/odds/README.md)).
Crucially the bars differ in **sharpness**, and that turns out to be the whole story:

| Tournament | Odds | Bar | n | Squad data? |
|---|---|---|--:|:--:|
| **WC-2022** | Pinnacle **closing** | *hard* (sharpest book, closing ≈ efficient market) | 64 | ✅ |
| WC-2018 | **average** pre-match (group only) | soft | 48 | ✅ |
| WC-2014 | average pre-match | soft | 63 | ❌ |
| WC-2010 | average pre-match | soft | 64 | ❌ |

Forecasts are scored on identical W/D/L outcomes. The squad blend (and its
Elo-only probit twin) is fit **leave-one-tournament-out** over the editions with
validated squad values (2018/2022) → fully out-of-sample. 2010/2014 have no free
squad values (see §1b), so they test the **control** (Elo → Dixon-Coles) vs market.

| Forecast | WC-2022 *(closing, sharp)* | WC-2018 *(avg)* | WC-2014 *(avg)* | WC-2010 *(avg)* |
|---|--:|--:|--:|--:|
| **MARKET (de-vigged)** | **0.2079** | **0.1963** | **0.1945** | **0.1989** |
| Baseline Elo → Dixon-Coles | 0.2251 | 0.2026 | **0.1915** | **0.1916** |
| Elo + squad value (OOS) | **0.2102** | **0.1933** | — | — |

**Paired ΔRPS vs the market** (negative = model beats market):

| Model | WC-2022 (closing) | WC-2018 | WC-2014 | WC-2010 |
|---|--:|--:|--:|--:|
| Baseline Elo → DC | +0.0172 (t=1.63) | +0.0063 (t=0.47) | **−0.0030 (t=−0.30)** | **−0.0073 (t=−0.72)** |
| **Elo + squad value** | **+0.0023 (t=0.35)** | **−0.0030 (t=−0.27)** | — | — |

### Read it honestly

- **It depends entirely on how sharp the line is.**
  - Against the **soft** average lines (2010/2014/2018) the model is **level or
    better** — plain Elo even *beats* the 2010 and 2014 average prices. None of
    these gaps is significant (all |t| < 1), so "roughly level" is the fair call.
  - Against the one **sharp** line — Pinnacle **closing** 2022 — plain Elo clearly
    **trails** (+0.0172 RPS, t = 1.63), and it takes **squad value** to claw back
    to market level (+0.0023, t = 0.35, indistinguishable from zero).
- **No proven edge over a closing price** — exactly as the README predicted — but
  the model is not beaten by one either, and it *does* beat soft bookmaker lines.
- **Squad value is the difference-maker** at the hard bar: it removes ~87% of
  plain Elo's deficit to the 2022 closing line, the same signal that beat Elo
  out-of-sample.
- **Robustness:** Shin de-vig instead of proportional moves each market bar by
  <0.001; conclusions unchanged.

### Caveats (don't over-read)

- **239 matches across 4 editions, wide CIs.** Only WC-2022 uses a true *closing*
  line; the other three use *average pre-match* odds, which are softer (easier to
  beat) — so the soft-line "wins" are the weaker results. The 2022 closing
  comparison is the meaningful bar.
- WC-2018 is group stage only (the upstream capture was frozen mid-tournament);
  WC-2014 is missing 1 game upstream.
- Drop a better/longer odds file into `data/odds/` with the same columns and
  `run_market_backtest.py` extends automatically.

### 1b. Why the squad edge stops at 2018/2022 (an honest dead-end)

The obvious way to make the squad edge *significant* is more tournaments. I tried
to reconstruct 2010/2014 squad values for free and **deliberately did not ship
it**, because the data isn't clean enough:

- **Value source — validated.** Transfermarkt market-value histories from the free
  `salimt/football-datasets` reproduce the project's trusted kickoff values almost
  exactly: joined by Transfermarkt **player_id**, 2018 per-team totals match the
  `ericsanmiguel` source at ratio **~1.00**.
- **Rosters — too noisy.** Free 2010/2014 squad *rosters* exist only as **names**
  (no ids), so they need name-matching to recover player_ids. That matches ~82% of
  players (96% correct when matched), but it **breaks on transliteration-heavy
  squads** (e.g. Saudi Arabia 2018: 0 of 12 matched). Reconstructed team totals
  rank-correlate 0.99 with truth but the *level* correlation is only ~0.69 — too
  much measurement error to stake the headline edge on.

Manufacturing a noisy squad feature and calling it an extended edge would violate
this project's "no fake precision" rule, so 2010/2014 stay **control-only**. A
clean extension needs an **id-based roster source** (e.g. the Kaggle
`dcaribou/transfermarkt-datasets` national-team lineups — gated behind a login).

Reproduce: `python scripts/fetch_odds.py` then `python scripts/run_market_backtest.py`.

---

## 2. Outright market — where the 2026 model disagrees (the live, falsifiable bet)

Champion odds, **FanDuel, 22 June 2026** (mid-group-stage), vs the model's
1M-sim forecast. Raw implied % is shown as quoted; an outright book over a
48-team field carries a large overround, so the market column sums to well over
100% and its **absolute** levels are inflated by roughly a third — compare
*shape and direction*, not raw magnitudes.

| Team | Market odds | Market implied (raw) | **Model** | Model vs market |
|---|--:|--:|--:|:--|
| 🇫🇷 France | +390 | 20.4% | 11.1% | model **lower** (market top-heavy) |
| 🇪🇸 Spain | +500 | 16.7% | 10.1% | model lower |
| 🏴 England | +600 | 14.3% | 9.5% | model lower |
| 🇦🇷 Argentina | +700 | 12.5% | 6.5% | model lower |
| 🇵🇹 Portugal | +1200 | 7.7% | 5.9% | ~in line |
| 🇩🇪 Germany | +1200 | 7.7% | 6.1% | ~in line |
| 🇧🇷 Brazil | +1300 | 7.1% | 6.4% | ~in line |
| 🇳🇱 Netherlands | +1600 | 5.9% | 5.2% | ~in line |
| 🇺🇸 **USA** *(host)* | +3300 | 2.9% | **6.0%** | model **much higher** ⬅ the bet |
| 🇲🇽 Mexico *(host)* | +4500 | 2.2% | 3.7% | model higher |

Two structural disagreements, both flowing from the same modelling choices:

1. **The market is far more top-heavy.** It concentrates probability on the four
   favourites (France alone ~15% even after de-vigging); the model spreads it
   more evenly across a deeper field. This is the match-level "matches but doesn't
   beat" picture projected onto the tournament: similar information, flatter tails.
2. **USA is the live, falsifiable call.** The model says **6.0%**, the market
   ~2% even after de-vig — a 3× gap driven entirely by the data-calibrated host
   advantage. If the host bump is right, the USA is badly underpriced; if it is
   too aggressive, this is where the model is wrong. Either way it is a clean,
   checkable bet, settled on the pitch.

*Live odds drift; this is a 22 June 2026 snapshot. Re-pull before quoting.*
