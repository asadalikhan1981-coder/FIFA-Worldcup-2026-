# Versus the market — the hard benchmark

The README states the project's honest position up front: **we do not expect to
beat bookmaker closing odds at the match level** — closing lines are
near-efficient and almost nothing public beats them there. This file is where
that claim is *tested with numbers* rather than asserted, plus a live read of
how the model disagrees with the 2026 outright market.

---

## 1. Match level — does the model beat the line? (It sits *at market level*.)

**Benchmark:** de-vigged 1X2 odds for two World Cups, each the best freely
available for its edition (see [`data/odds/README.md`](data/odds/README.md)).
The two bars are **different sharpness and not pooled naively**:

| Tournament | Odds | Bar | n |
|---|---|---|--:|
| **WC-2022** | Pinnacle **closing** | the *hard* bar (sharpest book, closing price ≈ efficient market) | 64 |
| **WC-2018** | **average** pre-match (group stage only) | a *softer* bar — easier to beat | 48 |

Every forecast is scored on identical W/D/L outcomes. Both probit models are fit
**leave-one-tournament-out** (each tournament predicted from the other), so every
number is genuinely out-of-sample — the line being scored never trained the model.

| Forecast | WC-2022 RPS *(vs closing)* | WC-2018 RPS *(vs average)* |
|---|--:|--:|
| **MARKET (de-vigged)** | **0.2079** | **0.1963** |
| Baseline Elo → Dixon-Coles | 0.2251 | 0.2026 |
| Elo only (probit, OOS) | 0.2162 | 0.2091 |
| **Elo + squad value (OOS)** | **0.2102** | **0.1933** |

**Paired ΔRPS vs the market** (negative = model beats market):

| Model | WC-2022 (closing) | WC-2018 (average) |
|---|--:|--:|
| Baseline Elo → Dixon-Coles | +0.0172 (t=1.63) | +0.0063 (t=0.47) |
| **Elo + squad value** | **+0.0023 (t=0.35)** | **−0.0030 (t=−0.27)** |

### Read it honestly

- **The model sits right at market level.** Against the **sharp** 2022 closing
  line it trails by a hair (+0.0023 RPS, t = 0.35 — indistinguishable from zero);
  against the **softer** 2018 average line it edges ahead (−0.0030, t = −0.27).
  The sign of the gap just tracks how sharp the specific line is. No *proven* edge
  over a closing price — exactly as the README predicted — but it is not beaten by
  one either.
- **Squad value is what gets it there.** In **both** tournaments plain Elo clearly
  trails the market and the squad-value signal closes the gap — the same signal
  that beat Elo out-of-sample erases the deficit to the line.
- **Pooled group stage** (96 games, clean W/D/L, mixed bars): model 0.2095 vs
  market 0.2108 — level (Δ −0.0013, t = −0.19). Pooling mixes a soft and a sharp
  line, so read the per-tournament rows as the real result.
- **Robustness:** Shin de-vig instead of proportional moves each market bar by
  <0.001; conclusions unchanged.

### Caveats (don't over-read)

- **Two tournaments, 112 matches**, wide confidence intervals. "At market level"
  is the honest summary — not "beats", not "clearly worse".
- **Mixed benchmarks.** 2018 is *average pre-match* odds (no free 2018 *closing*
  line was found) and *group stage only* (the upstream set was frozen
  mid-tournament). Average odds are softer than closing, so the 2018 "win" is the
  weaker of the two results; the 2022 closing comparison is the meaningful bar.
- Drop a better/longer odds file into `data/odds/` with the same columns and
  `run_market_backtest.py` extends automatically.

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
