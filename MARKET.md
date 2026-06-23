# Versus the market — the hard benchmark

The README states the project's honest position up front: **we do not expect to
beat bookmaker closing odds at the match level** — closing lines are
near-efficient and almost nothing public beats them there. This file is where
that claim is *tested with numbers* rather than asserted, plus a live read of
how the model disagrees with the 2026 outright market.

---

## 1. Match level — does the model beat the closing line? (No, it *matches* it.)

**Benchmark:** Pinnacle **closing** 1X2 odds for all **64 WC-2022 matches**
(the one past tournament with free public closing odds — see
[`data/odds/README.md`](data/odds/README.md)). Pinnacle is the sharpest book, so
its de-vigged closing price is, for practical purposes, *the* efficient-market
probability. This is the hardest honest bar there is.

Every forecast is scored on identical W/D/L outcomes. The squad-value blend is
fit on **WC-2018** and predicted on **WC-2022** — genuinely out-of-sample; the
2022 line never trained it.

| Forecast (n=64) | RPS | log-loss | Brier |
|---|--:|--:|--:|
| **MARKET — Pinnacle closing (de-vigged)** | **0.2079** | 1.0017 | 0.5846 |
| Baseline Elo → Dixon-Coles (control) | 0.2251 | 1.0899 | 0.6235 |
| Elo only (ordered probit, OOS) | 0.2162 | 1.0323 | 0.6093 |
| **Elo + squad value (OOS)** | **0.2102** | 1.0084 | 0.5912 |

**Paired ΔRPS vs the market** (negative = model beats market):

| Model | ΔRPS | t |
|---|--:|--:|
| Baseline Elo → Dixon-Coles | +0.0172 | +1.63 |
| Elo + squad value | **+0.0023** | **+0.35** |

### Read it honestly

- **The model does not beat the closing line** — exactly as the README predicted
  it wouldn't. The best model trails Pinnacle by **+0.0023 RPS**, a gap that is
  statistically indistinguishable from zero (t = 0.35).
- **But it essentially *matches* the sharpest book.** Closing to within 0.0023
  RPS of Pinnacle's *closing* price is a strong result — closing lines are the
  thing public models are not supposed to be able to touch.
- **Squad value is what closes the gap.** Plain Elo trails the market by a clear
  +0.0172 RPS (t = 1.63); adding the squad-value signal removes ~87% of that gap.
  The same signal that beat Elo out-of-sample also nearly erases the model's
  deficit to the market.
- **Group stage only** (48 games, no extra-time/penalty labelling ambiguity, the
  cleanest apples-to-apples slice): market 0.2253 vs Elo+squad **0.2258** —
  dead level.
- **Robustness:** de-vigging by the Shin method instead of proportional moves the
  market bar by 0.0009 (0.2079 → 0.2088); the conclusion is unchanged.

### Caveats (do not over-read one tournament)

- **One tournament, 64 matches.** Confidence intervals are wide; "matches the
  market" is the honest summary, not "beats" and not "clearly worse".
- **2018 closing odds were not freely available** (the upstream set has Pinnacle
  closing for 2022 only; no clean free 2018 source was found). A second
  tournament would tighten this. Drop a 2018 file into `data/odds/` with the same
  columns and `run_market_backtest.py` extends automatically.

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
