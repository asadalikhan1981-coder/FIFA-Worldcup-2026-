# World Cup 2026 — a *differentiated* forecasting model (or an honest "no edge")

**Hard rule for this project:** do not rebuild the standard pipeline (Elo →
Dixon-Coles Poisson → Monte-Carlo) and call it novel. That pipeline already
exists — Opta, the betting markets, FIFA's own ranking, and live open-source
repos all run it. Here it is built **only as the control**: the thing a genuinely
differentiated model has to *beat out-of-sample*, proven with a backtest number.
Different-but-not-better is failure, and if after real effort nothing beats the
baseline, this README will say "no edge found" in those words.

## The edge hypothesis (stated before any model is trusted)

We are **not** trying to out-forecast bookmaker closing odds at the *match*
level — closing lines are near-efficient and almost nothing public beats them
there; a model that claims to is overfit. The defensible edge lives where the
supercomputers structurally don't compete:

> The **tournament-winner futures market exhibits favourite-longshot bias** —
> the most robustly documented inefficiency in sports-betting economics — and a
> 48-team field is full of longshots. Bettors overpay for lottery tickets
> (a no-hoper "to win it all"), so the market's implied win-probabilities are
> too fat in the tails. A Monte-Carlo tournament simulation produces a full
> *distribution* of outcomes; if our simulated outright probabilities are
> **better calibrated in the longshot region** than the market's, then fading
> overpriced longshots (and concentrating on favourites) is positive-EV. The
> edge is **tail calibration, not the point forecast** — a tradeable edge, not a
> duplicate forecast.

The novel *input* that sharpens the distribution is **projected-XI /
availability-adjusted strength** (who is actually fit and starting), which static
Elo is blind to and thin international markets price lazily.

This is falsifiable two ways:
1. **RPS / Brier / log-loss vs the vanilla control** on past tournaments — the
   easy bar.
2. **Backtested ROI of "fade the longshots" vs closing futures odds** on
   2018/2022 WC and Euro 2020/24 — the real bar.

Early supporting signal: the control is already visibly **over-confident in the
favourite tail** at the match level (see `results/baseline_calibration.png`),
which is the match-level shadow of the same bias.

## Status (honest)

| Component | State |
|---|---|
| Vanilla Elo → Dixon-Coles match model (control) | ✅ built, tested |
| Out-of-sample match backtest (2018/22 WC, Euro 16/20/24) | ✅ runs on real data |
| Proper scoring: RPS, Brier, log-loss, calibration plot | ✅ |
| **Volatility-aware model (level + swing)** — candidate novel method | ✅ built + gradient-checked; **no match-level edge — see Findings** |
| Monte-Carlo bracket sim over the real 2026 draw | ✅ built + tested; conditions on live results |
| **2026 forecast (knockouts → winner)** | ✅ see [REPORT.md](REPORT.md) — Argentina 16.6%, Spain 12.5% (differentiated, **edge unproven**) |
| Projected-XI availability input | ⬜ needs squad/availability data (laptop) |
| Market tail-bias exploit + ROI backtest | ⬜ needs historical *outright* odds |
| **Benchmark vs bookmaker closing odds** | ✅ done — **matches Pinnacle closing, doesn't beat it** ([MARKET.md](MARKET.md)) |

### The number to beat — baseline, out-of-sample

```
edition   n     rps    brier  log_loss
WC2018    64   0.2112  0.5867  0.9921
WC2022    64   0.2251  0.6235  1.0899
EURO2016  51   0.2353  0.6705  1.1030
EURO2020  51   0.1739  0.5310  0.9117
EURO2024  51   0.2031  0.6384  1.0686
ALL      281   0.2105  0.6096  1.0338
ref:uniform     0.2357  ...    (1/3,1/3,1/3)
ref:climatology 0.2334  ...    (test-set marginal — generous reference)
```

The control beats the trivial references but sits a touch worse than a good
bookmaker (~0.19–0.20 RPS) — appropriately unimpressive for a control. Any
"novel" model that does not push the **ALL** RPS below **0.2105** out-of-sample
has earned nothing.

## Findings (honest, updated as we go)

**The volatility-aware "level + swing" model gives no edge at the match level.**
Nested, out-of-sample, 281 matches:

| Question | ΔRPS (paired) | t | Verdict |
|---|---|---|---|
| Swing ON vs OFF (the novelty) | −0.0002 ± 0.0006 | −0.27 | **no measurable edge** (noise) |
| New family vs Elo→Dixon-Coles | −0.0027 ± 0.0053 | −0.51 | better, **not significant** |

Two things this teaches us, both kept honestly on the record:

1. Match-level W/D/L is **mean-dominated**, so it is nearly blind to a *variance*
   innovation by construction. Swing only compounds over a 7-game tournament
   path, so its real test is **tournament outright probability vs market
   longshot prices** — pending odds data. Until then we claim nothing for it.
2. Adding model *flexibility* on results-only data moved the needle by t ≈ 0.
   This empirically confirms the "variance tax": a real edge must come from new
   **information** (availability / squad value / xG) or **market structure**
   (the longshot fade), not from fancier maths on the same results.

Run it: `python scripts/run_vol_backtest.py`.

**Novel reachable inputs don't beat plain Elo either.** A leave-one-tournament-out
loop (`scripts/run_edge_loop.py`) tested goal-timing "late-game strength" and
recent "form" — the only novel signals reachable from this sandbox:

| Model | OOS RPS | vs Elo (paired t) |
|---|--:|---|
| Elo only | 0.2073 | — |
| Elo + form | 0.2069 | −0.0004 (t=−0.28, NS) |
| Elo + late-strength | 0.2078 | worse; coefficient ≈ 0 |
| Elo + form + late | 0.2077 | worse |

Three results-only attempts (volatility, goal-timing, form) found no edge — the
bottleneck is *information*. So we searched public sources for a forward-looking
signal and found one that is **free and reachable**.

### ✅ Squad market value beats Elo out-of-sample (the edge)

Using free public **Transfermarkt squad values at kickoff** (`ericsanmiguel/football_elo`,
2018 + 2022 World Cups), a leave-one-tournament-out test
(`scripts/run_squad_value_test.py`):

| Model | OOS RPS | note |
|---|--:|---|
| Elo only | 0.2152 | the standard signal |
| **Squad value only** | **0.2046** | beats Elo *on its own* |
| **Elo + squad value** | **0.2039** | best; ΔRPS −0.0114 vs Elo (t=−1.49) |

Improves **both** tournaments, **robust to heavy regularization** (squad keeps ~3×
Elo's weight), and squad-value-alone already beats Elo. **Honest caveat:** only 2
tournaments / 128 matches have public squad data, so it is **directionally strong
but not yet statistically significant** (t=−1.49). Beating the *market* (closing
odds) is still untested.

**Final model = Elo + squad value.** On 2026 it corrects Elo's blind spots —
France/England/Germany (elite squads, lagged by results-Elo) rise toward the
market; Argentina (strong results, older squad) falls. See `scripts/run_final_forecast.py`.

### ➖ Versus the market: matches Pinnacle's closing line, does not beat it

The hard bar, now tested. On all **64 WC-2022 matches**, scored against
**Pinnacle closing** odds (the sharpest book; de-vigged), with the squad blend
fit on 2018 and predicted on 2022 (out-of-sample):

| Forecast | RPS | ΔRPS vs market (paired t) |
|---|--:|---|
| **Market — Pinnacle closing** | **0.2079** | — |
| Baseline Elo → Dixon-Coles | 0.2251 | +0.0172 (t=1.63) |
| **Elo + squad value (OOS)** | **0.2102** | +0.0023 (t=0.35) |

**No edge over the closing line — exactly as predicted above; closing odds are
near-efficient.** But the model *matches* the sharpest book to within 0.0023 RPS
(indistinguishable from zero), and **squad value removes ~87% of plain Elo's
deficit** to the market. Group-stage-only (cleanest slice): market 0.2253 vs
model 0.2258 — dead level. Honest caveat: one tournament, 64 matches, wide CIs;
2018 closing odds weren't freely available. Full write-up and the live outright
comparison (where the model backs **USA at 6% vs the market's ~2%**) in
[MARKET.md](MARKET.md). Run: `python scripts/fetch_odds.py && python scripts/run_market_backtest.py`.

## Why the network setup looks the way it does

This was developed in Claude Code's web sandbox. On the GitHub-only egress
policy, the historical-results backbone (every international since 1872), the
Transfermarkt squad values and now the **WC-2022 Pinnacle closing odds** are all
pulled from public GitHub mirrors — enough to stand up the control, the squad
edge *and* the bookmaker benchmark with zero paid feeds. (Live outright odds and
any 2018 historical odds need a wider egress policy or a manual drop into
`data/odds/`.)

## Run it

```bash
pip install -r requirements.txt
python scripts/run_baseline_backtest.py     # prints the table above, writes results/
python scripts/fetch_odds.py                 # pull WC-2022 Pinnacle closing odds
python scripts/run_market_backtest.py        # model vs the closing line
PYTHONPATH=src python -m pytest tests/ -q    # 25 tests
```

The backtest auto-fetches the results CSV from GitHub if `data/raw/results.csv`
is absent.

## Layout

```
src/wc2026/
  metrics.py       RPS / Brier / log-loss / calibration
  elo.py           eloratings.net build + shrunk in-tournament (K=8) update
  match_model.py   Dixon-Coles bivariate-Poisson -> W/D/L
  data.py          adapters (local CSV first, GitHub fallback)
  backtest.py      sequential, out-of-sample tournament backtest
  models/          (projected-XI and market-edge models land here)
scripts/           runnable entry points
tests/             unit tests for the maths
data/raw/          user-gathered data (gitignored)
```

## Anti-goals

No presenting the standard pipeline as original. No novelty claim without a
backtest number attached. No tuning on the test set. No 48-team ranking tables
or pretty brackets until the method earns its place.
