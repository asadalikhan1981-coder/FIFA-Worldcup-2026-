# 2026 World Cup — forecast & method (plain English)

*Locked one-shot forecast, dated 21 June 2026 (40 of 72 group games played),
**1,000,000** simulations. The public version is [POST.md](POST.md). Reproduce
with `python scripts/run_forecast.py 1000000`.*

## The prediction (knockouts → winner)

Each number is the model's probability, from 1,000,000 simulated tournaments that
**lock the games already played** and simulate everything left.

| Team | Reach R16 | Reach QF | Reach SF | Reach Final | **Win** |
|---|--:|--:|--:|--:|--:|
| Argentina | 75% | 53% | 36% | 25% | **16.3%** |
| Spain | 72% | 48% | 32% | 20% | **12.6%** |
| France | 64% | 38% | 22% | 12% | **6.7%** |
| Brazil | 64% | 37% | 21% | 12% | **6.2%** |
| Morocco | 61% | 36% | 20% | 11% | **5.5%** |
| Colombia | 60% | 34% | 18% | 10% | **4.9%** |
| Portugal | 49% | 29% | 16% | 9% | **4.6%** |
| Japan | 58% | 32% | 18% | 9% | **4.4%** |
| England | 60% | 33% | 17% | 9% | **4.2%** |
| Germany | 60% | 31% | 15% | 7% | **3.4%** |

**Headline:** Argentina most likely (16.3%), then Spain (12.6%), then a tight
chasing pack (France, Brazil, Morocco ~5–7%).

## How the method works (no jargon)

Standard forecasters (Elo, the bookmakers' models, Opta's "supercomputer") give
each team **one number** — how strong it is. Our model gives each team **two**:

1. **Level** — how strong, on average. (Same idea as everyone else.)
2. **Swing** — how *streaky* the team is. Some teams are metronomes (rarely have
   a shocker); some are coin-flips (brilliant one day, flat the next).

Why the second number matters: **everyone else silently assumes every team is
equally consistent.** That's wrong, and it's wrong exactly where tournaments are
decided — the upsets and the long unbeaten runs needed to win seven knockout
games in a row. A streaky team is both more likely to pull off a shock *and* more
likely to trip up. Modelling that "swing" is what makes us different.

The rest is honest plumbing: we learn both numbers from ~8 years of real
international results (recent games count more), then play the remaining 2026
tournament 20,000 times and count how often each team reaches each round.

## Is it actually *better*? Honest answer: not proven yet.

This is the part I won't dress up.

- **At the single-match level, the "swing" idea adds nothing measurable** (we
  tested it: it neither beat nor lost to the standard model — see README
  Findings). Swing only matters over a long tournament run, so a single game
  can't show it.
- **The real test — does our "win it all" number beat the bookmakers? — needs
  betting-odds data this environment can't reach.** It's wired and waiting.
- **Our forecast already disagrees with the market** in an interesting way: the
  market makes Spain favourite (~16%) and rates Argentina lower (~10–11%); we
  flip them. That disagreement is *either* an edge *or* our model being
  over-confident — and **we can't yet say which.** Our own calibration check
  shows the model is a little over-confident about strong favourites, so treat
  Argentina's 16.6% as "probably a touch high."

So: this is a **genuinely different method with a real, conditioned-on-results
2026 forecast** — but it has **not** earned the label "better than the standard
model or the market." That verdict waits on the odds/squad data.

## Caveats (so the numbers aren't over-read)

- **As of 21 June.** Re-run after each matchday to refresh.
- **Bracket averaged over valid draws.** The exact Round-of-32 pairings finalise
  after the group stage (June 27) via a fixed FIFA slot table we couldn't verify
  here, so we average over valid brackets. This barely moves the "win"
  numbers; it mostly affects who-meets-who paths.
- **Data:** results from the martj42 international-results dataset (GitHub);
  ratings computed transparently from those results. No betting tips, no
  unverified scrapes.

## What would make this trustworthy (next, needs your laptop data)

1. **Bookmaker closing/futures odds** → run the real test: are our longshot
   probabilities better calibrated than the market's? (the tradeable edge)
2. **Squad value / who's-injured** → feed availability into each team's *level*.

Until then the model is differentiated but unproven, and this report says so.
