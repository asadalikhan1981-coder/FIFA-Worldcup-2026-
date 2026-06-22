# I built a different kind of World Cup forecast. Here it is — locked.

*Pre-registered forecast. Dated 21 June 2026. 1,000,000 simulations, conditioned
on every group game played so far.*

## The idea (in one sentence)

Most World Cup models — Elo, the betting markets, Opta's "supercomputer" — boil
each team down to **one number**: how strong it is. Mine uses **two**: strength
*and* **volatility** — how streaky a team is from match to match.

Why the second number matters: a World Cup isn't won by the best team on paper,
it's won by the team that strings seven good days together across a month of
knockouts. Every standard model silently assumes all teams are equally
consistent. They aren't — and that assumption is wrong exactly where tournaments
are decided: the upsets and the deep runs.

## The forecast

| Team | Reach R16 | QF | SF | Final | **Win the Cup** |
|---|--:|--:|--:|--:|--:|
| 🇦🇷 Argentina | 75% | 53% | 36% | 25% | **16.3%** |
| 🇪🇸 Spain | 72% | 48% | 32% | 20% | **12.6%** |
| 🇫🇷 France | 64% | 38% | 22% | 12% | **6.7%** |
| 🇧🇷 Brazil | 64% | 37% | 21% | 12% | **6.2%** |
| 🇲🇦 Morocco | 61% | 36% | 20% | 11% | **5.5%** |
| 🇨🇴 Colombia | 60% | 34% | 18% | 10% | **4.9%** |
| 🇵🇹 Portugal | 49% | 29% | 16% | 9% | **4.6%** |
| 🇯🇵 Japan | 58% | 32% | 18% | 9% | **4.4%** |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 60% | 33% | 17% | 9% | **4.2%** |
| 🇩🇪 Germany | 60% | 31% | 15% | 7% | **3.4%** |

**Most likely champion: Argentina (16.3%)**, then Spain (12.6%), then a tight
chasing pack.

## Where I disagree with the market — my calls, named up front

The betting market right now makes **France** the favourite (~21%) and rates
**Argentina** 4th (~12%). My model **flips it**: Argentina first, and I sit well
*below* the market on France (~7%) and England (~4%). Those are my bets, stated
before the knockouts — not explained away after.

## The honest part — which is the entire point

This is a **pre-registered** forecast: dated, locked, public. When the Cup ends,
**grade it** — Brier score and Ranked Probability Score against the bookmakers'
closing odds. If the two-number method beats the market, it earned the label. If
it doesn't, I'll post that too. The test is fixed in advance; nothing gets
cherry-picked or quietly revised afterwards.

A forecast you can't be scored on is a horoscope. This one you can.

---

*Built transparently from public international-match results (no black box).
Method, code, scoring metrics and backtests in the repo. The two-number model is
genuinely different from the standard approach; whether it's genuinely **better**
is the open question this pre-registered test is designed to settle.*
