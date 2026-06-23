"""Beat the bookmakers? Score the model against Pinnacle's closing line.

This is the project's hardest benchmark. Pinnacle is the sharpest book; its
de-vigged *closing* price is, for practical purposes, the efficient-market
probability. Beating it out-of-sample is the strongest possible claim a
football model can make; *matching* it is already a strong result.

For every WC-2022 match (the one tournament with free closing odds, see
scripts/fetch_odds.py) we compare four forecasts on identical W/D/L outcomes:

  * MARKET      - Pinnacle closing 1X2, de-vigged (overround removed).
  * Baseline    - Elo -> Dixon-Coles, walk-forward (the control engine).
  * Elo (probit)- ordered probit on the Elo gap only, fit on WC-2018.
  * Elo+squad   - the edge model: probit on Elo gap + squad-value gap,
                  fit on WC-2018 and predicted on WC-2022. Genuinely
                  out-of-sample: the 2022 line never trained the blend.

Scored by RPS (primary), log-loss and Brier, with a paired t-test of per-match
RPS against the market. De-vig is proportional (normalise inverse odds); a Shin
de-vig is printed alongside as a robustness check.

    python scripts/run_market_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wc2026 import data, elo, metrics  # noqa: E402
from wc2026.backtest import backtest_tournament  # noqa: E402
from wc2026.market import devig_proportional, devig_shin  # noqa: E402

EPS = 1e-12
ODDS_CSV = Path("data/odds/wc2022_pinnacle_closing.csv")
# WC-2022 group stage ended 2022-12-02; later games are knockout (90-min 1X2,
# so a tie after 90 is a draw - matching martj42's regulation label).
GROUP_END = pd.Timestamp("2022-12-02")

NAME_FIX = {
    "South Korea": "South Korea", "Korea Republic": "South Korea", "IR Iran": "Iran",
    "USA": "United States", "Cabo Verde": "Cape Verde", "Czechia": "Czech Republic",
    "Türkiye": "Turkey", "Turkiye": "Turkey", "Curacao": "Curaçao",
}


# ------------------------- squad-value blend (OOS) ------------------------ #
def squad_strength(year: int) -> dict[str, float]:
    df = pd.read_csv(f"data/raw/squads/{year}.csv")
    df["team"] = df["team"].map(lambda t: NAME_FIX.get(t, t))
    tot = df.groupby("team")["value_at_kickoff"].sum()
    logv = np.log(tot.clip(lower=1.0))
    return ((logv - logv.mean()) / (logv.std(ddof=0) + 1e-9)).to_dict()


def feature_frame(results: pd.DataFrame, name: str) -> pd.DataFrame:
    spec = data.BACKTEST_TOURNAMENTS[name]
    model = elo.build_history(results, until=pd.Timestamp(spec["start"]))
    sv = squad_strength(int(name[-4:]))
    rows = []
    for g in data.tournament_matches(results, spec).itertuples(index=False):
        rows.append(dict(
            date=g.date, home=g.home_team, away=g.away_team,
            elo=(model.get(g.home_team) - model.get(g.away_team)) / 200.0,
            sqval=sv.get(g.home_team, 0.0) - sv.get(g.away_team, 0.0),
            outcome=0 if g.home_score > g.away_score
            else (1 if g.home_score == g.away_score else 2),
        ))
    return pd.DataFrame(rows)


def _nll(theta, X, y, ridge=1e-3):
    k = X.shape[1]
    beta, delta = theta[:k], np.log1p(np.exp(theta[-1]))
    eta = X @ beta
    pW, pL = ndtr(eta - delta), ndtr(-eta - delta)
    pD = np.clip(1 - pW - pL, EPS, None)
    p = np.where(y == 0, pW, np.where(y == 1, pD, pL))
    return -np.sum(np.log(np.clip(p, EPS, None))) + ridge * np.sum(beta * beta)


def fit_probit(Xtr, ytr, Xte):
    k = Xtr.shape[1]
    theta0 = np.concatenate([np.zeros(k), [-1.0]])
    theta0[0] = 0.5
    r = minimize(_nll, theta0, args=(Xtr, ytr), method="L-BFGS-B", options={"maxiter": 500})
    beta, delta = r.x[:k], np.log1p(np.exp(r.x[-1]))
    eta = Xte @ beta
    P = np.vstack([
        ndtr(eta - delta),
        np.clip(1 - ndtr(eta - delta) - ndtr(-eta - delta), EPS, None),
        ndtr(-eta - delta),
    ]).T
    return P / P.sum(axis=1, keepdims=True), beta


# ------------------------------ assembly ---------------------------------- #
def build() -> tuple[pd.DataFrame, np.ndarray]:
    if not ODDS_CSV.exists():
        raise SystemExit(f"Missing {ODDS_CSV}. Run: python scripts/fetch_odds.py")
    odds = pd.read_csv(ODDS_CSV, parse_dates=["date"])
    results = data.load_results()

    # Baseline Elo -> Dixon-Coles, walk-forward over WC2022.
    bt = backtest_tournament(results, data.BACKTEST_TOURNAMENTS["WC2022"])
    bkey = {(r.date.date(), frozenset((r.home, r.away))):
            ((r.p_home, r.p_draw, r.p_away), r.home) for r in bt.itertuples()}

    # Squad blend, fit on 2018, predicted on 2022 (out-of-sample).
    tr, te = feature_frame(results, "WC2018"), feature_frame(results, "WC2022")
    P_elo, _ = fit_probit(tr[["elo"]].to_numpy(), tr["outcome"].to_numpy(), te[["elo"]].to_numpy())
    P_sv, beta = fit_probit(tr[["elo", "sqval"]].to_numpy(), tr["outcome"].to_numpy(),
                            te[["elo", "sqval"]].to_numpy())
    skey = {(r.date.date(), frozenset((r.home, r.away))): (P_elo[i], P_sv[i], r.home)
            for i, r in enumerate(te.itertuples(index=False))}

    rows = []
    for r in odds.itertuples():
        key = (r.date.date(), frozenset((r.home, r.away)))
        mk_p = devig_proportional(r.odds_home, r.odds_draw, r.odds_away)
        mk_shin = devig_shin(r.odds_home, r.odds_draw, r.odds_away)
        (bp, bhome) = bkey[key]
        base = list(bp) if bhome == r.home else [bp[2], bp[1], bp[0]]
        peb, psv, shome = skey[key]
        elo_p = list(peb) if shome == r.home else [peb[2], peb[1], peb[0]]
        blend = list(psv) if shome == r.home else [psv[2], psv[1], psv[0]]
        y = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
        rows.append(dict(
            date=r.date, home=r.home, away=r.away, outcome=y,
            stage="group" if r.date <= GROUP_END else "knockout",
            market=mk_p, market_shin=mk_shin, baseline=base, elo=elo_p, blend=blend,
        ))
    return pd.DataFrame(rows), beta


def _rps(col, df, y):
    return metrics.ranked_probability_score(np.array(df[col].tolist()), y)


def main() -> None:
    df, beta = build()
    y = df["outcome"].to_numpy()
    n = len(df)
    print(f"\nMarket backtest: WC-2022, {n} matches, Pinnacle closing (de-vigged).")
    print(f"Squad blend fit on WC-2018 (out-of-sample): elo={beta[0]:+.2f}, squad={beta[1]:+.2f}\n")

    print(f"{'forecast':<26}{'RPS':>8}{'logloss':>9}{'brier':>8}")
    models = [
        ("MARKET (Pinnacle close)", "market"),
        ("Baseline Elo->Dixon-Coles", "baseline"),
        ("Elo only (probit, OOS)", "elo"),
        ("Elo + squad value (OOS)", "blend"),
    ]
    for label, col in models:
        P = np.array(df[col].tolist())
        print(f"{label:<26}{_rps(col, df, y).mean():>8.4f}"
              f"{metrics.log_loss(P, y).mean():>9.4f}{metrics.brier_score(P, y).mean():>8.4f}")

    rps_mkt = _rps("market", df, y)
    print("\nPaired ΔRPS vs MARKET  (negative = model beats market):")
    for label, col in [("Baseline", "baseline"), ("Elo+squad", "blend")]:
        d = _rps(col, df, y) - rps_mkt
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {label:<11} Δ={d.mean():+.4f}  t={d.mean() / se:+.2f}")

    print(f"\nMarket RPS by de-vig method: proportional={rps_mkt.mean():.4f}, "
          f"Shin={_rps('market_shin', df, y).mean():.4f}")

    print("\nGroup stage only (no extra-time/penalty label ambiguity):")
    g = df[df.stage == "group"]
    yg = g["outcome"].to_numpy()
    for label, col in [("MARKET", "market"), ("Baseline", "baseline"), ("Elo+squad", "blend")]:
        print(f"  {label:<11} RPS={metrics.ranked_probability_score(np.array(g[col].tolist()), yg).mean():.4f}"
              f"  (n={len(g)})")

    best = min(models[1:], key=lambda m: _rps(m[1], df, y).mean())
    gap = _rps(best[1], df, y).mean() - rps_mkt.mean()
    print(f"\nVERDICT: best model = {best[0]} (RPS {_rps(best[1], df, y).mean():.4f}). "
          f"Market = {rps_mkt.mean():.4f}.")
    if gap < 0:
        print("  -> model BEATS the closing line (check significance above).")
    else:
        print(f"  -> model does NOT beat the market; gap {gap:+.4f} RPS. "
              "Squad value closes most of the gap to the sharpest book.")


if __name__ == "__main__":
    main()
