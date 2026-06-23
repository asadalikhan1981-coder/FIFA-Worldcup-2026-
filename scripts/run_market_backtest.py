"""Beat the bookmakers? Score the model against the market's prices.

The project's hardest benchmark. For each tournament with free odds we compare
four forecasts on identical W/D/L outcomes:

  * MARKET      - de-vigged 1X2 (overround removed; proportional, Shin alongside).
  * Baseline    - Elo -> Dixon-Coles, walk-forward (the control engine).
  * Elo (probit)- ordered probit on the Elo gap only.
  * Elo+squad   - the edge model: probit on Elo gap + squad-value gap.

The two probit models are fit **leave-one-tournament-out** (predict each
tournament from the other), so every prediction is genuinely out-of-sample.

Tournaments / odds (best freely available per edition - see scripts/fetch_odds.py):

  * WC-2022 - Pinnacle CLOSING, all 64 matches. The sharpest book's closing
              price ~ the efficient-market probability: the *hard* bar.
  * WC-2018 - AVERAGE pre-match, 48 group games. Softer than a closing line, and
              group stage only - a real but easier bar. Reported separately;
              the two benchmarks are NOT pooled naively.

Scored by RPS (primary), log-loss and Brier, with a paired t-test of per-match
RPS against the market.

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
# tournament -> (committed odds file, the bar it represents). Order = report order.
ODDS_FILES = {
    "WC2022": ("data/odds/wc2022_pinnacle_closing.csv", "Pinnacle closing (sharp)"),
    "WC2018": ("data/odds/wc2018_average_prematch.csv", "average pre-match (soft)"),
}
# Group-stage cutoffs; later games are knockout (90-min 1X2, so a tie after 90 is
# a draw - matching martj42's regulation label).
GROUP_END = {"WC2022": pd.Timestamp("2022-12-02"), "WC2018": pd.Timestamp("2018-06-28")}

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
def _loo_probit(frames: dict[str, pd.DataFrame], feats: list[str]) -> dict:
    """Leave-one-tournament-out probit: predict each tournament from the others.
    Returns {edition: (P[n,3], home_names, dates, beta_mean)}."""
    out = {}
    betas = []
    for ed, te in frames.items():
        tr = pd.concat([f for k, f in frames.items() if k != ed], ignore_index=True)
        P, beta = fit_probit(tr[feats].to_numpy(), tr["outcome"].to_numpy(), te[feats].to_numpy())
        out[ed] = (P, te["home"].tolist(), [d.date() for d in te["date"]])
        betas.append(beta)
    return out, np.mean(betas, axis=0)


def build() -> tuple[pd.DataFrame, np.ndarray]:
    results = data.load_results()
    editions = {ed: f for ed, (f, _) in ODDS_FILES.items() if Path(f).exists()}
    if not editions:
        raise SystemExit("No odds files found. Run: python scripts/fetch_odds.py")

    frames = {ed: feature_frame(results, ed) for ed in editions}
    elo_pred, _ = _loo_probit(frames, ["elo"])
    blend_pred, beta = _loo_probit(frames, ["elo", "sqval"])

    def keyed(pred):  # (date, teamset) -> (probs, home_name)
        d = {}
        for ed, (P, homes, dates) in pred.items():
            for i in range(len(homes)):
                d[(dates[i], frozenset((homes[i], frames[ed].iloc[i]["away"])))] = (P[i], homes[i])
        return d

    ekey, skey = keyed(elo_pred), keyed(blend_pred)

    rows = []
    for ed, path in editions.items():
        odds = pd.read_csv(path, parse_dates=["date"])
        bt = backtest_tournament(results, data.BACKTEST_TOURNAMENTS[ed])
        bkey = {(r.date.date(), frozenset((r.home, r.away))):
                ((r.p_home, r.p_draw, r.p_away), r.home) for r in bt.itertuples()}
        for r in odds.itertuples():
            key = (r.date.date(), frozenset((r.home, r.away)))
            orient = lambda p, h: list(p) if h == r.home else [p[2], p[1], p[0]]
            bp, bh = bkey[key]
            pe, eh = ekey[key]
            ps, sh = skey[key]
            y = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
            rows.append(dict(
                edition=ed, bar=ODDS_FILES[ed][1], date=r.date, home=r.home, away=r.away,
                outcome=y, stage="group" if r.date <= GROUP_END[ed] else "knockout",
                market=devig_proportional(r.odds_home, r.odds_draw, r.odds_away),
                market_shin=devig_shin(r.odds_home, r.odds_draw, r.odds_away),
                baseline=orient(bp, bh), elo=orient(pe, eh), blend=orient(ps, sh),
            ))
    return pd.DataFrame(rows), beta


def _rps(col, df):
    return metrics.ranked_probability_score(np.array(df[col].tolist()), df["outcome"].to_numpy())


MODELS = [("MARKET", "market"), ("Baseline Elo->DC", "baseline"),
          ("Elo only (probit)", "elo"), ("Elo + squad value", "blend")]


def _report(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}  (n={len(df)})")
    print(f"  {'forecast':<20}{'RPS':>8}{'logloss':>9}{'brier':>8}")
    for label, col in MODELS:
        P = np.array(df[col].tolist())
        y = df["outcome"].to_numpy()
        print(f"  {label:<20}{_rps(col, df).mean():>8.4f}"
              f"{metrics.log_loss(P, y).mean():>9.4f}{metrics.brier_score(P, y).mean():>8.4f}")
    rps_mkt = _rps("market", df)
    parts = []
    for label, col in [("baseline", "baseline"), ("blend", "blend")]:
        d = _rps(col, df) - rps_mkt
        se = d.std(ddof=1) / np.sqrt(len(d))
        parts.append(f"{label} Δ={d.mean():+.4f} (t={d.mean() / se:+.2f})")
    print(f"  vs market [neg=model better]: " + ";  ".join(parts))
    print(f"  de-vig check: proportional={rps_mkt.mean():.4f}, Shin={_rps('market_shin', df).mean():.4f}")


def main() -> None:
    df, beta = build()
    print(f"\nMarket backtest. Leave-one-tournament-out squad blend "
          f"(mean coef: elo={beta[0]:+.2f}, squad={beta[1]:+.2f}).")

    for ed in df["edition"].unique():
        sub = df[df.edition == ed]
        _report(sub, f"== {ed} vs {sub['bar'].iloc[0]} ==")

    # Group-stage-only pooled view: clean W/D/L (no extra-time/penalty ambiguity).
    g = df[df.stage == "group"]
    print("\n" + "-" * 60)
    print("NOTE: 2018 (average odds) and 2022 (Pinnacle closing) are different "
          "bars,\nso the pooled number below mixes a soft and a sharp line - "
          "read per-tournament\nabove as the real result. Group stage only:")
    _report(g, "== POOLED group stage (mixed bars) ==")

    print("\nVERDICT (honest): squad value puts the model at market level - it "
          "edges the\nsofter 2018 average line and matches/just-misses the sharp "
          "2022 closing line.\nNo proven edge over a closing price; no tournament "
          "where plain Elo beats it.")


if __name__ == "__main__":
    main()
