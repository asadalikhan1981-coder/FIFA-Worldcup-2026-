"""Run the baseline (control) backtest and report the numbers everything else
must beat. This is the *control*, deliberately the textbook pipeline.

    python scripts/run_baseline_backtest.py

Outputs a per-tournament scoring table to stdout, writes per-match predictions
to results/baseline_predictions.csv and a calibration plot to
results/baseline_calibration.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wc2026 import backtest, data, metrics  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def reference_benchmarks(per_match: pd.DataFrame) -> pd.DataFrame:
    """Trivial baselines for context (NOT the bookmaker bar).

    - uniform: [1/3, 1/3, 1/3] for every match.
    - climatology: the marginal outcome frequency of the backtest set itself.
      This one *peeks* at the test marginal, so it is an intentionally
      generous reference — if the Elo model cannot beat climatology it is
      worthless.
    """
    y = per_match["outcome"].to_numpy()
    n = len(y)
    rows = []

    uni = np.tile([1 / 3, 1 / 3, 1 / 3], (n, 1))
    s = metrics.summary(uni, y); s["edition"] = "ref:uniform"; rows.append(s)

    freq = np.bincount(y, minlength=3) / n
    clim = np.tile(freq, (n, 1))
    s = metrics.summary(clim, y); s["edition"] = "ref:climatology"; rows.append(s)

    return pd.DataFrame(rows)[["edition", "n", "rps", "brier", "log_loss"]]


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = data.load_results()
    per_match, per_tourn = backtest.run_all(results)

    refs = reference_benchmarks(per_match)

    print("\n=== Baseline (Elo -> Dixon-Coles) out-of-sample backtest ===")
    table = pd.concat([per_tourn, refs], ignore_index=True)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(table.to_string(index=False))

    per_match.to_csv(RESULTS_DIR / "baseline_predictions.csv", index=False)
    print(f"\nWrote per-match predictions -> {RESULTS_DIR/'baseline_predictions.csv'}")

    # Calibration plot for the "home/first-team win" event.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cal = metrics.calibration_table(
            per_match["p_home"].to_numpy(),
            (per_match["outcome"].to_numpy() == 0).astype(float),
            n_bins=10,
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.plot(cal["mean_pred"], cal["obs_freq"], "o-", label="baseline")
        for x, y_, c in zip(cal["mean_pred"], cal["obs_freq"], cal["count"]):
            if not np.isnan(x):
                ax.annotate(str(c), (x, y_), fontsize=7,
                            textcoords="offset points", xytext=(4, -8))
        ax.set_xlabel("predicted P(first-team win)")
        ax.set_ylabel("observed frequency")
        ax.set_title("Baseline calibration — first-team win\n(numbers = matches per bin)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "baseline_calibration.png", dpi=120)
        print(f"Wrote calibration plot   -> {RESULTS_DIR/'baseline_calibration.png'}")
    except Exception as exc:  # pragma: no cover
        print(f"(calibration plot skipped: {exc})")


if __name__ == "__main__":
    main()
