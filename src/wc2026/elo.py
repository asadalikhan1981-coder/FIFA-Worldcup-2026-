"""World Football Elo ratings (eloratings.net algorithm) + in-tournament update.

Two regimes, both implemented here:

1. ``build_history`` walks the full match history and maintains a rating per
   team using the standard eloratings.net weights. This produces the
   pre-tournament *priors*.

2. ``shrunk_update`` is the low-K (K=8) update the handover specifies for
   nudging ratings *during* a tournament, so that a single group game barely
   moves a team (one match is mostly noise).

The point of keeping them separate is the backtest: priors are frozen using
only matches strictly before a tournament starts (no peeking), then the shrunk
update conditions later matches on earlier results within the same tournament.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_RATING = 1500.0
HOME_ADVANTAGE = 100.0  # eloratings.net standard, in Elo points


# eloratings.net match-importance weights, mapped from the tournament string in
# the martj42 dataset. Anything unmatched falls back to FRIENDLY.
def importance_weight(tournament: str) -> float:
    t = (tournament or "").lower()
    if t == "friendly":
        return 20.0
    if "qualification" in t or "qualifier" in t:
        return 40.0  # World Cup / continental qualifiers
    if "fifa world cup" in t:
        return 60.0  # World Cup finals
    if "confederations cup" in t:
        return 50.0
    if any(k in t for k in ("uefa euro", "copa américa", "copa america",
                            "african cup of nations", "afc asian cup",
                            "gold cup", "concacaf", "nations league")):
        return 50.0  # continental finals
    return 30.0  # other tournaments


def goal_diff_multiplier(goal_diff: int) -> float:
    """eloratings.net margin-of-victory multiplier."""
    g = abs(int(goal_diff))
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0


def expected_score(r_home: float, r_away: float, home_adv: float) -> float:
    """Logistic expectation for the home team (include home_adv=0 if neutral)."""
    return 1.0 / (1.0 + 10.0 ** (-((r_home + home_adv) - r_away) / 400.0))


@dataclass
class EloModel:
    """Mutable rating store with a chronological update."""

    ratings: dict[str, float] = field(default_factory=dict)
    home_advantage: float = HOME_ADVANTAGE
    default_rating: float = DEFAULT_RATING

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.default_rating)

    def update_match(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        k: float,
        neutral: bool = False,
    ) -> None:
        h_adv = 0.0 if neutral else self.home_advantage
        rh, ra = self.get(home), self.get(away)
        we = expected_score(rh, ra, h_adv)
        if home_score > away_score:
            w = 1.0
        elif home_score == away_score:
            w = 0.5
        else:
            w = 0.0
        g = goal_diff_multiplier(home_score - away_score)
        delta = k * g * (w - we)
        self.ratings[home] = rh + delta
        self.ratings[away] = ra - delta

    def copy(self) -> "EloModel":
        return EloModel(dict(self.ratings), self.home_advantage, self.default_rating)


def build_history(
    matches: pd.DataFrame,
    until: pd.Timestamp | None = None,
    home_advantage: float = HOME_ADVANTAGE,
) -> EloModel:
    """Build ratings from match history, optionally only using matches < ``until``.

    ``matches`` must have columns: date, home_team, away_team, home_score,
    away_score, tournament, neutral (bool). Rows are processed in date order.
    """
    df = matches
    if until is not None:
        df = df[df["date"] < until]
    df = df.sort_values("date", kind="stable")
    model = EloModel(home_advantage=home_advantage)
    # Iterate as numpy for speed over ~50k rows.
    homes = df["home_team"].to_numpy()
    aways = df["away_team"].to_numpy()
    hs = df["home_score"].to_numpy()
    as_ = df["away_score"].to_numpy()
    tours = df["tournament"].to_numpy()
    neut = df["neutral"].to_numpy()
    for i in range(len(df)):
        model.update_match(
            homes[i], aways[i], hs[i], as_[i],
            k=importance_weight(tours[i]), neutral=bool(neut[i]),
        )
    return model


def shrunk_update(
    model: EloModel,
    home: str,
    away: str,
    home_score: int,
    away_score: int,
    neutral: bool = True,
    k: float = 8.0,
) -> None:
    """In-tournament nudge with low K and the handover's margin multipliers.

    Margin multiplier m in {1, 1.5, 1.75, 2} for |gd| in {0/1, 2, 3, >=4}.
    With K=8 a single result moves a team only a few Elo points, so early
    games barely shift the priors — which is correct, one game is mostly noise.
    """
    h_adv = 0.0 if neutral else model.home_advantage
    rh, ra = model.get(home), model.get(away)
    we = expected_score(rh, ra, h_adv)
    if home_score > away_score:
        w = 1.0
    elif home_score == away_score:
        w = 0.5
    else:
        w = 0.0
    gd = abs(home_score - away_score)
    m = {0: 1.0, 1: 1.0, 2: 1.5, 3: 1.75}.get(gd, 2.0)
    delta = k * m * (w - we)
    model.ratings[home] = rh + delta
    model.ratings[away] = ra - delta
