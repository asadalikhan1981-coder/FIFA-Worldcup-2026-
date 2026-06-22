"""Monte-Carlo tournament simulator for the real 2026 World Cup.

Conditions on the group results already played (locked) and simulates the rest:
remaining group games -> final group tables -> 12 winners + 12 runners-up +
8 best third-placed -> Round of 32 -> ... -> winner.

Match probabilities come from the volatility-aware model (level + swing). In the
knockout rounds there are no draws, so a draw is resolved in proportion to the
two teams' relative win chances (the extra-time/penalties stand-in):
    P(i advances) = pW(i) / (pW(i) + pL(i)).

Bracket honesty: the exact official R32 slot table finalises only after the
group stage and follows a fixed FIFA allocation we could not verify in this
environment. So each simulation draws a *valid* bracket at random — group
winners against runners-up/thirds in the real 8/4/4 pattern, with no same-group
rematch in the R32 — and the reported probabilities average over brackets. This
avoids the favourite-inflating bias of a strength-seeded bracket.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


ROUNDS = ["reach_R16", "reach_QF", "reach_SF", "reach_final", "win"]


class Pairwise:
    """Pre-computed win/draw/loss and no-draw win probabilities between teams."""

    def __init__(self, teams, pW, pD, pL):
        self.teams = teams
        self.idx = {t: k for k, t in enumerate(teams)}
        self.pW, self.pD, self.pL = pW, pD, pL
        # Knockout: a draw goes to extra-time/penalties, treated as ~a coin flip
        # (pW + 0.5*pD) rather than resolving in proportion to strength, which
        # would over-reward favourites.
        self.pWin_nd = pW + 0.5 * pD

    @classmethod
    def from_model(cls, fit_res, teams):
        from .models import volatility_strength as vs
        n = len(teams)
        pW = np.zeros((n, n)); pD = np.zeros((n, n)); pL = np.zeros((n, n))
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                p = vs.predict(fit_res, teams[a], teams[b], neutral=True)
                pW[a, b], pD[a, b], pL[a, b] = p
        return cls(teams, pW, pD, pL)


def _play_group_game(i, j, pw, rng):
    """Sample 0=i win, 1=draw, 2=j win for a group game (index space)."""
    r = rng.random()
    if r < pw.pW[i, j]:
        return 0
    if r < pw.pW[i, j] + pw.pD[i, j]:
        return 1
    return 2


def _ko(i, j, pw, rng):
    """Knockout: return the index of the advancing team."""
    return i if rng.random() < pw.pWin_nd[i, j] else j


def _rank_group(members, pts, gd, gf, rng):
    """Rank a group's teams: points, goal diff, goals for, then random."""
    jitter = rng.random(len(members))
    order = sorted(range(len(members)),
                   key=lambda k: (-pts[k], -gd[k], -gf[k], jitter[k]))
    return [members[k] for k in order]


def simulate(
    groups: dict[str, list[str]],
    base_pts: dict[str, list[int]],   # group letter -> [pts] aligned to groups[letter]
    base_gd: dict[str, list[int]],
    base_gf: dict[str, list[int]],
    remaining: list[tuple[str, str, str]],  # (group_letter, home, away) not yet played
    pw: Pairwise,
    n_sims: int = 20000,
    seed: int = 12345,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pw.idx
    letters = list(groups.keys())
    # remaining games grouped by letter, in index space
    rem_by_group = {L: [] for L in letters}
    for L, h, a in remaining:
        rem_by_group[L].append((idx[h], idx[a]))

    team_of_group = {L: [idx[t] for t in groups[L]] for L in letters}
    counts = {t: dict.fromkeys(ROUNDS, 0) for t in pw.teams}

    for _ in range(n_sims):
        winners, runners, thirds = [], [], []  # thirds: (team_idx, pts, gd, gf)
        for L in letters:
            members = list(team_of_group[L])
            local = {ti: k for k, ti in enumerate(members)}
            pts = list(base_pts[L]); gd = list(base_gd[L]); gf = list(base_gf[L])
            for i, j in rem_by_group[L]:
                o = _play_group_game(i, j, pw, rng)
                ki, kj = local[i], local[j]
                if o == 0:
                    pts[ki] += 3; gd[ki] += 1; gd[kj] -= 1; gf[ki] += 1
                elif o == 2:
                    pts[kj] += 3; gd[kj] += 1; gd[ki] -= 1; gf[kj] += 1
                else:
                    pts[ki] += 1; pts[kj] += 1
                    # neutral goal bump for draws (1-1 typical) keeps gf sane
                    gf[ki] += 1; gf[kj] += 1
            ranked = _rank_group(members, pts, gd, gf, rng)
            kmap = {ti: k for k, ti in enumerate(members)}
            winners.append(ranked[0]); runners.append(ranked[1])
            t3 = ranked[2]
            thirds.append((t3, pts[kmap[t3]], gd[kmap[t3]], gf[kmap[t3]]))

        # best 8 of 12 third-placed teams
        thirds.sort(key=lambda x: (-x[1], -x[2], -x[3], rng.random()))
        best_thirds = [t[0] for t in thirds[:8]]

        # group membership for the no-rematch constraint
        grp = {}
        for L in letters:
            for ti in team_of_group[L]:
                grp[ti] = L

        # ---- random valid R32: 8 winners vs thirds, 4 winners vs runners, 4 ru vs ru
        w = winners[:]; ru = runners[:]; th = best_thirds[:]
        rng.shuffle(w); rng.shuffle(ru); rng.shuffle(th)
        w_vs_third, w_rest = w[:8], w[8:]
        r32 = []
        ok = _pair_no_rematch(w_vs_third, th, grp, rng)
        if ok is None:
            # extremely rare; fall back to ignoring the constraint
            ok = list(zip(w_vs_third, th))
        r32 += ok
        # 4 winners vs 4 runners-up
        ru_for_w, ru_rest = ru[:4], ru[4:]
        ok2 = _pair_no_rematch(w_rest, ru_for_w, grp, rng) or list(zip(w_rest, ru_for_w))
        r32 += ok2
        # remaining 8 runners-up among themselves
        ok3 = _self_pair_no_rematch(ru_rest, grp, rng)
        r32 += ok3

        # ---- play the fixed tree over the 16 R32 matches
        survivors = r32  # list of (a,b)
        round_names = ["reach_R16", "reach_QF", "reach_SF", "reach_final", "win"]
        # winners of R32 -> reach_R16
        advancing = [_ko(a, b, pw, rng) for a, b in survivors]
        for t in advancing:
            counts[pw.teams[t]]["reach_R16"] += 1
        for rname in round_names[1:]:
            nxt = []
            for k in range(0, len(advancing), 2):
                a, b = advancing[k], advancing[k + 1]
                nxt.append(_ko(a, b, pw, rng))
            advancing = nxt
            for t in advancing:
                counts[pw.teams[t]][rname] += 1

    rows = []
    for t in pw.teams:
        row = {"team": t}
        for r in ROUNDS:
            row[r] = counts[t][r] / n_sims
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("win", ascending=False).reset_index(drop=True)
    return df


def _pair_no_rematch(left, right, grp, rng, tries=40):
    """Pair two equal-length lists avoiding same-group pairs."""
    for _ in range(tries):
        r = right[:]; rng.shuffle(r)
        if all(grp[a] != grp[b] for a, b in zip(left, r)):
            return list(zip(left, r))
    return None


def _self_pair_no_rematch(teams, grp, rng, tries=60):
    """Pair a list with itself into len/2 matches avoiding same-group pairs."""
    for _ in range(tries):
        t = teams[:]; rng.shuffle(t)
        pairs = [(t[k], t[k + 1]) for k in range(0, len(t), 2)]
        if all(grp[a] != grp[b] for a, b in pairs):
            return pairs
    return [(teams[k], teams[k + 1]) for k in range(0, len(teams), 2)]
