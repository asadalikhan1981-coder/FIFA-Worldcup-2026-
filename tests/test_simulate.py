import numpy as np

from wc2026 import simulate


def _toy_pairwise(teams):
    n = len(teams)
    # mild strength gradient so stronger (lower index) teams win more
    pW = np.zeros((n, n)); pD = np.zeros((n, n)); pL = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            edge = 0.04 * (b - a)
            pw = np.clip(0.40 + edge, 0.1, 0.8)
            pd_ = 0.25
            pW[a, b] = pw; pD[a, b] = pd_; pL[a, b] = 1 - pw - pd_
    return simulate.Pairwise(teams, pW, pD, pL)


def _toy_tournament():
    teams = [f"T{k}" for k in range(48)]
    groups, base_pts, base_gd, base_gf, remaining = {}, {}, {}, {}, []
    for n in range(12):
        L = chr(65 + n)
        members = teams[n * 4:(n + 1) * 4]
        groups[L] = members
        base_pts[L] = [0, 0, 0, 0]; base_gd[L] = [0, 0, 0, 0]; base_gf[L] = [0, 0, 0, 0]
        # all 6 group games unplayed
        for i in range(4):
            for j in range(i + 1, 4):
                remaining.append((L, members[i], members[j]))
    return teams, groups, base_pts, base_gd, base_gf, remaining


def test_probabilities_valid_and_monotone():
    teams, groups, bp, bgd, bgf, rem = _toy_tournament()
    pw = _toy_pairwise(teams)
    df = simulate.simulate(groups, bp, bgd, bgf, rem, pw, n_sims=300, seed=1)

    # exactly one champion's worth of probability mass
    assert abs(df["win"].sum() - 1.0) < 1e-9
    # round reach probabilities are ordered: R16 >= QF >= SF >= final >= win
    for _, r in df.iterrows():
        assert r["reach_R16"] >= r["reach_QF"] >= r["reach_SF"] >= r["reach_final"] >= r["win"]
    # 32 teams reach the R16 on every run -> total reach_R16 mass == 32
    assert abs(df["reach_R16"].sum() - 32.0) < 1e-9
    # stronger teams (low index) should win more often than the weakest
    assert df.set_index("team").loc["T0", "win"] > df.set_index("team").loc["T47", "win"]


def test_no_same_group_in_r32_constraint_helpers():
    grp = {0: "A", 1: "A", 2: "B", 3: "B"}
    rng = np.random.default_rng(0)
    pairs = simulate._pair_no_rematch([0, 1], [2, 3], grp, rng)
    assert pairs is not None
    assert all(grp[a] != grp[b] for a, b in pairs)
