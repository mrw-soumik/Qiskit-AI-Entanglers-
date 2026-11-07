"""Self-checks for the classical model and the QUBO -> Ising mapping. Run: python test_wakewise.py"""
import numpy as np

import wakewise as ww


def test_ising_matches_qubo_for_every_bitstring():
    rng = np.random.default_rng(0)
    n = 5
    lin = rng.normal(size=n)
    W = np.triu(rng.normal(size=(n, n)), 1)
    W += W.T
    H, offset = ww.qubo_to_ising(lin, W)
    # Diagonal entry s of a Z-only operator is the energy of basis state |s>, with qubit i = bit i of s.
    energies = H.to_matrix(sparse=True).diagonal().real + offset
    assert np.allclose(energies, ww.qubo_values(ww.bits(np.arange(2 ** n), n), lin, W))


def test_three_site_case_by_hand():
    # Each site alone is worth 10; sites 0 and 1 together lose 15, site 2 is independent.
    lin = np.array([-10.0, -10.0, -10.0])
    W = np.zeros((3, 3))
    W[0, 1] = W[1, 0] = 15.0
    x, f, _ = ww.exact(lin, W)
    assert f == -20 and x[2] == 1 and x[0] + x[1] == 1
    assert ww.qubo_values([1, 1, 1], lin, W)[0] == -15
    assert ww.greedy(lin, W).sum() == 2


def test_bit_order_round_trip():
    assert ww.bits([0b0110], 4).tolist() == [[0, 1, 1, 0]]  # integer bit i -> site i


def test_wake_only_hits_downstream_turbines():
    xy = np.array([[0.0, 0.0], [650.0, 0.0], [0.0, 650.0]])  # site 1 east of 0, site 2 north of 0
    west = np.array([270.0])                                   # wind FROM the west blows toward the east
    assert ww.wake_deficit(xy, 0, 1, west)[0] > 0              # 1 is downstream of 0
    assert ww.wake_deficit(xy, 1, 0, west)[0] == 0             # 0 is upstream of 1
    assert ww.wake_deficit(xy, 0, 2, west)[0] == 0             # 2 is crosswind of 0
    near, far = ww.wake_deficit(np.array([[0, 0], [650, 0], [1300, 0.0]]), 0, 1, west), \
        ww.wake_deficit(np.array([[0, 0], [650, 0], [1300, 0.0]]), 0, 2, west)
    assert near[0] > far[0] > 0                                # the wake recovers with distance


def test_summarize_scores_samples():
    lin = np.array([-1.0, -1.0])
    W = np.array([[0.0, 3.0], [3.0, 0.0]])
    _, f_opt, v = ww.exact(lin, W)
    s = ww.summarize({1: 50, 2: 50}, lin, W, f_opt, v.mean())  # both samples are optimal layouts
    assert s["p_opt"] == 1.0 and s["gap_closed"] == 1.0 and s["best_gap_pct"] == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
