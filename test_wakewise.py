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


def test_pywake_pair_losses_are_physical():
    # Three sites in an east-west row, 650 m apart, with a steady westerly wind at 9 m/s.
    xy = np.array([[0.0, 0.0], [650.0, 0.0], [1300.0, 0.0]])
    wd, u = np.full(24, 270.0), np.full(24, 9.0)
    lin, W, info = ww.build_qubo(xy, wd, u)
    assert np.allclose(W, W.T) and np.all(np.diag(W) == 0) and np.all(W >= 0)
    assert W[0, 1] > W[0, 2] > 0                     # the wake recovers with distance
    one = ww.farm_energy(ww.wake_model(), xy[:1], wd, u)[0]
    assert np.isclose(one, info["E"])                # a lone turbine loses nothing
    crosswind = np.array([[0.0, 0.0], [0.0, 650.0]])  # north-south pair in a westerly: no shading
    _, Wc, _ = ww.build_qubo(crosswind, wd, u)
    assert Wc[0, 1] == 0


def test_cvar_limits():
    p = np.array([0.1, 0.2, 0.3, 0.4]); v = np.array([3.0, -1.0, 2.0, 0.0])
    assert np.isclose(ww.cvar(p, v, 1.0), p @ v)     # alpha = 1 is the plain average
    assert np.isclose(ww.cvar(p, v, 0.1), -1.0)      # a small alpha keeps only the best outcome


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
