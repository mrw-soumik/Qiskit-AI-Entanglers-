"""Builds the training set for the QAOA-angle predictor (Section 11 of the notebook). Run once (~10 min, 10 cores):
    python make_ai_training_set.py
Instances: the flat 4 x 4 grid (sub-parcels of 9, 12 and 16 sites) and the real-terrain parcel (first 12 and 16
sites), for each wind year 2007-2025 and three cost levels. For each instance: QAOA angles at depth 1 and 2 trained
with CVaR on the sparsified Hamiltonian (as used on hardware) and the quality they reach. Writes
results/ai_qubos.npz (every QUBO) and results/ai_training_set.csv (features, angles, quality).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool

import numpy as np
import pandas as pd

import wakewise as ww

YEARS = range(2007, 2026)
COSTS = (0.84, 0.88, 0.92)
KEEP = 0.25  # sparsification used for the hardware circuits
FLAT_SUBSETS = {9: [0, 1, 2, 4, 5, 6, 8, 9, 10], 12: list(range(12)), 16: list(range(16))}
TERRAIN_SUBSETS = {12: list(range(12)), 16: list(range(16))}


def year_qubos(y):
    """Base QUBOs (cost level 0.88) of the flat 4 x 4 grid and of the 24-site terrain parcel for one wind year."""
    wd, u = ww.load_wind((y,))
    lin_f, W_f, info_f = ww.build_qubo(ww.grid(4, 4, 5 * ww.D), wd, u)
    sites = ww.terrain_sites()[0]
    model = ww.terrain_model(sites)
    xy = sites[["x", "y"]].to_numpy()[:16]
    E_ref = np.mean([ww.farm_energy(model, sites[["x", "y"]].to_numpy()[[i]], wd, u)[0] for i in range(len(sites))])
    lin_t, W_t, info_t = ww.build_qubo(xy, wd, u, model=model, E_ref=E_ref)
    return y, (lin_f, W_f, info_f["E"]), (lin_t, W_t, info_t["E"])


def features(lin, W):
    """Statistics of the normalized sparse Ising Hamiltonian (what the circuit sees)."""
    H, _ = ww.qubo_to_ising(lin, ww.sparsify(W, KEEP))
    c = np.real(H.coeffs) / np.abs(H.coeffs).max()
    one = np.array([len(p.to_label().replace("I", "")) == 1 for p in H.paulis])
    h, J = c[one], c[~one]
    n = len(lin)
    return dict(n=n, h_mean=h.mean(), h_std=h.std(), h_absmax=np.abs(h).max(), J_mean=J.mean() if len(J) else 0.0,
                J_std=J.std() if len(J) else 0.0, degree=2 * len(J) / n, h_pos=(h > 0).mean())


def canonical(th, p):
    """Map QAOA angles to one representative: (gamma, beta) ~ (-gamma, -beta), and beta has period pi."""
    th = np.array(th, float)
    if th[p] < 0:
        th = -th
    th[:p] = np.mod(th[:p], np.pi)
    return th


def train(job):
    """Trained depth-1 and depth-2 angles and their quality for one instance."""
    key, lin, W = job
    values = ww.exact(lin, W)[2]
    f_opt, f_rand = values.min(), values.mean()
    f_gr = ww.qubo_values(ww.greedy(lin, W), lin, W)[0]
    Ws = ww.sparsify(W, KEEP)
    diag = ww.ising_diagonal(lin, Ws)
    row = dict(key=key, **features(lin, W), f_opt=f_opt, f_rand=f_rand, f_greedy=f_gr)
    for p in (1, 2):
        r = ww.train_cvar(diag, values, p)
        th = canonical(r.x, p)
        pr = ww.qaoa_probabilities(diag, th)
        row.update({f"p{p}_th{k}": v for k, v in enumerate(th)})
        row[f"p{p}_gap"] = (f_rand - pr @ values) / (f_rand - f_opt)
        row[f"p{p}_popt"] = pr[np.isclose(values, f_opt)].sum()
        row[f"p{p}_cvar"] = r.fun
    return row


if __name__ == "__main__":
    with Pool(10) as pool:
        base = pool.map(year_qubos, YEARS)
    jobs, qubos = [], {}
    for y, (lin_f, W_f, E_f), (lin_t, W_t, E_t) in base:
        for cf in COSTS:
            for src, lin, W, E, subsets in (("flat", lin_f, W_f, E_f, FLAT_SUBSETS), ("terrain", lin_t, W_t, E_t, TERRAIN_SUBSETS)):
                lin_c = lin + (cf - 0.88) * E  # same parcel, different cost level: cost = cf * E_ref + cable
                for n, idx in subsets.items():
                    key = f"{src}|{n}|{y}|{cf}"
                    qubos[key] = (lin_c[idx], W[np.ix_(idx, idx)])
                    jobs.append((key, *qubos[key]))
    print(len(jobs), "instances; training...", flush=True)
    with Pool(10) as pool:
        rows = pool.map(train, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    df[["source", "n_sites", "year", "cost_frac"]] = df.key.str.split("|", expand=True)
    df.to_csv(ww.RESULTS / "ai_training_set.csv", index=False)
    np.savez_compressed(ww.RESULTS / "ai_qubos.npz", **{k + "|lin": v[0] for k, v in qubos.items()},
                        **{k + "|W": v[1] for k, v in qubos.items()})
    print("wrote results/ai_training_set.csv and results/ai_qubos.npz:", len(df), "instances")
