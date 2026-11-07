"""WakeWise NL: wake-aware wind-turbine siting for Newfoundland's west coast, solved with QAOA.

Classical side: real ECCC hourly wind -> Jensen wake model -> QUBO -> exact / greedy / random baselines.
Quantum side: QUBO -> Ising Hamiltonian -> QAOA (simulator, noisy simulator, IBM hardware).
"""
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA, RESULTS = ROOT / "data", ROOT / "results"

# Generic 4 MW onshore turbine (illustrative values, not a specific product).
TURBINE = dict(P_rated=4.0, D=130.0, hub=110.0, cut_in=3.0, rated=12.0, cut_out=25.0, Ct=0.8)
WAKE_K = 0.075  # Jensen wake-decay constant, common onshore value
HOURS = 8760

ECCC_URL = ("https://api.weather.gc.ca/collections/climate-hourly/items?CLIMATE_IDENTIFIER={cid}"
            "&LOCAL_YEAR={year}&properties=LOCAL_DATE,WIND_DIRECTION,WIND_SPEED&limit=10000&f=json")


# ---------------------------------------------------------------- wind data
def fetch_wind(cid="8404343", year=2025, path=None):
    """Download one year of hourly ECCC wind (Wreckhouse by default) and freeze it to CSV."""
    url = ECCC_URL.format(cid=cid, year=year)
    with urllib.request.urlopen(url, timeout=180) as r:
        feats = json.load(r)["features"]
    df = pd.DataFrame([f["properties"] for f in feats]).sort_values("LOCAL_DATE")
    df.to_csv(path or DATA / f"wreckhouse_{year}_hourly.csv", index=False)
    return df, url


def load_wind(path=DATA / "wreckhouse_2025_hourly.csv", hub=TURBINE["hub"], alpha=0.14):
    """Return per-hour wind direction (degrees, blowing FROM) and hub-height speed (m/s).

    ECCC gives direction in tens of degrees (0 = calm) and 10 m speed in km/h. Hours with a
    missing value are dropped; calm hours stay in with zero speed so the energy average is fair.
    The 10 m -> hub extrapolation uses a power-law shear profile (alpha is an assumption).
    """
    df = pd.read_csv(path)
    ok = df.WIND_DIRECTION.notna() & df.WIND_SPEED.notna()
    wd = df.WIND_DIRECTION[ok].to_numpy(float) * 10
    u = df.WIND_SPEED[ok].to_numpy(float) / 3.6 * (hub / 10) ** alpha
    return wd, u


# ---------------------------------------------------------------- physics
def power_mw(u, t=TURBINE):
    """Generic power curve: cubic rise from cut-in to rated speed, flat to cut-out."""
    ramp = t["P_rated"] * ((np.asarray(u) - t["cut_in"]) / (t["rated"] - t["cut_in"])) ** 3
    return np.where((u >= t["cut_in"]) & (u < t["cut_out"]), np.minimum(ramp, t["P_rated"]), 0.0)


def grid(rows, cols, spacing):
    """Site coordinates in metres (x = east, y = north); site index = row * cols + col."""
    return np.array([(c * spacing, r * spacing) for r in range(rows) for c in range(cols)], float)


def wake_deficit(xy, i, j, wd, t=TURBINE, k=WAKE_K):
    """Fractional speed deficit at turbine j caused by turbine i, per hour (Jensen top-hat wake)."""
    th = np.radians(wd)
    fx, fy = -np.sin(th), -np.cos(th)          # direction the air moves (wd is where it comes FROM)
    dx, dy = xy[j] - xy[i]
    down = dx * fx + dy * fy                   # distance j sits downstream of i
    cross = np.abs(dx * fy - dy * fx)          # sideways offset from i's wake centreline
    r = t["D"] / 2
    inside = (down > 0) & (cross < r + k * down)
    return np.where(inside, (1 - np.sqrt(1 - t["Ct"])) / (1 + k * down / r) ** 2, 0.0)


def build_qubo(xy, wd, u, cost_frac=0.88, cable_frac=0.02, t=TURBINE, k=WAKE_K):
    """QUBO in MWh/yr: f(x) = lin . x + sum_{i<j} W_ij x_i x_j  (lower is better).

    lin_i = -(E - c_i): free-stream energy of a turbine minus its annualized cost, where cost is
            cost_frac * E plus cable_frac * E per grid spacing of cable to the substation at site 0
            (illustrative economics expressed as energy).
    W_ij  = energy lost when both i and j are built (i shading j plus j shading i).
    """
    n = len(xy)
    p_free = power_mw(u, t)
    E = p_free.mean() * HOURS
    loss = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                p_waked = power_mw(u * (1 - wake_deficit(xy, i, j, wd, t, k)), t)
                loss[i, j] = (p_free - p_waked).mean() * HOURS
    W = loss + loss.T
    spacing = np.min(np.hypot(*(xy[1:] - xy[0]).T))
    cost = cost_frac * E + cable_frac * E * np.hypot(*(xy - xy[0]).T) / spacing
    return -(E - cost), W, dict(E=E, cost=cost)


def layout_aep(x, xy, wd, u, t=TURBINE, k=WAKE_K):
    """Farm energy (MWh/yr) with sum-of-squares wake superposition: a non-pairwise check of the QUBO."""
    on = np.flatnonzero(x)
    total = 0.0
    for j in on:
        d2 = sum(wake_deficit(xy, i, j, wd, t, k) ** 2 for i in on if i != j)
        total += power_mw(u * (1 - np.sqrt(d2)), t).mean() * HOURS
    return total


# ---------------------------------------------------------------- QUBO tools and baselines
def bits(states, n):
    """Integer outcomes -> 0/1 rows. Bit i of the integer is site i (Qiskit's little-endian order)."""
    return ((np.asarray(states)[:, None] >> np.arange(n)) & 1).astype(float)


def qubo_values(X, lin, W):
    X = np.atleast_2d(X)
    return X @ lin + 0.5 * np.einsum("mi,ij,mj->m", X, W, X)


def exact(lin, W):
    """Brute force over all 2^n layouts. Returns (best x, best value, all values indexed by integer)."""
    n = len(lin)
    v = qubo_values(bits(np.arange(2 ** n), n), lin, W)
    return bits([v.argmin()], n)[0], v.min(), v


def greedy(lin, W):
    """Keep adding the site with the best marginal gain until no site improves the objective."""
    x = np.zeros(len(lin))
    while True:
        gain = lin + W @ x
        gain[x == 1] = np.inf
        i = int(gain.argmin())
        if gain[i] >= 0:
            return x
        x[i] = 1


def sparsify(W, keep):
    """Drop couplings weaker than keep * strongest (fewer ZZ terms -> shallower hardware circuits)."""
    return np.where(np.abs(W) >= keep * np.abs(W).max(), W, 0.0)


def qubo_to_ising(lin, W):
    """x_i = (1 - z_i)/2  ->  H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j,  f(x) = <H> + offset."""
    from qiskit.quantum_info import SparsePauliOp
    n = len(lin)
    iu = np.triu_indices(n, 1)
    terms = [("Z", [i], -lin[i] / 2 - W[i].sum() / 4) for i in range(n)]
    terms += [("ZZ", [i, j], W[i, j] / 4) for i, j in zip(*iu) if W[i, j] != 0]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n), lin.sum() / 2 + W[iu].sum() / 4


def summarize(counts, lin, W, f_opt, f_mean):
    """Score sampled layouts with the full QUBO.

    gap_closed = (f_random - <f>_samples) / (f_random - f_opt): 0 = no better than uniform random
    guessing, 1 = always the optimum.
    """
    states = np.array([int(s) for s in counts])  # keys may be int (live) or str (from JSON)
    p = np.array(list(counts.values()), float)
    p /= p.sum()
    v = qubo_values(bits(states, len(lin)), lin, W)
    best = states[v.argmin()]
    return dict(p_opt=float(p[np.isclose(v, f_opt)].sum()), mean=float(p @ v),
                gap_closed=float((f_mean - p @ v) / (f_mean - f_opt)),
                best_state=int(best), best_value=float(v.min()), best_gap_pct=float((v.min() - f_opt) / abs(f_opt) * 100))


# ---------------------------------------------------------------- QAOA
def qaoa_circuit(H, reps):
    from qiskit.circuit.library import qaoa_ansatz  # function form; the QAOAAnsatz class is deprecated in Qiskit 2.x
    return qaoa_ansatz(H, reps=reps)


def train_qaoa(ansatz, H, starts=5, seed=7, maxiter=300):
    """Minimize <H> on the exact statevector estimator with COBYLA from several seeded starts."""
    from qiskit.primitives import StatevectorEstimator
    from scipy.optimize import minimize
    est = StatevectorEstimator()
    cost = lambda th: float(est.run([(ansatz, H, th)]).result()[0].data.evs)
    rng = np.random.default_rng(seed)
    runs = [minimize(cost, rng.uniform(0, np.pi, ansatz.num_parameters), method="COBYLA",
                     options={"maxiter": maxiter}) for _ in range(starts)]
    return min(runs, key=lambda r: r.fun), [float(r.fun) for r in runs]


def ideal_counts(circuit, params, shots=10_000, seed=11):
    from qiskit.primitives import StatevectorSampler
    qc = circuit.assign_parameters(params)
    qc.measure_all()
    return StatevectorSampler(seed=seed).run([qc], shots=shots).result()[0].data.meas.get_int_counts()


def load_env(path=ROOT / ".env"):
    """Read KEY=VALUE lines from .env into os.environ (already-set variables win)."""
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            key, sep, val = line.partition("=")
            if sep and not key.strip().startswith("#"):
                os.environ.setdefault(key.strip(), val.strip())


def hardware_backend():
    """IBM backend from .env credentials: QISKIT_IBM_BACKEND if set, else the least busy one."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    load_env()
    service = QiskitRuntimeService(channel=os.environ.get("QISKIT_IBM_CHANNEL") or "ibm_cloud",
                                   token=os.environ["QISKIT_IBM_TOKEN"],
                                   instance=os.environ["QISKIT_IBM_INSTANCE"])
    name = os.environ.get("QISKIT_IBM_BACKEND")
    return service.backend(name) if name else service.least_busy(operational=True, simulator=False)


def run_sampler(circuits, backend, shots, out=None, label=""):
    """Transpile (level 3) and sample on a backend with dynamical decoupling + gate twirling.

    Works for real IBM devices and for local fake/Aer backends (where the error-suppression
    options are ignored). Saves counts, device, settings and circuit resources to JSON.
    """
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=1)
    isa = []
    for qc in circuits:
        qc = qc.copy()
        qc.measure_all()
        isa.append(pm.run(qc))
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = shots
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.num_randomizations = "auto"
    t0 = time.time()
    job = sampler.run(isa)
    res = job.result()
    record = dict(label=label, backend=backend.name, job_id=job.job_id(), shots=shots,
                  wall_s=round(time.time() - t0, 1),
                  utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  options=dict(optimization_level=3, dynamical_decoupling="XY4", twirling_gates=True),
                  circuits=[dict(two_qubit_gates=sum(v for g, v in c.count_ops().items() if g in ("cz", "ecr", "cx")),
                                 depth=c.depth(), layout=c.layout.final_index_layout() if c.layout else None)
                            for c in isa],
                  counts=[{str(k): v for k, v in r.data.meas.get_int_counts().items()} for r in res])
    try:
        record["usage_s"] = job.usage()
    except Exception:  # local fake backends have no usage metering
        record["usage_s"] = None
    if out:
        RESULTS.mkdir(exist_ok=True)
        Path(out).write_text(json.dumps(record, indent=1))
    return record
