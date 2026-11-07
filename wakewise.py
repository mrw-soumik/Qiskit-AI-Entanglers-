"""WakeWise NL: wake-aware wind-turbine siting for Newfoundland's west coast, solved with QAOA.

Classical side: real ECCC hourly wind -> DTU PyWake wind-farm simulation (Jensen wake model,
generic 4 MW turbine) -> QUBO -> exact / greedy / random baselines.
Quantum side: QUBO -> Ising Hamiltonian -> QAOA (simulator, IBM noise models, IBM hardware).
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

D, HUB, RATED_KW = 130.0, 110.0, 4000  # rotor diameter (m), hub height (m), rated power
WAKE_K = 0.075                          # Jensen wake-expansion constant, common onshore value
HOURS = 8760
YEARS = (2023, 2024, 2025)            # Wreckhouse hourly wind years used as the long-term climate

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


def load_wind(years=None, hub=HUB, alpha=0.14):
    """Per-hour wind direction (degrees, blowing FROM) and hub-height speed (m/s) for the given years
    (default: all of YEARS, i.e. a multi-year wind climate with every hour weighted equally).

    ECCC gives direction in tens of degrees (0 = calm) and 10 m speed in km/h. Hours with a
    missing value are dropped; calm hours stay in with zero speed so the energy average is fair.
    The 10 m -> hub extrapolation uses a power-law shear profile (alpha is an assumption).
    """
    df = pd.concat([pd.read_csv(DATA / f"wreckhouse_{y}_hourly.csv") for y in (years or YEARS)])
    ok = df.WIND_DIRECTION.notna() & df.WIND_SPEED.notna()
    wd = df.WIND_DIRECTION[ok].to_numpy(float) * 10
    u = df.WIND_SPEED[ok].to_numpy(float) / 3.6 * (hub / 10) ** alpha
    return wd, u


# ---------------------------------------------------------------- wind-farm physics (DTU PyWake)
def turbine():
    """Generic 4 MW, 130 m rotor, 110 m hub turbine from PyWake's physics-based generic model
    (realistic power and thrust curves: thrust falls above rated wind speed, as on real turbines)."""
    from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
    return GenericWindTurbine("Generic 4 MW", diameter=D, hub_height=HUB, power_norm=RATED_KW,
                              turbulence_intensity=0.1)


def wake_model(kind="jensen"):
    """PyWake wind-farm model: 'jensen' (Jensen/Park top-hat wake, used to build the QUBO) or
    'gaussian' (Bastankhah & Porte-Agel 2014, a more modern model used only as an independent check)."""
    from py_wake.site import UniformSite
    site = UniformSite(ti=0.1)
    if kind == "jensen":
        from py_wake import NOJ
        return NOJ(site, turbine(), k=WAKE_K)
    from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
    return Bastankhah_PorteAgel_2014(site, turbine(), k=0.0324555)


def farm_energy(model, xy, wd, u):
    """Annual energy (MWh/yr) of each turbine placed at xy, simulated hour by hour with all wakes."""
    if len(xy) == 0:
        return np.zeros(0)
    sim = model(xy[:, 0], xy[:, 1], wd=wd, ws=u, time=True)
    return sim.Power.values.mean(axis=1) * HOURS / 1e6


def grid(rows, cols, spacing):
    """Site coordinates in metres (x = east, y = north); site index = row * cols + col."""
    return np.array([(c * spacing, r * spacing) for r in range(rows) for c in range(cols)], float)


def build_qubo(xy, wd, u, cost_frac=0.88, cable_frac=0.02, model=None):
    """QUBO in MWh/yr: f(x) = lin . x + sum_{i<j} W_ij x_i x_j  (lower is better).

    lin_i = -(E - c_i): free-stream energy of one turbine minus its annualized cost, where cost is
            cost_frac * E plus cable_frac * E per grid spacing of cable to the substation at site 0
            (illustrative economics expressed as energy).
    W_ij  = energy the pair loses to each other's wakes when both i and j are built, simulated with
            PyWake on that pair alone (2 E minus the pair's energy).
    """
    model = model or wake_model("jensen")
    n = len(xy)
    E = farm_energy(model, xy[:1], wd, u)[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            W[i, j] = W[j, i] = 2 * E - farm_energy(model, xy[[i, j]], wd, u).sum()
    spacing = np.min(np.hypot(*(xy[1:] - xy[0]).T))
    cost = cost_frac * E + cable_frac * E * np.hypot(*(xy - xy[0]).T) / spacing
    return -(E - cost), W, dict(E=E, cost=cost)


def layout_net(x, xy, wd, u, cost, model):
    """Net value (MWh/yr) of a layout with every wake acting at once: a check of the pairwise QUBO."""
    on = np.flatnonzero(x)
    return farm_energy(model, xy[on], wd, u).sum() - cost[on].sum()


# Named test instances: (wind years, grid rows, grid cols). "climate" = all of YEARS.
INSTANCES = {"climate_12": (YEARS, 3, 4), "climate_16": (YEARS, 4, 4), "y2023_12": ((2023,), 3, 4),
             "y2024_12": ((2024,), 3, 4), "y2025_12": ((2025,), 3, 4)}


def instance(key, model=None, cost_frac=0.88):
    """Build a named instance: wind, PyWake QUBO, exact optimum, greedy answer and the random-guess baseline."""
    years, rows, cols = INSTANCES[key]
    wd, u = load_wind(years)
    xy = grid(rows, cols, 5 * D)
    lin, W, info = build_qubo(xy, wd, u, cost_frac=cost_frac, model=model)
    x_opt, f_opt, values = exact(lin, W)
    x_gr = greedy(lin, W)
    return dict(key=key, years=years, wd=wd, u=u, xy=xy, rows=rows, cols=cols, lin=lin, W=W, E=info["E"], cost=info["cost"],
                x_opt=x_opt, f_opt=f_opt, values=values, f_rand=values.mean(), x_gr=x_gr, f_gr=qubo_values(x_gr, lin, W)[0],
                p_rand=float(np.isclose(values, f_opt).mean()))


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


def cvar(probabilities, values, alpha):
    """Mean QUBO value of the best `alpha` share of outcomes (CVaR objective, Barkoutsos et al. 2020)."""
    o = np.argsort(values)
    pc = np.cumsum(probabilities[o])
    k = min(int(np.searchsorted(pc, alpha)), len(o) - 1)
    w = probabilities[o][:k + 1].copy()
    w[-1] -= pc[k] - alpha
    return float(w @ values[o][:k + 1] / alpha)


def train_qaoa(ansatz, H, starts=5, seed=7, maxiter=300, values=None, alpha=None):
    """Tune QAOA angles in exact noiseless simulation with COBYLA from several seeded starts.

    alpha=None: minimize <H> (standard QAOA). alpha in (0, 1]: minimize the CVaR of the QUBO values
    (`values`, indexed by measured integer), which rewards putting probability on the best layouts.
    """
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize
    if alpha is None:
        est = StatevectorEstimator()
        cost = lambda th: float(est.run([(ansatz, H, th)]).result()[0].data.evs)
    else:
        cost = lambda th: cvar(Statevector(ansatz.assign_parameters(th)).probabilities(), values, alpha)
    rng = np.random.default_rng(seed)
    runs = [minimize(cost, rng.uniform(0, np.pi, ansatz.num_parameters), method="COBYLA",
                     options={"maxiter": maxiter}) for _ in range(starts)]
    return min(runs, key=lambda r: r.fun), [float(r.fun) for r in runs]


def ideal_probabilities(circuit, params):
    """Exact outcome probabilities, indexed by measured integer (no sampling noise)."""
    from qiskit.quantum_info import Statevector
    return Statevector(circuit.assign_parameters(params)).probabilities()


def ideal_counts(circuit, params, shots=10_000, seed=11):
    from qiskit.primitives import StatevectorSampler
    qc = circuit.assign_parameters(params)
    qc.measure_all()
    return StatevectorSampler(seed=seed).run([qc], shots=shots).result()[0].data.meas.get_int_counts()


def hardware_circuits(lin, W, values, keep=0.25, alpha=0.25, starts=5, seed=7):
    """The 9-site circuits for the noise models and IBM hardware, trained in noiseless simulation.

    p1_full: p=1 with every coupling; p1_sparse / p2_sparse: couplings >= keep * strongest;
    control_uniform: gamma = beta = 0, which ideally gives uniformly random layouts (noise-floor check).
    """
    W_sparse = sparsify(W, keep)
    circuits, names, params = [], [], {}
    for name, Wc, p in [("p1_full", W, 1), ("p1_sparse", W_sparse, 1), ("p2_sparse", W_sparse, 2)]:
        H, _ = qubo_to_ising(lin, Wc)
        H = H / np.abs(H.coeffs).max()
        ans = qaoa_circuit(H, p)
        best, _ = train_qaoa(ans, H, starts=starts, seed=seed, values=values, alpha=alpha)
        circuits.append(ans.assign_parameters(best.x))
        names.append(name)
        params[name] = best.x.tolist()
    circuits.append(qaoa_circuit(H, 1).assign_parameters([0.0, 0.0]))
    names.append("control_uniform")
    return circuits, names, params


# ---------------------------------------------------------------- IBM hardware
def load_env(path=ROOT / ".env"):
    """Read KEY=VALUE lines from a .env file into os.environ (already-set variables win)."""
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            key, sep, val = line.partition("=")
            if sep and not key.strip().startswith("#"):
                os.environ.setdefault(key.strip(), val.strip().strip("'\""))


def credentials(env_path=ROOT / ".env", json_path=ROOT / "api.json"):
    """(apikey, crn) from api.json (the Fall Fest IBM example's format) or else from .env; both are gitignored."""
    if Path(json_path).exists():
        c = json.loads(Path(json_path).read_text())
        return c["apikey"], c["crn"]
    load_env(env_path)
    env = os.environ.get
    return env("apikey") or env("QISKIT_IBM_TOKEN"), env("crn") or env("QISKIT_IBM_INSTANCE")


def hardware_backend(env_path=ROOT / ".env"):
    """IBM backend via channel ibm_quantum_platform (as in the Fall Fest example); QISKIT_IBM_BACKEND picks a
    device, otherwise the least busy one."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    token, crn = credentials(env_path)
    env = os.environ.get
    service = QiskitRuntimeService(channel=env("QISKIT_IBM_CHANNEL") or "ibm_quantum_platform", token=token, instance=crn)
    name = env("QISKIT_IBM_BACKEND")
    return service.backend(name) if name else service.least_busy(operational=True, simulator=False)


def device_errors(backend):
    """Median two-qubit-gate and readout error of the device at run time (calibration snapshot)."""
    t = backend.target
    two_q = next((g for g in ("cz", "ecr", "cx") if g in t.operation_names), None)
    med = lambda op: float(np.median([p.error for p in t[op].values() if p is not None and p.error is not None]))
    return dict(two_qubit_gate=two_q, median_2q_error=med(two_q) if two_q else None,
                median_readout_error=med("measure"))


def run_sampler(circuits, backend, shots, out=None, names=None, meta=None, seed=None):
    """Transpile (level 3) and sample on a backend with dynamical decoupling + gate twirling.

    Works for real IBM devices and for local fake backends (which ignore the error-suppression
    options; `seed` fixes their noise sampling). Saves counts, device, calibration snapshot,
    settings and per-circuit resources to JSON so results can be inspected without rerunning.
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
    if seed is not None:
        sampler.options.simulator.seed_simulator = seed
    t0 = time.time()
    job = sampler.run(isa)
    res = job.result()
    record = dict(backend=backend.name, job_id=job.job_id(), shots=shots, wall_s=round(time.time() - t0, 1),
                  utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), calibration=device_errors(backend),
                  options=dict(optimization_level=3, seed_transpiler=1, dynamical_decoupling="XY4",
                               twirling_gates=True, num_randomizations="auto", seed_simulator=seed),
                  names=names or [f"circuit_{i}" for i in range(len(isa))], meta=meta or {},
                  circuits=[dict(two_qubit_gates=sum(v for g, v in c.count_ops().items() if g in ("cz", "ecr", "cx")),
                                 depth=c.depth(), physical_qubits=c.layout.final_index_layout() if c.layout else None)
                            for c in isa],
                  counts=[{str(k): v for k, v in r.data.meas.get_int_counts().items()} for r in res])
    try:  # QPU time billed for the job; local fake backends have no usage metering
        usage = job.metrics().get("usage", {})
        record["usage_s"] = usage.get("qpu_charge_time_seconds", usage.get("quantum_seconds"))
        record["timestamps"] = job.metrics().get("timestamps")
    except Exception:
        record["usage_s"] = None
    if out:
        Path(out).parent.mkdir(exist_ok=True)
        Path(out).write_text(json.dumps(record, indent=1))
    return record


if __name__ == "__main__":
    # `python wakewise.py hardware <instance> [repeats]`: run that instance's 4 circuits on IBM hardware
    # (credentials from api.json or .env) and save results/hardware_<instance>_r<k>.json, which the notebook
    # loads. Waits for the queue, however long.
    import sys
    if sys.argv[1:2] == ["hardware"]:
        key, repeats = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1
        inst = instance(key)
        circuits, names, params = hardware_circuits(inst["lin"], inst["W"], inst["values"])
        backend = hardware_backend()
        for r in range(1, repeats + 1):
            print(f"submitting {key} run {r} to {backend.name}", flush=True)
            rec = run_sampler(circuits, backend, shots=4000, names=names, meta=dict(instance=key, run=r, params=params),
                              out=RESULTS / f"hardware_{key}_r{r}.json")
            print("done:", rec["backend"], rec["job_id"], "QPU s:", rec["usage_s"], flush=True)
