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


def wake_model(kind="jensen", site=None):
    """PyWake wind-farm model: 'jensen' (Jensen/Park top-hat wake, used to build the QUBO) or
    'gaussian' (Bastankhah & Porte-Agel 2014, a more modern model used only as an independent check).
    site: PyWake site; default a uniform (flat) site, see `terrain_model` for real terrain."""
    from py_wake.site import UniformSite
    site = UniformSite(ti=0.1) if site is None else site
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


def build_qubo(xy, wd, u, cost_frac=0.88, cable_frac=0.02, model=None, E_ref=None):
    """QUBO in MWh/yr: f(x) = lin . x + sum_{i<j} W_ij x_i x_j  (lower is better).

    lin_i = -(E_i - c_i): free-stream energy of one turbine at site i minus its annualized cost, where cost is
            cost_frac * E_ref plus cable_frac * E_ref per grid spacing of cable to the substation at site 0
            (illustrative economics expressed as energy; E_ref = mean E_i unless given). On a flat site every
            E_i is the same; terrain speed-ups make them differ.
    W_ij  = energy the pair loses to each other's wakes when both i and j are built, simulated with
            PyWake on that pair alone (E_i + E_j minus the pair's energy).
    """
    model = model or wake_model("jensen")
    n = len(xy)
    E = np.array([farm_energy(model, xy[[i]], wd, u)[0] for i in range(n)])
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            W[i, j] = W[j, i] = E[i] + E[j] - farm_energy(model, xy[[i, j]], wd, u).sum()
    if E_ref is None:  # flat site: use E itself, since a mean of equal floats can differ in the last bit
        E_ref = E[0] if np.ptp(E) == 0 else E.mean()
    spacing = np.min(np.hypot(*(xy[1:] - xy[0]).T))
    cost = cost_frac * E_ref + cable_frac * E_ref * np.hypot(*(xy - xy[0]).T) / spacing
    return -(E - cost), W, dict(E=E_ref, E_site=E, cost=cost)


def layout_net(x, xy, wd, u, cost, model):
    """Net value (MWh/yr) of a layout with every wake acting at once: a check of the pairwise QUBO."""
    on = np.flatnonzero(x)
    return farm_energy(model, xy[on], wd, u).sum() - cost[on].sum()


# ---------------------------------------------------------------- real terrain (screening level)
STATION = (47.7117, -59.3083)    # Wreckhouse climate station (lat, lon), ECCC climate ID 8404343
ESTUARY = (47.8458, -59.3133)    # Grand Codroy Estuary, Ramsar site 364
BOUNDS = (-59.55, 47.60, -59.00, 47.95)  # lon/lat window read from the rasters (west, south, east, north)
GWA_URL = "https://gwa.cdn.nazkamapps.com/country_tifs_v4/CAN_wind-speed_100m.tif"   # Global Wind Atlas, 100 m
DEM_URL = ("https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_N47_00_W060_00_DEM/"
           "Copernicus_DSM_COG_30_N47_00_W060_00_DEM.tif")                           # Copernicus DEM GLO-90
GWA_TIF, DEM_TIF = DATA / "gwa_wreckhouse_ws100.tif", DATA / "dem_wreckhouse_glo90.tif"
KX, KY = 111320 * np.cos(np.radians(STATION[0])), 110540  # metres per degree of longitude / latitude here


def fetch_raster(url, path):
    """Freeze the BOUNDS window of a cloud-optimized GeoTIFF to `path` (HTTP range reads; needs network)."""
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(url) as src:
        w = from_bounds(*BOUNDS, src.transform)
        a = src.read(1, window=w)
        prof = dict(driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype=a.dtype, crs=src.crs,
                    transform=src.window_transform(w), compress="deflate")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(a, 1)


def load_raster(path):
    """(values, lat, lon) at cell centres, latitude ascending."""
    import rasterio
    with rasterio.open(path) as src:
        a, t = src.read(1).astype(float), src.transform
    lat = t.f + t.e * (np.arange(a.shape[0]) + 0.5)
    lon = t.c + t.a * (np.arange(a.shape[1]) + 0.5)
    return a[::-1], lat[::-1], lon


def terrain():
    """Interpolators (lat, lon) -> wind-atlas speed at 100 m, elevation, slope and ruggedness index (RIX).

    RIX (Bowen & Mortensen 1996): share of terrain steeper than 30% within 3.5 km, here the area-based version.
    Linear flow models such as WAsP (used inside the wind atlas) transfer wind reliably between two places when
    their RIX values are similar.
    """
    from scipy.interpolate import RegularGridInterpolator as RGI
    from scipy.signal import fftconvolve
    ws, glat, glon = load_raster(GWA_TIF)
    z, lat, lon = load_raster(DEM_TIF)
    hy, hx = np.diff(lat).mean() * KY, np.diff(lon).mean() * KX
    slope = np.hypot(*np.gradient(z, hy, hx))
    ny, nx = int(3500 / hy), int(3500 / hx)
    yy, xx = np.mgrid[-ny:ny + 1, -nx:nx + 1]
    disk = ((yy / ny) ** 2 + (xx / nx) ** 2 <= 1).astype(float)
    rix = fftconvolve((slope > 0.3).astype(float), disk / disk.sum(), mode="same")
    f = lambda a, la, lo: RGI((la, lo), a)
    return dict(ws=f(ws, glat, glon), z=f(z, lat, lon), slope=f(slope, lat, lon), rix=f(rix, lat, lon),
                maps=dict(ws=(ws, glat, glon), z=(z, lat, lon)))


def terrain_sites(n=24, spacing=5 * D, radius=10_000, min_elev=10.0, max_slope=0.10, max_drix=0.05, buffer=3000):
    """Candidate turbine sites on real terrain near Wreckhouse, chosen by rule (screening level, not micrositing).

    Lattice with `spacing` anchored at the station; keep points within `radius` that are on land (elevation above
    `min_elev`), buildable (slope <= `max_slope`), in terrain like the station's (|RIX - RIX_station| <= max_drix,
    so the station's wind transfers with the wind atlas's flow model) and at least `buffer` from the estuary.
    The parcel grows from the windiest point by adding the closest remaining point (ties: the windier), so the
    first 12, 16 and 20 sites are nested sub-parcels of the 24. Site 0 hosts the substation.
    """
    T = terrain()
    k = int(radius // spacing)
    X, Y = [a.ravel() for a in np.meshgrid(np.arange(-k, k + 1) * spacing, np.arange(-k, k + 1) * spacing)]
    lat, lon = STATION[0] + Y / KY, STATION[1] + X / KX
    P = np.c_[lat, lon]
    ws, z, slope, rix = (T[q](P) for q in ("ws", "z", "slope", "rix"))
    rix0, ws0 = T["rix"]([STATION])[0], T["ws"]([STATION])[0]
    to_estuary = np.hypot(X - (ESTUARY[1] - STATION[1]) * KX, Y - (ESTUARY[0] - STATION[0]) * KY)
    ok = (np.hypot(X, Y) <= radius) & (z > min_elev) & (slope <= max_slope) & (np.abs(rix - rix0) <= max_drix) \
        & (to_estuary >= buffer)
    cand = np.flatnonzero(ok)
    order = [cand[np.argmax(ws[cand])]]
    while len(order) < n:
        rest = np.setdiff1d(cand, order)
        d = np.min(np.hypot(X[rest, None] - X[order], Y[rest, None] - Y[order]), axis=1)
        near = rest[np.isclose(d, d.min())]
        order.append(near[np.argmax(ws[near])])
    o = np.array(order)
    return pd.DataFrame(dict(x=X[o], y=Y[o], lat=lat[o], lon=lon[o], elevation=z[o], slope=slope[o], rix=rix[o],
                             gwa_ws=ws[o], speedup=ws[o] / ws0)), dict(candidates=int(ok.sum()), rix_station=rix0,
                                                                         gwa_station=ws0)


def terrain_model(sites, kind="jensen"):
    """PyWake model on an XRSite whose Speedup is the wind-atlas ratio GWA(x, y) / GWA(station) on a 100 m grid
    around the sites (x east, y north, metres from the station): the station's hourly wind, scaled for terrain."""
    import xarray as xr
    from py_wake.site import XRSite
    T = terrain()
    xs = np.arange(sites.x.min() - 1000, sites.x.max() + 1001, 100.0)
    ys = np.arange(sites.y.min() - 1000, sites.y.max() + 1001, 100.0)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    up = T["ws"](np.c_[STATION[0] + YY.ravel() / KY, STATION[1] + XX.ravel() / KX]).reshape(XX.shape) / T["ws"]([STATION])[0]
    ds = xr.Dataset({"Speedup": (["x", "y"], up), "P": 1, "TI": 0.1}, coords={"x": xs, "y": ys})
    return wake_model(kind, XRSite(ds, interp_method="linear"))


# Named test instances: (wind years, grid rows, grid cols). "climate" = all of YEARS.
INSTANCES = {"climate_12": (YEARS, 3, 4), "climate_16": (YEARS, 4, 4), "y2023_12": ((2023,), 3, 4),
             "y2024_12": ((2024,), 3, 4), "y2025_12": ((2025,), 3, 4)}
TERRAIN = {f"terrain_{n}": n for n in (12, 16, 20, 24)}  # nested real-terrain parcels, 3-year wind climate


def instance(key, model=None, cost_frac=0.88, years=None):
    """Build a named instance: wind, PyWake QUBO, exact optimum, greedy answer and the random-guess baseline.
    Terrain instances share one economics (E_ref = mean free-stream energy of all 24 parcel sites)."""
    if key in TERRAIN:
        years, sites = years or YEARS, terrain_sites()[0]
        wd, u = load_wind(years)
        model = model or terrain_model(sites)
        xy_all = sites[["x", "y"]].to_numpy()
        E_ref = np.mean([farm_energy(model, xy_all[[i]], wd, u)[0] for i in range(len(xy_all))])
        xy, rows, cols = xy_all[:TERRAIN[key]], None, None
        lin, W, info = build_qubo(xy, wd, u, cost_frac=cost_frac, model=model, E_ref=E_ref)
    else:
        years, rows, cols = INSTANCES[key]
        wd, u = load_wind(years)
        xy = grid(rows, cols, 5 * D)
        lin, W, info = build_qubo(xy, wd, u, cost_frac=cost_frac, model=model)
    x_opt, f_opt, values = exact(lin, W)
    x_gr = greedy(lin, W)
    return dict(key=key, years=years, wd=wd, u=u, xy=xy, rows=rows, cols=cols, lin=lin, W=W, E=info["E"], cost=info["cost"],
                E_site=info["E_site"], x_opt=x_opt, f_opt=f_opt, values=values, f_rand=values.mean(), x_gr=x_gr,
                f_gr=qubo_values(x_gr, lin, W)[0], p_rand=float(np.isclose(values, f_opt).mean()))


# ---------------------------------------------------------------- QUBO tools and baselines
def bits(states, n):
    """Integer outcomes -> 0/1 rows. Bit i of the integer is site i (Qiskit's little-endian order)."""
    return ((np.asarray(states)[:, None] >> np.arange(n)) & 1).astype(float)


def qubo_values(X, lin, W):
    X = np.atleast_2d(X)
    return X @ lin + 0.5 * np.einsum("mi,ij,mj->m", X, W, X)


def exact(lin, W, chunk=2 ** 18):
    """Brute force over all 2^n layouts, in chunks (2^24 layouts fit in memory). Returns (best x, best value,
    all values indexed by integer)."""
    n = len(lin)
    v = np.concatenate([qubo_values(bits(np.arange(s, min(s + chunk, 2 ** n)), n), lin, W)
                        for s in range(0, 2 ** n, chunk)])
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


def ising_diagonal(lin, W, values=None):
    """Energy of every basis state under the normalized Ising Hamiltonian H / max|coeff| that the QAOA circuits use
    (H from `qubo_to_ising(lin, W)`), indexed by measured integer. `values` = QUBO values of (lin, W) if known."""
    H, offset = qubo_to_ising(lin, W)
    values = exact(lin, W)[2] if values is None else values
    return (values - offset) / np.abs(H.coeffs).max()


def qaoa_probabilities(diag, params):
    """Exact QAOA outcome probabilities straight from the cost diagonal, no circuit: |+>^n, then per layer
    exp(-i gamma H) and exp(-i beta X) on every qubit. `params` in qaoa_ansatz order (betas, then gammas); equals the
    statevector of `qaoa_circuit` (tested), but fast enough for 24 qubits on a laptop."""
    n, p = int(np.log2(len(diag))), len(params) // 2
    psi = np.full(2 ** n, 2 ** (-n / 2), complex)
    for b, g in zip(params[:p], params[p:]):
        psi *= np.exp(-1j * g * diag)
        c, s = np.cos(b), -1j * np.sin(b)
        for q in range(n):  # middle axis = bit q of the index (Qiskit's qubit q)
            v = psi.reshape(-1, 2, 2 ** q)
            a0 = v[:, 0, :].copy()
            v[:, 0, :] = c * a0 + s * v[:, 1, :]
            v[:, 1, :] = s * a0 + c * v[:, 1, :]
    return np.abs(psi) ** 2


def train_cvar(diag, values, p, alpha=0.25, starts=5, seed=7, maxiter=300):
    """CVaR training like `train_qaoa`, on the fast simulator (same seeded starting points)."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    cost = lambda th: cvar(qaoa_probabilities(diag, th), values, alpha)
    runs = [minimize(cost, rng.uniform(0, np.pi, 2 * p), method="COBYLA", options={"maxiter": maxiter})
            for _ in range(starts)]
    return min(runs, key=lambda r: r.fun)


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


def angle_circuits(lin, W, angles, keep=0.25):
    """Hardware circuits from given angles (e.g. the AI predictor's) on the sparsified Hamiltonian: depth 1 and 2,
    the uniform control (all angles 0) and a structure-matched control: the depth-1 circuit with the mixer angle's
    sign flipped, which has the same gates but steers away from good layouts (worse than random when noiseless)."""
    H, _ = qubo_to_ising(lin, sparsify(W, keep))
    H = H / np.abs(H.coeffs).max()
    p1, p2 = np.asarray(angles["p1"], float), np.asarray(angles["p2"], float)
    flipped = np.r_[-p1[0], p1[1]]
    circuits = [qaoa_circuit(H, 1).assign_parameters(p1), qaoa_circuit(H, 2).assign_parameters(p2),
                qaoa_circuit(H, 1).assign_parameters([0.0, 0.0]), qaoa_circuit(H, 1).assign_parameters(flipped)]
    return circuits, ["p1_sparse", "p2_sparse", "control_uniform", "control_flipped"]


# Qubits our own calibration circuits found faulty: ibm_quebec qubit 101 read |1> as 0 in 68% of shots on 7 Nov 2026
# (jobs dbnncp8qtr0c73adbsi0 and dbnnd7k4148s7391ahe0), while the device's reported readout error looked normal.
FAULTY = {"ibm_quebec": {101}}


def bad_qubits(backend, max_readout=0.05, max_2q=0.05):
    """Physical qubits to keep out of a layout: readout or two-qubit-gate error above the limits in the device's
    current calibration, plus qubits in FAULTY."""
    t = backend.target
    bad = {q for (q,), p in t["measure"].items() if p is not None and p.error is not None and p.error > max_readout}
    two_q = next((g for g in ("cz", "ecr", "cx") if g in t.operation_names), None)
    if two_q:
        bad |= {q for qs, p in t[two_q].items() if p is not None and p.error is not None and p.error > max_2q for q in qs}
    return bad | FAULTY.get(backend.name, set())


def readout_cal_circuits(isa_circuits, backend):
    """Two calibration circuits per transpiled circuit, prepared all-0 and all-1 on the same physical qubits it measures."""
    from qiskit import QuantumCircuit, transpile
    cals = []
    for c in isa_circuits:
        phys = c.layout.final_index_layout()
        for flip in (0, 1):
            qc = QuantumCircuit(len(phys))
            if flip:
                qc.x(range(len(phys)))
            qc.measure_all()
            cals.append(transpile(qc, backend, initial_layout=phys, optimization_level=0))
    return cals


def mitigate_readout(counts, cal0, cal1, n):
    """Tensored readout-error mitigation: estimate each qubit's flip rates from the all-0 and all-1 calibration
    counts, then apply the inverse of each qubit's 2x2 assignment matrix to the measured distribution.
    Returns quasi-probabilities indexed by measured integer (they can dip slightly below zero)."""
    bit = lambda s: (int(s) >> np.arange(n)) & 1
    p10 = sum(c * bit(s) for s, c in cal0.items()) / sum(cal0.values())        # P(read 1 | prepared 0)
    p01 = sum(c * (1 - bit(s)) for s, c in cal1.items()) / sum(cal1.values())  # P(read 0 | prepared 1)
    p = np.zeros(2 ** n)
    for s, c in counts.items():
        p[int(s)] += c
    t = (p / p.sum()).reshape([2] * n)  # axis n-1-i holds qubit i (bit i of the integer)
    for i in range(n):
        A = np.array([[1 - p10[i], p01[i]], [p10[i], 1 - p01[i]]])  # columns: prepared 0/1, rows: read 0/1
        ax = n - 1 - i
        t = np.moveaxis(np.tensordot(np.linalg.inv(A), t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


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


def run_sampler(circuits, backend, shots, out=None, names=None, meta=None, seed=None, readout_cal=False, avoid=()):
    """Transpile (level 3) and sample on a backend with dynamical decoupling + gate twirling.

    Works for real IBM devices and for local fake backends (which ignore the error-suppression
    options; `seed` fixes their noise sampling). Saves counts, device, calibration snapshot,
    settings and per-circuit resources to JSON so results can be inspected without rerunning.
    readout_cal=True adds all-0 / all-1 calibration circuits on each circuit's physical qubits to the same job
    (stored under "readout_cal") for readout-error mitigation with `mitigate_readout`.
    avoid: physical qubits no gate may touch (see `bad_qubits`). They get error 1 in a copy of the device target, so
    the noise-aware layout steps steer around them; the transpiler seed is raised from 1 if routing still uses one.
    """
    from qiskit.transpiler import InstructionProperties, generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2
    target = None
    if avoid:
        import copy
        target = copy.deepcopy(backend.target)
        for op in target.operation_names:
            for qargs, props in list(target[op].items()):
                if qargs and props is not None and set(qargs) & set(avoid):
                    target.update_instruction_properties(op, qargs, InstructionProperties(duration=props.duration, error=1.0))
    isa, seeds = [], []
    for qc in circuits:
        qc = qc.copy()
        qc.measure_all()
        for s in range(1, 50):
            c = generate_preset_pass_manager(optimization_level=3, backend=backend, target=target, seed_transpiler=s).run(qc)
            if not {c.find_bit(q).index for inst in c.data for q in inst.qubits} & set(avoid):
                break
        else:
            raise RuntimeError("no transpiler seed avoids the excluded qubits")
        isa.append(c)
        seeds.append(s)
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = shots
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.num_randomizations = "auto"
    if seed is not None:
        sampler.options.simulator.seed_simulator = seed
    cals = readout_cal_circuits(isa, backend) if readout_cal else []
    t0 = time.time()
    job = sampler.run(isa + cals)
    res = list(job.result())
    res, cal_res = res[:len(isa)], res[len(isa):]
    record = dict(backend=backend.name, job_id=job.job_id(), shots=shots, wall_s=round(time.time() - t0, 1),
                  utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), calibration=device_errors(backend),
                  options=dict(optimization_level=3, seed_transpiler=seeds[0] if len(set(seeds)) == 1 else seeds,
                               dynamical_decoupling="XY4", twirling_gates=True, num_randomizations="auto",
                               seed_simulator=seed, avoided_qubits=sorted(avoid)),
                  names=names or [f"circuit_{i}" for i in range(len(isa))], meta=meta or {},
                  circuits=[dict(two_qubit_gates=sum(v for g, v in c.count_ops().items() if g in ("cz", "ecr", "cx")),
                                 depth=c.depth(), physical_qubits=c.layout.final_index_layout() if c.layout else None)
                            for c in isa],
                  counts=[{str(k): v for k, v in r.data.meas.get_int_counts().items()} for r in res])
    if readout_cal:
        record["readout_cal"] = [{str(k): v for k, v in r.data.meas.get_int_counts().items()} for r in cal_res]
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
    # `python wakewise.py hardware <instance> [repeats] [--readout] [--healthy] [--ai]`: run that instance's circuits
    # on IBM hardware (credentials from api.json or .env) and save results/hardware_<instance>_<tag>.json, which the
    # notebook loads. --readout adds readout-calibration circuits (tag rm<k>); --healthy keeps faulty qubits out of
    # the layout (tag h<k>); --ai uses the AI-predicted angles in results/ai_angles.json (tag ai<k>, implies
    # --healthy). Waits for the queue, however long.
    import sys
    if sys.argv[1:2] == ["hardware"]:
        flags = {a for a in sys.argv[2:] if a.startswith("--")}
        args = [a for a in sys.argv[2:] if not a.startswith("--")]
        key, repeats = args[0], int(args[1]) if len(args) > 1 else 1
        inst = instance(key)
        if "--ai" in flags:
            angles = json.loads((RESULTS / "ai_angles.json").read_text())[key]
            circuits, names = angle_circuits(inst["lin"], inst["W"], angles)
            params = angles
        else:
            circuits, names, params = hardware_circuits(inst["lin"], inst["W"], inst["values"])
        backend = hardware_backend()
        avoid = bad_qubits(backend) if flags & {"--healthy", "--ai"} else set()
        prefix = "ai" if "--ai" in flags else "h" if "--healthy" in flags else "r"
        for r in range(1, repeats + 1):
            tag = f"{prefix}m{r}" if "--readout" in flags else f"{prefix}{r}"  # m = also carries readout calibration
            print(f"submitting {key} run {tag} to {backend.name}, avoiding {len(avoid)} qubits", flush=True)
            rec = run_sampler(circuits, backend, shots=4000, names=names, meta=dict(instance=key, run=tag, params=params),
                              out=RESULTS / f"hardware_{key}_{tag}.json", readout_cal="--readout" in flags, avoid=avoid)
            print("done:", rec["backend"], rec["job_id"], "QPU s:", rec["usage_s"], flush=True)
