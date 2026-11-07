# WakeWise NL: wake-aware wind-turbine siting for Newfoundland's west coast with QAOA

**Team AI Entanglers** · Qiskit Fall Fest 2026 · Memorial University · *Quantum for a Sustainable Newfoundland and Labrador*

Wind turbines slow the air behind them. On a parcel next to Wreckhouse, NL (next to the Codroy Valley, where large wind-to-hydrogen farms are proposed), we ask which candidate sites should get a turbine so the farm delivers the most net clean energy once those wake losses are counted. Every wake loss involves a *pair* of turbines, so the problem is a QUBO that maps one-to-one onto an Ising Hamiltonian (turbine value = Z field, wake loss = ZZ coupling). We solve it with QAOA in Qiskit and test it against exact enumeration, a greedy planner and random guessing on the same instances, in noiseless simulation, a realistic IBM Heron noise model and on IBM quantum hardware.

**Start here:** [`wakewise.ipynb`](wakewise.ipynb) (all results with saved outputs; its first section documents environment, data, seeds and how to reproduce everything).

## Results at a glance

| | |
|---|---|
| Real input | 8,740 hours of 2025 hourly wind at Wreckhouse (Environment and Climate Change Canada) |
| Classical | 12-site optimum builds 6 turbines; a greedy planner misses 9.6% of achievable net energy |
| QAOA, noiseless, 12 qubits | p=3 returns the optimal layout in 1.26% of shots (about 50x random): 236 shots find it with 95% confidence vs 12,269 for random guessing |
| QAOA, Heron noise model, 9 qubits | signal survives (gap closed 0.39 to 0.51 vs 0.44 to 0.57 noiseless); sparsifying weak couplings cuts 2-qubit gates 144 to 64 |
| IBM hardware | see Section 6 of the notebook and `results/hardware_9site.json` |
| Model check | pairwise QUBO ranks layouts like a multi-wake model (Spearman 0.99) |

No quantum advantage is claimed: at these sizes the exact answer is computable, which is what lets us measure QAOA's quality honestly.

## Run it

```bash
git clone https://github.com/mrw-soumik/Qiskit-AI-Entanglers-.git && cd Qiskit-AI-Entanglers-
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python test_wakewise.py                                                   # self-checks
jupyter nbconvert --to notebook --execute --inplace wakewise.ipynb        # ~2 min, no IBM account needed
```

To rerun the hardware experiment: `cp .env.example .env`, add your IBM Quantum API key and instance CRN, then
`RUN_HARDWARE=1 jupyter nbconvert --to notebook --execute --inplace wakewise.ipynb`. `.env` is gitignored.

## Files

| Path | What |
|---|---|
| `wakewise.ipynb` | Working notebook with saved outputs |
| `wakewise.py` | Wind data, wake model, QUBO, Ising mapping, baselines, QAOA and hardware helpers |
| `test_wakewise.py` | Self-checks (Ising energy equals QUBO value for every bitstring, bit order, wake geometry, scoring) |
| `data/` | Frozen ECCC wind data and its provenance (`data/SOURCE.md`) |
| `results/` | Saved noisy-simulation and hardware runs: device, job ID, options, calibration snapshot, counts |
| `figures/` | Figures used in the slides |

## Credits

QAOA workflow pattern adapted from Tinh Bui's *Quantum Optimization* workshop notebook (Qiskit Fall Fest 2026); runtime setup from IBM's *Qiskit 101* hands-on notebook (Apache 2.0); Qiskit, Qiskit Aer, Qiskit IBM Runtime; ECCC hourly climate data (Open Government Licence, Canada); Jensen/Park wake model; QAOA (Farhi, Goldstone and Gutmann, 2014). Full credits and our own contribution are listed in Section 10 of the notebook.
