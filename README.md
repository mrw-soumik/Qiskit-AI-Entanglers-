# WakeWise NL: wake-aware wind-turbine siting for Newfoundland's west coast with QAOA

**Team AI Entanglers** (Mostafa Wasib) · Qiskit Fall Fest 2026 · Memorial University · *Quantum for a Sustainable Newfoundland and Labrador*

Wind turbines slow the air behind them (their *wake*), so turbines placed in each other's wakes waste wind. On a parcel at Wreckhouse, NL, beside the Codroy Valley where a large wind-to-hydrogen project is proposed, we ask which candidate sites should get a turbine so the farm delivers the most net clean energy. Using **real 2025 hourly wind** (Environment and Climate Change Canada) and **DTU PyWake** (an industry-standard wind-farm simulator), every wake loss is between a *pair* of turbines, so the problem is a QUBO that maps one-to-one onto an Ising Hamiltonian: turbine value = Z field, wake loss = ZZ coupling. We solve it with **QAOA in Qiskit** and test it against exact enumeration, a greedy planner and random guessing on the same instances: noiseless, under realistic noise models of IBM hardware (including IBM Quantum System One in Quebec), and on IBM quantum hardware.

**Start here:** [`wakewise.ipynb`](wakewise.ipynb). All results are saved in it; its first section documents environment, data, seeds and how to reproduce everything. Slides: [`slides/`](slides/).

## Results at a glance

| | |
|---|---|
| Real input | 8,740 hours of 2025 Wreckhouse wind; one 4 MW turbine alone ≈ 18,500 MWh/yr (53% capacity factor) |
| Physics | DTU PyWake: a badly placed neighbour costs up to 1.9% of a turbine's output; a dense 12-turbine block loses 3.6% |
| Classical | 12-site optimum builds 10 turbines (≈ 180 GWh/yr, ≈ 10,600 NL homes); greedy falls 195 MWh/yr short; greedy is optimal on 9 sites |
| QAOA, noiseless, 12 qubits | depth 3 with CVaR training returns the optimal layout in 9.2% of shots (≈ 380× random): 31 shots find it with 95% confidence vs 12,269 for random guessing |
| **Real hardware: IBM Quantum System One in Quebec** (`ibm_quebec`, 156 qubits) | 12 qubits: the optimal layout in 1.65% of shots, **68× random guessing**; 9 qubits: 10.3% of shots (53× random). A control circuit that should be random is (gap closed ≈ 0). 7 s of QPU time per job; job IDs in `results/hardware_*.json` |
| IBM Heron noise model (simulator) | predicts the hardware results closely (Section 6); dropping weak couplings cuts two-qubit gates by about 40% |
| Model check | QUBO ranks layouts like full PyWake simulations (rank correlation 0.99 with Jensen and Gaussian wake models); its optimum is best under both |

No quantum advantage is claimed: at these sizes the exact answer is computable, which is what lets us measure QAOA's quality honestly.

## Run it

```bash
git clone https://github.com/mrw-soumik/Qiskit-AI-Entanglers-.git && cd Qiskit-AI-Entanglers-
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python test_wakewise.py                                                   # self-checks
jupyter nbconvert --to notebook --execute --inplace wakewise.ipynb        # ~2-4 min, no IBM account needed
```

To rerun the hardware experiment: `cp .env.example .env`, paste your IBM Quantum `apikey` and `crn`, then
`python wakewise.py hardware` and re-execute the notebook. `.env` is gitignored.

## Files

| Path | What |
|---|---|
| `wakewise.ipynb` | Working notebook with saved outputs |
| `wakewise.py` | Wind data, PyWake wind-farm model, QUBO, Ising mapping, baselines, QAOA (incl. CVaR) and IBM hardware helpers |
| `test_wakewise.py` | Self-checks (Ising energy equals QUBO for every bitstring, bit order, wake physics, CVaR, scoring) |
| `data/` | Frozen ECCC wind data and its provenance (`data/SOURCE.md`) |
| `results/` | Noise-model and hardware runs (device, job ID, options, calibration snapshot, counts) and `summary.json` (every number in the slides) |
| `figures/`, `slides/` | Figures and the presentation (PPTX and PDF) |

## Credits

QAOA workflow pattern adapted from Tinh Bui's *Quantum Optimization* workshop notebook (Qiskit Fall Fest 2026); IBM runtime setup from IBM's *Qiskit 101* notebook (Apache 2.0) and the Fall Fest IBM hardware example; Qiskit, Qiskit Aer, Qiskit IBM Runtime; DTU PyWake (MIT); ECCC hourly climate data (Open Government Licence, Canada); Jensen/Park and Bastankhah-Porte-Agel wake models; QAOA (Farhi et al., 2014); CVaR objective (Barkoutsos et al., 2020). Full credits and our own contribution: Section 10 of the notebook.
