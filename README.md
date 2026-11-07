# WakeWise NL: wake-aware wind-turbine siting for Newfoundland's west coast with QAOA

**Team AI Entanglers** (Mostafa Wasib and Mahmoud Al-Moghrabi) · Qiskit Fall Fest 2026 · Memorial University · *Quantum for a Sustainable Newfoundland and Labrador*

Wind turbines slow the air behind them (their *wake*), so turbines placed in each other's wakes waste wind. On a parcel at Wreckhouse, NL, beside the Codroy Valley where a large wind-to-hydrogen project is proposed, we ask which candidate sites should get a turbine so the farm delivers the most net clean energy, whether that choice holds up across **19 years of real wind**, and what protecting a sensitive wetland costs. Using real hourly wind from Environment and Climate Change Canada and **DTU PyWake** (an industry-standard wind-farm simulator), every wake loss is between a *pair* of turbines, so the problem is a QUBO that maps one-to-one onto an Ising Hamiltonian: turbine value = Z field, wake loss = ZZ coupling. We solve it with **QAOA in Qiskit** and test it against exact enumeration, a greedy planner and random guessing on the same instances: noiseless, under an IBM noise model, and on real hardware, IBM Quantum System One in Quebec (`ibm_quebec`), with repeated runs.

**Start here:** [`wakewise.ipynb`](wakewise.ipynb). All results are saved in it; its first section documents environment, data, seeds and how to reproduce everything. Slides: [`slides/`](slides/).

## Results at a glance

| | |
|---|---|
| Real input | 26,264 hours of 2023-2025 Wreckhouse wind for the design; 163,068 hours (2007-2025) to test it. One 4 MW turbine alone ≈ 17,600 MWh/yr (50% capacity factor) |
| Physics | DTU PyWake: a badly placed neighbour costs up to 2.4% of a turbine's output; a dense 12-turbine block loses 4.2% |
| **Recommended layout** | 9 turbines on the 12-site parcel, ≈ 154 GWh/yr (≈ 9,100 NL homes). Also the optimum for the full 19-year climate, and the best possible layout in **14 of 19** single years (average shortfall 19 MWh/yr). A 2025-only design: best in 5 of 19 (average shortfall 104 MWh/yr) |
| Classical baseline | Greedy finds the 3-year optimum but misses the best layout in 16 of 19 single years (up to 8.2% short, 704 MWh/yr) |
| QAOA, noiseless, 12 qubits | Depth 3 with CVaR training returns the optimal layout in 8.4% of shots (≈ 345× random): 34 shots for 95% confidence vs 12,269. Across the 19 single years, 1 in 5 samples beat greedy on average where greedy misses |
| **Real hardware: `ibm_quebec`** (10 jobs, 88 s of QPU) | 3-year climate, 4 independent runs: the optimum ≈ 23× more often than random. 2023 wind, 4 runs: 15-18% of samples beat the greedy planner (random control: 2%). Readout-error mitigation: up to 3.2× on the deepest circuit. 16 qubits: clearly better than random (gap closed up to 0.42) but the optimum was not sampled |
| Model check | QUBO ranks layouts like full PyWake simulations (rank correlation 0.99 with Jensen and Gaussian wake models); its optimum is best under both |
| Wetland scenario | Keeping turbines out of an illustrative wetland buffer costs 519 MWh/yr; QAOA finds the new optimum 133× more often than random |

No quantum advantage is claimed: at these sizes the exact answer is computable, which is what lets us measure QAOA's quality honestly.

## Run it

```bash
git clone https://github.com/mrw-soumik/Qiskit-AI-Entanglers-.git && cd Qiskit-AI-Entanglers-
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python test_wakewise.py                                                   # self-checks
jupyter nbconvert --to notebook --execute --inplace wakewise.ipynb        # ~20 min, no IBM account needed
```

To rerun a hardware experiment: `cp .env.example .env` (or create `api.json`), paste your IBM Quantum `apikey` and `crn`, then
`python wakewise.py hardware climate_12 3` (add `--readout` for readout-calibration circuits) and re-execute the notebook. `.env` and `api.json` are gitignored.

## Files

| Path | What |
|---|---|
| `wakewise.ipynb` | Working notebook with saved outputs |
| `wakewise.py` | Wind data, PyWake wind-farm model, QUBO, Ising mapping, baselines, QAOA (incl. CVaR), readout mitigation and IBM hardware helpers |
| `test_wakewise.py` | Self-checks (Ising energy equals QUBO for every bitstring, bit order, wake physics, CVaR, readout mitigation, scoring) |
| `data/` | Frozen ECCC wind data for 2007-2025 and its provenance (`data/SOURCE.md`) |
| `results/` | Noise-model and hardware runs (device, job ID, options, calibration snapshot, counts) and `summary.json` (every number in the slides) |
| `figures/`, `slides/` | Figures and the presentation (PPTX and PDF) |

## Credits

QAOA workflow pattern adapted from Tinh Bui's *Quantum Optimization* workshop notebook (Qiskit Fall Fest 2026); IBM runtime setup from IBM's *Qiskit 101* notebook (Apache 2.0) and the Fall Fest IBM hardware example; Qiskit, Qiskit Aer, Qiskit IBM Runtime; DTU PyWake (MIT); ECCC hourly climate data (Open Government Licence, Canada); Jensen/Park and Bastankhah-Porte-Agel wake models; QAOA (Farhi et al., 2014); CVaR objective (Barkoutsos et al., 2020). Full credits and our own contribution: Section 12 of the notebook.
