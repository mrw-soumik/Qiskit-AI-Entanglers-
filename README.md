# WakeWise NL: wake-aware wind-turbine siting for Newfoundland's west coast with QAOA

**Team AI Entanglers** (Mostafa Wasib) · Team 8 · Qiskit Fall Fest 2026 · Memorial University · *Quantum for a Sustainable Newfoundland and Labrador*

Wind turbines slow the air behind them. This *wake* means a turbine standing downwind of another gets less wind, so where you put turbines matters.

**The question.** At Wreckhouse, NL, next to the Codroy Valley where a large wind-to-hydrogen project is proposed: which candidate sites should get a turbine so the farm makes the most clean energy? Does that choice hold up over **19 years of real wind** and on **real terrain**? And how far can today's quantum computers take the search?

**How.** We use real hourly wind from Environment and Climate Change Canada and **DTU PyWake**, an industry-standard wind-farm simulator. Wake losses happen between *pairs* of turbines, so the problem is a QUBO: yes/no choices with pairwise penalties. That maps directly onto qubits: a turbine's value acts on its own qubit (a Z field) and a wake loss links two qubits (a ZZ coupling). We solve it with **QAOA in Qiskit**, and set QAOA's angles without a training loop: our **AI predictor**, tested against simply reusing one good set of angles.

**How we check it.** Every result is compared with the exact best answer, a greedy planner and random guessing: in noiseless simulation, under an IBM noise model, and on **IBM Quantum System One in Quebec** (`ibm_quebec`) from 12 up to **24 qubits**, with repeated runs and control circuits.

**Start here:** [`wakewise.ipynb`](wakewise.ipynb). All results are saved in it; its first section documents environment, data, seeds and how to reproduce everything. Slides: [`slides/`](slides/).

## Results at a glance

| | |
|---|---|
| Real input | 26,264 hours of 2023-2025 Wreckhouse wind for the design; 163,068 hours (2007-2025) to test it. One 4 MW turbine alone ≈ 17,600 MWh/yr (50% capacity factor) |
| **Recommended layout** (flat parcel) | 9 turbines, ≈ 154 GWh/yr (≈ 9,100 NL homes). Best possible in **14 of 19** single years (a 2025-only design: 5 of 19). Greedy misses the best layout in 16 of 19 years |
| QAOA, noiseless, 12 qubits | Depth 3 with CVaR training returns the optimal layout in 8.4% of shots (≈ 345× random) |
| Hardware, flat parcel | 4 independent runs: the optimum ≈ 23× more often than random; 2023 wind: 15%-18% of samples beat greedy (random control: 2%). Calibration circuits exposed a faulty qubit (101, misread 1 as 0 in 65% of shots); faulty qubits are now kept out of layouts |
| **Real terrain** (screening level) | DTU Global Wind Atlas + Copernicus elevation: 24 plateau sites chosen by rule (423-550 m, 9.7-10.9 m/s). Designed on all 19 years: best in 12 of 19 years (2023-2025 design: 4) |
| **AI vs parameter transfer** | On held-out wind years our AI keeps 100% / 92% of trained-angle quality (depth 1 / 2), so no training loop is needed; one transferred angle set keeps 99% / 100%. Honest negative result for the AI |
| **24 qubits on `ibm_quebec`** | Gap closed 0.79 / 0.59 / 0.60 / 0.66 at 12 / 16 / 20 / 24 qubits (noiseless 0.84 / 0.65 / 0.66 / 0.76). At 24 qubits the exact best of 16,777,216 layouts appeared 4× in 24,000 shots (chance: 1 in 16.8 million). Matched control (same gates, one sign flipped): -1.09 to -0.85, below random as physics predicts |
| Model check | QUBO ranks layouts like full PyWake simulations (rank correlation 0.99 with Jensen and 0.99 with Gaussian wake models) |
| Wetland scenario | Keeping turbines out of an illustrative wetland buffer costs 519 MWh/yr; QAOA finds the new optimum 133× more often than random |

*Gap closed* measures how far QAOA's samples move from random guessing toward the best layout: 0 = no better than random, 1 = the best layout every time.

No quantum advantage is claimed: even at 24 sites the exact answer is computable in seconds, which is what lets us measure QAOA's quality honestly. Hardware: 21 jobs, 174 s of QPU in total.

## Run it

```bash
git clone https://github.com/mrw-soumik/Qiskit-AI-Entanglers-.git && cd Qiskit-AI-Entanglers-
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python test_wakewise.py                                                   # self-checks
jupyter nbconvert --to notebook --execute --inplace wakewise.ipynb        # ~30 min, no IBM account needed
python make_ai_training_set.py                                            # optional: rebuild the AI training set (~7 min, 10 cores)
```

To rerun a hardware experiment: `cp .env.example .env` (or create `api.json`), paste your IBM Quantum `apikey` and `crn`, then e.g.
`python wakewise.py hardware climate_12 3` (flags: `--readout` adds readout calibration, `--healthy` keeps faulty qubits out of the layout, `--ai` runs a terrain instance with the predicted and transferred angles, e.g. `python wakewise.py hardware terrain_24 2 --ai`) and re-execute the notebook. `.env` and `api.json` are gitignored.

## Files

| Path | What |
|---|---|
| `wakewise.ipynb` | Working notebook with saved outputs |
| `wakewise.py` | Wind data, terrain (wind atlas, elevation, site rule, PyWake `XRSite`), wind-farm model, QUBO, Ising mapping, baselines, QAOA (CVaR, fast exact simulator), readout mitigation, faulty-qubit-aware IBM hardware helpers |
| `make_ai_training_set.py` | Builds the AI training set (285 instances with trained QAOA angles) |
| `test_wakewise.py` | Self-checks (Ising energy equals QUBO for every bitstring, fast simulator equals Qiskit's statevector, bit order, wake physics, terrain rule and speed-ups, CVaR, readout mitigation, scoring) |
| `data/` | Frozen ECCC wind 2007-2025, wind-atlas and elevation windows, and their provenance (`data/SOURCE.md`) |
| `results/` | Noise-model and hardware runs (device, job ID, options, calibration snapshot, counts), the AI training set and angles, and `summary.json` (every number in the slides) |
| `figures/`, `slides/` | Figures and the presentation (PPTX and PDF) |

## Credits

QAOA workflow pattern adapted from Tinh Bui's *Quantum Optimization* workshop notebook (Qiskit Fall Fest 2026); IBM runtime setup from IBM's *Qiskit 101* notebook (Apache 2.0) and the Fall Fest IBM hardware example; Qiskit, Qiskit Aer, Qiskit IBM Runtime; DTU PyWake (MIT); scikit-learn; rasterio; ECCC hourly climate data (Open Government Licence, Canada); Global Wind Atlas (DTU with the World Bank Group/ESMAP, CC BY 4.0); Copernicus DEM GLO-90 (© DLR e.V. and © Airbus Defence and Space GmbH, provided under COPERNICUS by the European Union and ESA). Methods: Jensen/Park and Bastankhah-Porte-Agel wake models; QAOA (Farhi et al., 2014); CVaR objective (Barkoutsos et al., 2020); WAsP ruggedness index (Bowen and Mortensen, 1996); prior quantum wind-farm layout studies (Hancock et al., arXiv:2312.13123, 2023; Nigro et al., *Advanced Quantum Technologies*, 2025); QAOA parameter transfer (Brandão et al., 2018; Shaydulin et al., 2023; Sureshbabu et al., 2024). Full credits and our own contribution: Section 15 of the notebook.
