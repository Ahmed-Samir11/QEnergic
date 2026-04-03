# Q-Energic

**Quantum-Enhanced Microgrid Optimization for Rural Electrification**

This repository contains the full experimental codebase accompanying the paper:

> *QUBO Model for Energy Planning: Quantum-Enhanced Microgrid Optimization in Rural Electrification*

---

## Repository Structure

```
QEnergic/
├── generate_plots.py              # Main experiment script — runs all solvers, generates figures
├── qubo_builder.py                # QUBO matrix construction (core algorithm)
├── data_generator.py              # 50-site Ethiopia dataset generation
├── ibm_quantum_config.py          # IBM Quantum credentials helper
├── requirements.txt               # Python dependencies
├── data/
│   └── ETHIOPIA_SITES_DOCUMENTATION.md   # Provenance for all 50 sites
├── figures/                       # Generated figures (reproducible via generate_plots.py)
├── legacy/                        # Superseded scripts retained for reference
├── components/
│   └── MapOptimizer.jsx           # Interactive web map (companion demo)
└── pages/                         # Next.js web interface (companion demo)
```

---

## Dataset

50 validated candidate sites for off-grid solar microgrids across Ethiopia, derived from:

- **NEP 2.0** — Ethiopia National Electrification Program 2.0 priority sites
- **DREAM Project** — World Bank/ESMAP Agricultural Mini-grid Pilots (East Shewa, Akaki)
- **EEU/AfDB 25 Mini-Grids Tender** — operational system specifications
- **UNHCR/IOM** — refugee settlement population data (Dollo Ado, Pugnido, Sherkole)
- **NASA POWER** — satellite-derived global horizontal irradiance (GHI)

Sites span four regional clusters:

| Cluster | Sites | Key Characteristics |
|---|---|---|
| Oromia Agricultural | 1–15 | High energy capacity, proximity to logistics corridor |
| SNNP & South West | 16–25 | Mountain terrain, high installation cost |
| Somali & Afar Lowland | 26–39 | Maximum solar potential (GHI > 6.2 kWh/m²/day) |
| Western Periphery | 40–50 | Lowest solar potential, security and accessibility challenges |

Full provenance per site is documented in [`data/ETHIOPIA_SITES_DOCUMENTATION.md`](data/ETHIOPIA_SITES_DOCUMENTATION.md).

---

## QUBO Formulation

The optimization is encoded as min x'Qx where x ∈ {0,1}^n selects microgrid sites:

```
Q(x) = Σ C_i x_i  −  α Σ P_i x_i  −  γ Σ E_i x_i
       + θ (Σ C_i x_i − B)²
       + μ (Σ x_i − K)²
       + λ (M − Σ P_i x_i)²
```

| Symbol | Value | Description |
|---|---|---|
| α | 150 | Population coverage benefit weight |
| γ | 300 | Energy capacity benefit weight |
| θ | auto-calibrated | Budget constraint penalty |
| μ | 2 | Grid count penalty |
| λ | 1e-4 | Minimum population penalty |
| B | $900,000 | Maximum total installation budget |
| K | 10 | Maximum number of microgrid sites |
| M | 5,000 | Minimum target population coverage |

---

## Solvers

| Solver | Type | Notes |
|---|---|---|
| NAR Greedy | Classical domain heuristic | Population/cost ratio greedy; hard budget constraint |
| Simulated Annealing | Classical metaheuristic | `simanneal`, 100 restarts, auto-calibrated temperature |
| Tabu Search | Classical metaheuristic | D-Wave `TabuSampler`, 100 reads |
| D-Wave QBSolv | Hybrid decomposition | `dwave-hybrid` decomposition + Tabu subproblem solver |
| QAOA (hardware) | Gate-based quantum | IBM Quantum via Qiskit Runtime, `reps=1`, COBYLA optimizer, 5 iterations |

---

## Reproducing Results

### Requirements

- Python 3.11 (use the included `quantum/` virtual environment)
- IBM Quantum account with instance CRN and API token (for hardware QAOA)

### Setup

```powershell
# Activate the virtual environment (Windows)
quantum\Scripts\Activate.ps1

# Install/update dependencies
# Use python -m pip — the pip entry-point may have a stale path in some environments
python -m pip install -r requirements.txt
```

### Environment Variables

Create `.env.local` in the repo root with your IBM Quantum credentials:

```
IBM_QUANTUM_INSTANCE_CRN=crn:v1:bluemix:...
QISKIT_IBM_TOKEN=your_token_here
```

`ibm_quantum_config.py` loads `.env.local` automatically — no manual `export` needed.

### Running Experiments

```bash
# All local solvers — generates all 15 figures
python generate_plots.py --save_result figures

# Include IBM Quantum hardware QAOA (~10 min of quantum time)
python generate_plots.py --ibm_backend ibm_torino --save_result figures

# Regenerate figures from a saved run (no solver re-execution)
python generate_plots.py --load figures/solver_results.json
```

All figures are written to `figures/` as PNG (150 dpi) and PDF (300 dpi).

### Reproduce D-Wave QBSolv Result

This repository includes a validated 100-iteration D-Wave profile that reproduces:

- `num_sites = 6`
- `total_cost = 908000`
- `total_population = 6230`
- `total_energy = 3750`

Run:

```powershell
python generate_plots.py --from_csv figures/solver_results.csv --rerun "D-Wave QBSolv" --record_progress --dwave_max_iter 100 --dwave_subproblem_size 16 --dwave_num_reads 20 --dwave_timeout_ms 500 --save_results figures/solver_results.json
```

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@inproceedings{farhani2026qenergic,
  title     = {{QUBO} Model for Energy Planning: Quantum-Enhanced Microgrid
               Optimization in Rural Electrification},
  author    = {Farhani, Yousra and {Medie Fah}, Helarie Rose and Samir, Ahmed
               and Adam, Marzuq Yussif Etsie and Mulila, Kenedy Mwendwa
               and Osumanu, Abdulmajid},
  booktitle = {Genetic and Evolutionary Computation Conference (GECCO '26 Companion)},
  year      = {2026},
  address   = {San Jos\'{e}, Costa Rica},
  publisher = {ACM}
}
```

---

## License

MIT License.

50 validated off-grid solar sites in Ethiopia sourced from:
- **NEP 2.0** (National Electrification Program)
- **DREAM Agricultural Mini-grid Pilots**
- **IOM/UNHCR Settlement Data**
- **AfDB Feasibility Studies**