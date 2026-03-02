# Q-Energic

**Quantum-Enhanced Microgrid Optimization for Rural Electrification**

This repository contains the full experimental codebase accompanying the paper:

> *QUBO Model for Energy Planning: Quantum-Enhanced Microgrid Optimization in Rural Electrification*  
> Farhani et al., GECCO '26 Companion, July 13–17, 2026, San José, Costa Rica.

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
| D-Wave Neal | Quantum-inspired SA | `dwave-neal`, 100 reads × 5,000 sweeps |
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

Sites are organized into 4 regional clusters:
| Cluster | Sites | Characteristics |
|---------|-------|-----------------|
| Oromia Agricultural | 15 | High energy capacity (irrigation), moderate cost |
| SNNP & South West | 10 | Mountainous logistics, high installation cost |
| Somali & Afar Lowland | 14 | Maximum solar potential (>6.2 kWh/m²/day) |
| Western Periphery | 11 | Lower solar potential, security/logistics challenges |

## 🔬 QUBO Formulation

The optimization minimizes:

```
f(x) = Σ cost[i]·x[i] - α·Σ pop[i]·x[i] - γ·Σ energy[i]·x[i]
     + θ·(Σ cost[i]·x[i] - budget)²
     + μ·(Σ x[i] - max_grids)²
     + λ·(min_pop - Σ pop[i]·x[i])²
```

**Tuned Hyperparameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| α (alpha) | 150 | Population benefit weight |
| γ (gamma) | 300 | Energy capacity benefit weight |
| θ (theta) | 1e-6 | Budget constraint penalty |
| μ (mu) | 2 | Grid count penalty |
| λ (lambda) | 1e-4 | Minimum population penalty |
| budget | $900,000 | Maximum total installation cost |
| max_grids | 10 | Maximum number of sites |
| min_population | 5,000 | Target population coverage |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ with virtual environment
- Node.js 16+
- (Optional) Gurobi license for ILP solver
- (Optional) D-Wave account for quantum annealing

### Installation

```bash
# Create Python virtual environment
python -m venv quantum
quantum\Scripts\activate  # Windows
# source quantum/bin/activate  # Linux/Mac

# Install Python dependencies
pip install -r requirements.txt

# Install Node dependencies
npm install
```

### Environment Variables

This repo keeps secrets/instance identifiers out of source control.

1. Copy the example file:

```bash
copy .env.example .env.local  # Windows
# cp .env.example .env.local  # Linux/Mac
```

2. Fill in values in `.env.local` (it is git-ignored).

- `NEXT_PUBLIC_MAPBOX_TOKEN` (public)
- `IBM_QUANTUM_INSTANCE_CRN` (server-side only; required for hardware runs)
- `QISKIT_IBM_TOKEN` (server-side only; optional)

The Next.js quantum API route only returns detailed subprocess logs in development.

### Running Solvers

```bash
# Classical solvers via unified_solver.py
python unified_solver.py --solver nar --budget 900000
python unified_solver.py --solver sa --budget 900000

# Individual solvers
python sa_optimize.py --qubo_file QUBO.json --budget 900000 --costs_file costs.json
python tabu_search_optimize.py --qubo_file QUBO.json --data_file data.json

# Quantum solver (requires autoqubo + dwave-qbsolv)
python qubo.py
```

### Web Interface

```bash
npm run dev
# Open http://localhost:3000
```

## 📈 Solver Comparison Results

Results with tuned hyperparameters on 50 Ethiopia sites:

| Solver | Sites Selected | Total Cost | Budget Status | Population | Energy (kWh/day) |
|--------|----------------|------------|---------------|------------|------------------|
| **Simulated Annealing** | 3 | $787,500 | ✅ Within | 6,100 | 3,250 |
| NAR Greedy | 5 | $1,053,500 | ❌ Over 17% | 7,770 | 4,190 |
| Tabu Search | 5 | $946,000 | ❌ Over 5% | 7,070 | 4,000 |

**Best QUBO Solution Sites:** Metehara, Mieso, Moyale_North

## 🏗️ Architecture

```
Frontend (Next.js)          →  API Layer  →  Python Solvers
MapOptimizer.jsx               optimize.js     unified_solver.py
(Leaflet map, user input)      (QUBO build)    (solver orchestration)
                                    ↓
                               Solver Scripts
                               ├── sa_optimize.py      (Simulated Annealing)
                               ├── gurobi_optimize.py  (Gurobi ILP)
                               ├── quantum_optimize.py (QAOA via Qiskit)
                               ├── tabu_search_optimize.py
                               └── qubo.py             (autoqubo + D-Wave)
```

## 📁 Project Structure

```
QEnergic/
├── components/
│   └── MapOptimizer.jsx     # React map component
├── pages/
│   ├── api/
│   │   └── optimize.js      # API endpoint with QUBO builder
│   └── index.js             # Main page
├── data/
│   ├── ETHIOPIA_SITES_DOCUMENTATION.md
│   └── africa_cities.json
├── qubo_builder.py          # Core QUBO matrix construction
├── unified_solver.py        # Solver orchestration
├── sa_optimize.py           # Simulated Annealing
├── tabu_search_optimize.py  # Tabu Search
├── gurobi_optimize.py       # Gurobi ILP
├── quantum_optimize.py      # QAOA (Qiskit)
├── qubo.py                  # autoqubo symbolic QUBO
├── data_generator.py        # Python dataset (50 sites)
├── data_generator.js        # JavaScript dataset
└── requirements.txt
```

## 📚 Documentation

- [Data Documentation](data/ETHIOPIA_SITES_DOCUMENTATION.md) - Detailed provenance for all 50 sites
- [Copilot Instructions](.github/copilot-instructions.md) - AI coding agent guidelines

## 🔧 Development

### Adding a New Solver

1. Create `new_solver.py` accepting `--qubo_file`, `--data_file` args
2. Output JSON result to stdout: `{"selected_indices": [...], "fval": ..., "time_sec": ...}`
3. Add wrapper function in `unified_solver.py`
4. Optionally expose via `optimize.js` API

### Modifying QUBO Weights

Edit `qubo_builder.py` `build_qubo()` function parameters.

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please read the contribution guidelines first.
