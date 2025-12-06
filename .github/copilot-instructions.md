# QEnergic - Copilot Instructions

## Project Overview

QEnergic is a **microgrid placement optimization** system that combines a Next.js web interface with Python-based optimization solvers. The goal is to find optimal locations for microgrid installations in Ethiopia, balancing budget constraints, population coverage, and energy capacity using QUBO (Quadratic Unconstrained Binary Optimization) formulation.

## Architecture

```
Frontend (Next.js)          →  API Layer  →  Python Solvers
MapOptimizer.jsx               optimize.js     unified_solver.py
(Leaflet map, user input)      (QUBO build)    (solver orchestration)
                                    ↓
                               Solver Scripts
                               ├── sa_optimize.py      (Simulated Annealing)
                               ├── gurobi_optimize.py  (Gurobi ILP)
                               ├── quantum_optimize.py (QAOA via Qiskit)
                               └── tabu_search_optimize.py
```

### Data Flow
1. User draws 4-point polygon + population center on map
2. `optimize.js` filters Ethiopia sites within polygon, builds QUBO matrix
3. Selected solver minimizes `x'Qx` where `x` is binary selection vector
4. Results returned as grid locations with cost/energy/population

### QUBO Formulation (Critical Knowledge)
The QUBO matrix in `qubo_builder.py` encodes soft constraints as penalties:
- **Objective**: `Σ cost[i]*x[i] - alpha*Σ pop[i]*x[i] - gamma*Σ energy[i]*x[i]`
- **Budget penalty**: `theta * (Σ costs - budget)²` (theta=1e-6)
- **Grid count penalty**: `mu * (Σ x - max_grids)²` (mu=2)
- **Population penalty**: `lambda * (min_pop - Σ population)²` (lambda=1e-4)

**Tuned Hyperparameters** (calibrated to data scale):
- `alpha = 150` (population benefit, ~avg_cost/avg_pop ratio)
- `gamma = 300` (energy benefit, ~avg_cost/avg_energy ratio)
- `theta = 1e-6` (budget penalty)
- `mu = 2` (grid count penalty)
- `lambda_ = 1e-4` (population target penalty)
- `min_population = 5000` (achievable within $900k budget)
- `budget = 900000`, `max_grids = 10`

## Key Conventions

### Python Solvers
- All solvers accept `--qubo_file` (JSON with `{"Q": [[...]]}`) and output JSON to stdout
- Results format: `{"selected_indices": [...], "fval": ..., "time_sec": ...}`
- `unified_solver.py` orchestrates solvers via subprocess, handles temp file I/O
- Site data uses DataFrame with columns: `Site_ID`, `Installation_Cost_USD`, `Population_Coverage`, `Solar_Potential_kWh_m2_day`, `Energy_Capacity_kWh_day`, `X_coord`, `Y_coord`

### JavaScript/Frontend
- `data_generator.js` exports `ethiopiaSites` (50 sites with Ethiopia coords)
- Map uses react-leaflet with Turf.js for geospatial filtering
- API expects: `{polygon: [{lat, lng}×4], budget, popCenter: {lat, lng}, algo: 'nar'|'classical'|'gurobi'}`

## Developer Commands

```bash
# Frontend
npm run dev          # Start Next.js dev server

# Python solvers (standalone)
python unified_solver.py --solver all --budget 900000
python unified_solver.py --solver sa --record_progress  # Saves sa_progress.json

# Individual solvers
python sa_optimize.py --qubo_file QUBO.json --budget 10000 --costs_file costs.json
python gurobi_optimize.py --qubo_file QUBO.json --budget 10000 --costs_file costs.json
```

## Required Dependencies

**Python**: numpy, pandas, simanneal, gurobipy (optional), qiskit/qiskit-optimization (for quantum)
**Node**: See `package.json` - key deps are `@turf/turf`, `react-leaflet`, `leaflet`

## Common Patterns

### Adding a New Solver
1. Create `new_solver.py` accepting `--qubo_file`, `--data_file` args
2. Output JSON result to stdout (last line parsed)
3. Add wrapper function in `unified_solver.py` following `sa_solver()` pattern
4. Optionally expose via `optimize.js` API

### Modifying QUBO Weights
Edit `qubo_builder.py` `build_qubo()` - weights are: `alpha` (population), `gamma` (energy), `theta` (budget penalty), `mu` (grid count), `lambda_` (min population)

### Progress Visualization
Use `--record_progress` flag → generates `*_progress.json` → run `visualize_*_progress.py`
