# QEnergic

**Microgrid Placement Optimization for Ethiopia using QUBO Formulation**

A hybrid optimization system combining a Next.js web interface with Python-based solvers (classical, quantum-inspired, and quantum) to find optimal locations for off-grid solar microgrid installations in Ethiopia.

## 🎯 Project Overview

QEnergic addresses the challenge of optimal microgrid placement by:
- Balancing **budget constraints** with **population coverage** and **energy capacity**
- Using **QUBO (Quadratic Unconstrained Binary Optimization)** formulation compatible with quantum annealers
- Providing multiple solver options: classical heuristics, simulated annealing, and quantum (QAOA)

## 📊 Dataset

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
