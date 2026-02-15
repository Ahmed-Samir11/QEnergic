# QEnergic Solver Comparison Results

**Date:** December 6, 2025  
**Dataset:** Ethiopia Off-Grid Solar Sites (50 validated locations)  
**Budget Constraint:** $900,000  
**Max Grids:** 10  
**Min Population:** 15,000  

---

## Summary Comparison

| Solver | Sites | Total Cost | Population | Energy (kWh/day) | Time (s) | Budget Compliance |
|--------|-------|------------|------------|------------------|----------|-------------------|
| **NAR Greedy** | 5 | $890,500 | 6,800 | 3,490 | 0.00 | ✅ Under |
| **Simulated Annealing** | 2 | $645,000 | 4,300 | 1,900 | 0.11 | ✅ Under |
| **Tabu Search** | 3 | $673,000 | 5,100 | 2,900 | 0.16 | ✅ Under |
| **QUBO (dwave-neal)** | 4 | $1,015,500 | 7,850 | 4,070 | 0.88 | ⚠️ 12.8% over |

---

## Detailed Results by Solver

### 1. NAR Greedy Solver
**Strategy:** Selects sites by population-to-cost ratio (highest efficiency first)

| Site | Region | Cost ($) | Population | Energy (kWh/day) | Solar (kWh/m²/day) |
|------|--------|----------|------------|------------------|-------------------|
| Mieso | Oromia | 312,500 | 2,500 | 1,200 | 6.20 |
| Moyale_North | Oromia | 280,000 | 2,200 | 1,100 | 6.25 |
| Negele_Borena | Oromia | 198,000 | 1,500 | 850 | 6.05 |
| Chancho | Oromia | 48,000 | 320 | 180 | 5.70 |
| Aregawi | Oromia | 52,000 | 280 | 160 | 5.65 |

**Analysis:** NAR maximizes site count within budget by including low-cost Tier 1/2 sites (Chancho, Aregawi). All sites from Oromia cluster due to better logistics costs.

---

### 2. Simulated Annealing (simanneal)
**Strategy:** Stochastic optimization minimizing QUBO energy function

| Site | Region | Cost ($) | Population | Energy (kWh/day) | Solar (kWh/m²/day) |
|------|--------|----------|------------|------------------|-------------------|
| Yabello | Oromia | 225,000 | 1,650 | 920 | 6.00 |
| Sherkole | Benishangul-Gumuz | 420,000 | 2,650 | 980 | 5.40 |

**Analysis:** SA found a local minimum with only 2 sites. The Sherkole selection (Western Periphery) is interesting - it has lower solar but high population for refugee camp coverage.

---

### 3. Tabu Search
**Strategy:** Neighborhood search with memory to avoid cycling

| Site | Region | Cost ($) | Population | Energy (kWh/day) | Solar (kWh/m²/day) |
|------|--------|----------|------------|------------------|-------------------|
| Metehara | Oromia | 195,000 | 1,400 | 950 | 6.15 |
| Negele_Borena | Oromia | 198,000 | 1,500 | 850 | 6.05 |
| Moyale_North | Oromia | 280,000 | 2,200 | 1,100 | 6.25 |

**Analysis:** Tabu selected 3 high-solar Oromia sites. Good balance of cost efficiency and solar potential. All sites have GHI > 6.0 kWh/m²/day.

---

### 4. QUBO Solver (D-Wave Neal - Simulated Annealing Sampler)
**Strategy:** AutoQUBO symbolic formulation + D-Wave's production-grade SA sampler

| Site | Region | Cost ($) | Population | Energy (kWh/day) | Solar (kWh/m²/day) |
|------|--------|----------|------------|------------------|-------------------|
| Mieso | Oromia | 312,500 | 2,500 | 1,200 | 6.20 |
| Negele_Borena | Oromia | 198,000 | 1,500 | 850 | 6.05 |
| Yabello | Oromia | 225,000 | 1,650 | 920 | 6.00 |
| Moyale_North | Oromia | 280,000 | 2,200 | 1,100 | 6.25 |

**Analysis:** QUBO solver achieved highest population coverage (7,850) and energy capacity (4,070 kWh/day) but exceeded budget by ~$115k. The soft constraint formulation trades budget compliance for optimality. All 4 sites are in the southern Oromia corridor with excellent solar resources.

---

## Key Insights

### 1. Regional Clustering
All solvers preferentially selected sites from the **Oromia Agricultural Cluster**:
- Lower logistics costs (proximity to Addis-Djibouti corridor)
- High solar potential (5.65 - 6.25 kWh/m²/day)
- Good road access reduces installation CAPEX

### 2. Trade-offs Observed

| Solver | Strength | Weakness |
|--------|----------|----------|
| NAR | Budget compliant, fast | Lower per-site impact |
| SA | Fast, good exploration | Premature convergence (only 2 sites) |
| Tabu | Balanced solution | Fewer sites than possible |
| QUBO | Highest impact | Budget overrun |

### 3. Soft vs Hard Constraints
The QUBO formulation uses **soft constraints** (penalty terms) rather than hard budget limits. This allows the solver to find higher-quality solutions that slightly violate constraints. For strict budget compliance, the penalty weight `theta` should be increased from `1e-6` to `1e-4` or higher.

### 4. Site Overlap Analysis

Sites appearing in multiple solutions:
- **Moyale_North**: 3/4 solvers (NAR, Tabu, QUBO)
- **Negele_Borena**: 3/4 solvers (NAR, Tabu, QUBO)
- **Mieso**: 2/4 solvers (NAR, QUBO)
- **Yabello**: 2/4 solvers (SA, QUBO)

These sites represent **consensus high-value locations** for microgrid deployment.

---

## QUBO Hyperparameters Used

```python
alpha = 1e-1      # Population coverage weight
gamma = 1e-1      # Energy capacity weight  
theta = 1e-6      # Budget penalty weight
mu = 2            # Grid count penalty weight
lambda_ = 1e-2    # Minimum population penalty weight

budget = 900000
max_grids = 10
min_population = 15000
```

---

## Recommendations

1. **For strict budget compliance:** Use NAR or increase QUBO `theta` to `1e-4`
2. **For maximum impact:** Use QUBO solver with slight budget flexibility
3. **Consensus sites** (Moyale_North, Negele_Borena) should be prioritized in any deployment scenario
4. **Western Periphery sites** (Sherkole, Gambella region) only appear in SA results - may require explicit equity weighting to include in optimization
