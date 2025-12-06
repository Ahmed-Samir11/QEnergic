# Solver Comparison Results

**Date:** December 6, 2025  
**Dataset:** 50 validated Ethiopia off-grid solar sites  
**Budget:** $900,000  
**Hyperparameters:** α=150, γ=300, θ=1e-6, μ=2, λ=1e-4, min_population=5000

## Summary Table

| Solver | Sites | Total Cost | Budget Status | Population | Energy (kWh/day) | QUBO Energy |
|--------|-------|------------|---------------|------------|------------------|-------------|
| **Simulated Annealing** | 3 | $787,500 | ✅ Within | 6,100 | 3,250 | -1,493,822 |
| NAR Greedy | 5 | $1,053,500 | ❌ Over 17% | 7,770 | 4,190 | -1,295,361 |
| Tabu Search | 5 | $946,000 | ❌ Over 5% | 7,070 | 4,000 | -1,434,397 |

## Optimal Solution (SA)

**Selected Sites:** Metehara, Mieso, Moyale_North

| Site | Cost | Population | Energy | Region |
|------|------|------------|--------|--------|
| Metehara | $195,000 | 1,400 | 950 kWh | Oromia |
| Mieso | $312,500 | 2,500 | 1,200 kWh | Oromia |
| Moyale_North | $280,000 | 2,200 | 1,100 kWh | Oromia/Borena |

## Detailed Solver Results

### 1. NAR Greedy (Population/Cost Ratio)

**Algorithm:** Sorts sites by population-to-cost ratio and greedily selects within budget.

**Result:** 5 sites selected
- Sites: Mieso, Moyale_North, Negele_Borena, Adama_Rural, Telifa
- Total Cost: $1,053,500 (17% over budget)
- Total Population: 7,770
- Total Energy: 4,190 kWh/day

**Note:** NAR ignores the QUBO formulation entirely - it's a pure greedy heuristic.

### 2. Simulated Annealing (SA)

**Algorithm:** Stochastic optimization minimizing QUBO energy with temperature-based acceptance.

**Parameters:** steps=50000, Tmax=50000, Tmin=0.001

**Result:** 3 sites selected
- Sites: Metehara, Mieso, Moyale_North
- Total Cost: $787,500 ✅
- Total Population: 6,100
- Total Energy: 3,250 kWh/day
- QUBO Energy: -1,493,822 (best)

**Note:** Only solver that stays within budget while meeting population target.

### 3. Tabu Search

**Algorithm:** Local search with tabu list to avoid cycling.

**Parameters:** iterations=5000, tenure=20

**Result:** 5 sites selected
- Sites: Metehara, Chancho, Negele_Borena, Yabello, Moyale_North
- Total Cost: $946,000 (5% over budget)
- Total Population: 7,070
- Total Energy: 4,000 kWh/day
- QUBO Energy: -1,434,397

## Analysis

### Why SA Finds the Best Solution

1. **QUBO formulation correctly penalizes over-budget**: The budget penalty term θ*(cost-budget)² makes over-budget solutions less optimal in QUBO energy terms.

2. **SA explores globally**: Unlike greedy approaches, SA can escape local minima through stochastic acceptance of worse solutions.

3. **Hyperparameter tuning matters**: With balanced α and γ values (~cost/benefit ratio), the QUBO correctly trades off between benefits and constraints.

### Constraint Satisfaction

| Solver | Budget ≤ $900k | Population ≥ 5000 | Sites ≤ 10 |
|--------|----------------|-------------------|------------|
| SA | ✅ | ✅ | ✅ |
| NAR | ❌ | ✅ | ✅ |
| Tabu | ❌ | ✅ | ✅ |

## Recommendations

1. **For strict budget compliance:** Use Simulated Annealing
2. **For maximum population coverage:** Use NAR Greedy (but expect budget overrun)
3. **For quantum hardware:** The QUBO from `qubo_builder.py` is ready for D-Wave or gate-based quantum annealers
