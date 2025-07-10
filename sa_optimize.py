# sa_optimize.py
"""
Simulated annealing QUBO solver with budget constraint using the 'simanneal' library.
Usage:
    python sa_optimize.py --qubo_file QUBO.json --budget 10000 --costs_file costs.json
QUBO.json format: {"Q": [[...]]}
costs.json format: [cost_0, cost_1, ...] (cost for each candidate)
Prints: {"selected_indices": [...], "fval": ..., "time_sec": ...}
"""
import argparse
import json
import numpy as np
import time
from simanneal import Annealer

class QUBOBudgetAnnealer(Annealer):
    def __init__(self, Q, costs, budget):
        self.Q = Q
        self.costs = costs
        self.budget = budget
        n = len(Q)
        # Start with no grids selected
        state = [0] * n
        super(QUBOBudgetAnnealer, self).__init__(state)

    def move(self):
        # Flip a random bit
        n = len(self.state)
        i = self.random.randint(0, n - 1)
        self.state[i] = 1 - self.state[i]

    def energy(self):
        x = np.array(self.state)
        e = float(x @ self.Q @ x)
        total_cost = np.dot(x, self.costs)
        if total_cost > self.budget:
            e += 10000 * (total_cost - self.budget) ** 2 / self.budget
        return e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qubo_file', type=str, required=True)
    parser.add_argument('--budget', type=float, required=True)
    parser.add_argument('--costs_file', type=str, required=True)
    parser.add_argument('--steps', type=int, default=10000)
    args = parser.parse_args()
    with open(args.qubo_file, 'r') as f:
        Q = np.array(json.load(f)["Q"])
    with open(args.costs_file, 'r') as f:
        costs = np.array(json.load(f))
    t0 = time.time()
    annealer = QUBOBudgetAnnealer(Q, costs, args.budget)
    annealer.steps = args.steps
    state, e = annealer.anneal()
    elapsed = time.time() - t0
    selected = [i for i, v in enumerate(state) if v]
    print(json.dumps({"selected_indices": selected, "fval": e, "time_sec": elapsed}))

if __name__ == '__main__':
    main()
