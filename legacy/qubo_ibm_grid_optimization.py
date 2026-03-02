#!/usr/bin/env python3
# qubo_ibm_grid_optimization.py
"""
Quantum (QAOA) solver for microgrid optimization.

Uses the same validated Ethiopia dataset and QUBO builder as the classical solvers,
solved via qiskit-algorithms QAOA + qiskit-optimization MinimumEigenOptimizer.
Supports both local simulation (StatevectorSampler) and real IBM Quantum hardware.

Usage (standalone, local sim):
    python qubo_ibm_grid_optimization.py [--num_sites 12] [--reps 2] [--maxiter 300]

Usage (standalone, real hardware):
    python qubo_ibm_grid_optimization.py --ibm_backend ibm_torino \\
        [--num_sites 50] [--reps 1] [--maxiter 200]

Usage (as subprocess, called by unified_solver.py):
    python qubo_ibm_grid_optimization.py --qubo_file QUBO.json --data_file data.json \\
        [--ibm_backend ibm_torino] [--reps 2] [--maxiter 300]
"""

import argparse
import json
import time
import numpy as np
import pandas as pd

import sys

# Qiskit imports (verified working with qiskit 2.2.3, qiskit-algorithms 0.4.0,
# qiskit-optimization 0.7.0, qiskit-ibm-runtime 0.45.1)
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from ibm_quantum_config import get_ibm_instance_crn, get_qiskit_ibm_token


def build_quadratic_program_from_matrix(Q):
    """
    Convert an n x n QUBO numpy matrix into a qiskit QuadraticProgram.

    Args:
        Q (np.ndarray): QUBO matrix (n x n)

    Returns:
        QuadraticProgram: ready for MinimumEigenOptimizer
    """
    n = Q.shape[0]
    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var(name=f"x{i}")

    linear = {f"x{i}": float(Q[i, i]) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            coeff = float(Q[i, j] + Q[j, i])  # symmetrise off-diagonal
            if coeff != 0:
                quadratic[(f"x{i}", f"x{j}")] = coeff

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp


def solve_qaoa_local(Q, reps=2, maxiter=300, seed=42):
    """
    Solve a QUBO matrix with QAOA via local StatevectorSampler.
    Practical for up to ~20 qubits.

    Returns:
        tuple: (solution_vector, objective_value, raw_result)
    """
    algorithm_globals.random_seed = seed
    qp = build_quadratic_program_from_matrix(Q)

    sampler = StatevectorSampler(seed=seed)
    optimizer = COBYLA(maxiter=maxiter)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)

    min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
    result = min_eigen_optimizer.solve(qp)

    x = np.array([int(result.x[i]) for i in range(len(result.x))])
    return x, result.fval, result


def solve_qaoa_hardware(Q, backend_name, reps=1, maxiter=200, seed=42):
    """
    Solve a QUBO matrix with QAOA on real IBM Quantum hardware.
    Handles any qubit count supported by the target backend.

    Returns:
        tuple: (solution_vector, objective_value, raw_result)
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    print(f"Connecting to IBM Quantum (backend={backend_name})...", file=sys.stderr)
    instance = get_ibm_instance_crn(required=True)
    token = get_qiskit_ibm_token()
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        instance=instance,
        token=token,
    )
    backend = service.backend(backend_name)
    print(f"Backend {backend_name}: {backend.num_qubits} qubits", file=sys.stderr)

    algorithm_globals.random_seed = seed
    qp = build_quadratic_program_from_matrix(Q)

    # Do NOT pass transpiler to QAOA -- let SamplerV2 handle transpilation
    # internally so the circuit qubit count matches the observable.
    sampler = SamplerV2(mode=backend)
    optimizer = COBYLA(maxiter=maxiter)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)

    print(f"Running QAOA ({Q.shape[0]} qubits, {reps} layers, "
          f"maxiter={maxiter})...")

    min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
    result = min_eigen_optimizer.solve(qp)

    x = np.array([int(result.x[i]) for i in range(len(result.x))])
    return x, result.fval, result


def main():
    parser = argparse.ArgumentParser(
        description="Quantum (QAOA) microgrid optimizer"
    )
    parser.add_argument(
        "--num_sites", type=int, default=12,
        help="Number of sites (default 12; keep <=20 for local sim, "
             "up to 50 for real hardware)"
    )
    parser.add_argument(
        "--budget", type=float, default=900000, help="Budget constraint"
    )
    parser.add_argument(
        "--max_grids", type=int, default=10, help="Max number of grids"
    )
    parser.add_argument(
        "--min_population", type=int, default=5000,
        help="Minimum population coverage"
    )
    parser.add_argument("--reps", type=int, default=2, help="QAOA layers")
    parser.add_argument(
        "--maxiter", type=int, default=300, help="COBYLA max iterations"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--ibm_backend", type=str, default=None,
        help="IBM backend name (e.g. ibm_torino). "
             "If omitted, uses local StatevectorSampler."
    )
    # Subprocess mode: receive QUBO + data via files (same interface as gurobi/SA)
    parser.add_argument("--qubo_file", type=str, default=None,
                        help="Path to QUBO JSON (subprocess mode)")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Path to data JSON (subprocess mode)")

    args = parser.parse_args()

    t0 = time.time()

    # ---- Load or build QUBO and data ----
    if args.qubo_file and args.data_file:
        # Subprocess mode: QUBO and data provided by unified_solver.py
        with open(args.qubo_file, "r") as f:
            Q = np.array(json.load(f)["Q"])
        with open(args.data_file, "r") as f:
            data_records = json.load(f)
        df = pd.DataFrame(data_records)
    else:
        # Standalone mode: generate data + build QUBO ourselves
        from data_generator import generate_ethiopia_dataset
        from qubo_builder import build_qubo

        num = args.num_sites
        if args.ibm_backend:
            num = max(num, 50)  # default to full 50 on real hardware
        print(f"Generating Ethiopia dataset ({num} sites)...")
        df = generate_ethiopia_dataset(num)
        Q, offset = build_qubo(
            df, args.budget, args.max_grids, args.min_population
        )
        print(f"QUBO matrix shape: {Q.shape}")
        print(f"QUBO offset: {offset:.4f}")

    # ---- Solve with QAOA ----
    n = Q.shape[0]
    backend_label = args.ibm_backend or "StatevectorSampler (local)"
    print(f"Running QAOA with {n} qubits, {args.reps} layers, "
          f"maxiter={args.maxiter}, backend={backend_label}...")

    if args.ibm_backend:
        try:
            get_ibm_instance_crn(required=True)
        except Exception as e:
            print(str(e), file=sys.stderr)
            raise SystemExit(2)
        x, fval, raw_result = solve_qaoa_hardware(
            Q, args.ibm_backend, reps=args.reps,
            maxiter=args.maxiter, seed=args.seed
        )
    else:
        x, fval, raw_result = solve_qaoa_local(
            Q, reps=args.reps, maxiter=args.maxiter, seed=args.seed
        )

    elapsed = time.time() - t0

    # ---- Results ----
    selected = [i for i, v in enumerate(x) if v == 1]

    result_dict = {
        "selected_indices": selected,
        "fval": float(fval),
        "time_sec": elapsed,
        "backend": args.ibm_backend or "statevector_simulator",
    }

    # Add analysis from the data
    if len(selected) > 0:
        selected_data = [df.iloc[i].to_dict() for i in selected]
        total_cost = sum(s["Installation_Cost_USD"] for s in selected_data)
        total_population = sum(s["Population_Coverage"] for s in selected_data)
        total_energy = sum(s["Energy_Capacity_kWh_day"] for s in selected_data)
        result_dict.update({
            "total_cost": total_cost,
            "total_population": total_population,
            "total_energy": total_energy,
            "num_sites": len(selected),
        })

    # In subprocess mode: print JSON for unified_solver.py to parse
    if args.qubo_file:
        print(json.dumps(result_dict))
    else:
        # Standalone mode: pretty-print results
        print(f"\n{'='*60}")
        print(f"QAOA Result  ({elapsed:.2f}s, backend={backend_label})")
        print(f"{'='*60}")
        print(f"Selected sites ({len(selected)}):")
        for idx in selected:
            row = df.iloc[idx]
            print(f"  {idx:>2}: {row['Site_ID']:<22} "
                  f"cost=${row['Installation_Cost_USD']:>8,}  "
                  f"pop={row['Population_Coverage']:>5}  "
                  f"energy={row['Energy_Capacity_kWh_day']:>7.1f}")
        if "total_cost" in result_dict:
            print(f"\nTotal cost:       ${result_dict['total_cost']:,}")
            print(f"Total population: {result_dict['total_population']:,}")
            print(f"Total energy:     {result_dict['total_energy']:.2f} kWh/day")
        print(f"QUBO objective:   {fval:.4f}")
        # Also dump JSON for easy comparison
        print(f"\n{json.dumps(result_dict)}")


if __name__ == "__main__":
    main()
