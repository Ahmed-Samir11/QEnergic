# quantum_optimize.py
"""
Quantum QUBO solver using Qiskit QAOA via MinimumEigenOptimizer.
Supports both local simulation (StatevectorSampler) and real IBM Quantum hardware.

Usage (local sim, subprocess):
    python quantum_optimize.py --qubo_file QUBO.json [--data_file data.json] \\
        [--reps 2] [--maxiter 300]

Usage (real hardware, subprocess):
    python quantum_optimize.py --qubo_file QUBO.json [--data_file data.json] \\
        --ibm_backend ibm_torino [--reps 1] [--maxiter 200]

QUBO.json format:
    {"Q": [[Q00, Q01, ...], [Q10, Q11, ...], ...]}

Returns:
    Prints JSON: {"selected_indices": [...], "fval": ..., "time_sec": ...}
"""
import argparse
import json
import time
import numpy as np
import sys

# Core imports (always available)
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# IBM Quantum CRN for this project
from ibm_quantum_config import get_ibm_instance_crn, get_qiskit_ibm_token


def load_qubo(file_path):
    """Load QUBO matrix from JSON file. Supports {"Q": [[...]]} format."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'Q' in data:
        return np.array(data['Q'])
    return np.array(data)


def build_quadratic_program(Q):
    """
    Convert QUBO matrix to a qiskit QuadraticProgram.

    All constraints (budget, grid count, population) are already encoded
    in the QUBO matrix by qubo_builder.py, so no extra constraints are added.
    """
    n = Q.shape[0]
    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var(name=f'x{i}')

    linear = {f'x{i}': float(Q[i, i]) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            coeff = float(Q[i, j] + Q[j, i])  # symmetrise off-diagonal
            if coeff != 0:
                quadratic[(f'x{i}', f'x{j}')] = coeff

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp


def solve_qubo_qaoa_local(Q, reps=2, maxiter=300, seed=42):
    """
    Solve QUBO with QAOA using local StatevectorSampler.
    Practical for up to ~20 qubits.
    """
    qp = build_quadratic_program(Q)
    algorithm_globals.random_seed = seed

    sampler = StatevectorSampler(seed=seed)
    optimizer = COBYLA(maxiter=maxiter)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)

    min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
    result = min_eigen_optimizer.solve(qp)

    x = [int(result.x[i]) for i in range(len(result.x))]
    return x, result.fval


def solve_qubo_qaoa_hardware(Q, backend_name, reps=1, maxiter=200, seed=42):
    """
    Solve QUBO with QAOA on real IBM Quantum hardware.
    Handles any qubit count supported by the target backend.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    print(f"Connecting to IBM Quantum (backend={backend_name})...", file=sys.stderr)

    instance = get_ibm_instance_crn(required=True)
    token = get_qiskit_ibm_token()
    # If token is not provided, Qiskit may use stored credentials.
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        instance=instance,
        token=token,
    )
    backend = service.backend(backend_name)

    print(f"Backend {backend_name}: {backend.num_qubits} qubits", file=sys.stderr)

    qp = build_quadratic_program(Q)
    algorithm_globals.random_seed = seed

    # Do NOT pass transpiler to QAOA -- let SamplerV2 handle transpilation
    # internally so the circuit qubit count matches the observable.
    sampler = SamplerV2(mode=backend)
    optimizer = COBYLA(maxiter=maxiter)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)

    print(f"Running QAOA ({Q.shape[0]} qubits, {reps} layers, "
          f"maxiter={maxiter})...", file=sys.stderr)

    min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
    result = min_eigen_optimizer.solve(qp)

    x = [int(result.x[i]) for i in range(len(result.x))]
    return x, result.fval


def main():
    parser = argparse.ArgumentParser(
        description='Quantum QAOA QUBO solver (subprocess mode)'
    )
    parser.add_argument('--qubo_file', type=str, required=True,
                        help='Path to QUBO JSON file')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Path to data JSON for solution analysis')
    parser.add_argument('--reps', type=int, default=2,
                        help='Number of QAOA layers')
    parser.add_argument('--maxiter', type=int, default=300,
                        help='COBYLA max iterations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--ibm_backend', type=str, default=None,
                        help='IBM backend name (e.g. ibm_torino). '
                             'If omitted, uses local StatevectorSampler.')
    args = parser.parse_args()

    Q = load_qubo(args.qubo_file)

    start = time.time()
    if args.ibm_backend:
        try:
            # Validate configuration early so subprocess callers get a clear error.
            get_ibm_instance_crn(required=True)
        except Exception as e:
            print(str(e), file=sys.stderr)
            raise SystemExit(2)
        x, fval = solve_qubo_qaoa_hardware(
            Q, args.ibm_backend, reps=args.reps,
            maxiter=args.maxiter, seed=args.seed
        )
    else:
        x, fval = solve_qubo_qaoa_local(
            Q, reps=args.reps, maxiter=args.maxiter, seed=args.seed
        )
    elapsed = time.time() - start

    selected = [i for i, v in enumerate(x) if v]
    result = {
        'selected_indices': selected,
        'fval': float(fval),
        'time_sec': elapsed,
        'backend': args.ibm_backend or 'statevector_simulator'
    }

    # Add analysis if data file is provided
    if args.data_file:
        try:
            with open(args.data_file, 'r') as f:
                data = json.load(f)
            selected_data = [data[i] for i in selected]
            total_cost = sum(s["Installation_Cost_USD"] for s in selected_data)
            total_population = sum(s["Population_Coverage"] for s in selected_data)
            total_energy = sum(s["Energy_Capacity_kWh_day"] for s in selected_data)
            result.update({
                "total_cost": total_cost,
                "total_population": total_population,
                "total_energy": total_energy,
                "num_sites": len(selected_data)
            })
        except Exception as e:
            result["analysis_error"] = str(e)

    print(json.dumps(result))


if __name__ == '__main__':
    main()
