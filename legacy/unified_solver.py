#!/usr/bin/env python3
# unified_solver.py
"""
Unified solver that demonstrates the refactored codebase using the new data structure.
This script shows how to use the new data generator, QUBO builder, and solvers.
"""

import numpy as np
import pandas as pd
import json
import time
import argparse
import subprocess
import tempfile
import os
import sys
from data_generator import generate_ethiopia_dataset
from qubo_builder import build_qubo, analyze_solution, enforce_hard_budget, constraint_budget


def _evaluate_solution_metrics(x, df, Q, budget, max_grids, min_population, qubo_kwargs=None):
    """Compute objective diagnostics for a binary solution vector."""
    qubo_kwargs = qubo_kwargs or {}
    alpha = float(qubo_kwargs.get("alpha", 150))
    gamma = float(qubo_kwargs.get("gamma", 300))
    theta = float(qubo_kwargs.get("theta", 1e-6))
    mu = float(qubo_kwargs.get("mu", 2))
    lambda_ = float(qubo_kwargs.get("lambda_", 1e-4))
    auto_calibrate_theta = bool(qubo_kwargs.get("auto_calibrate_theta", True))

    x_arr = np.array(x, dtype=float)
    costs = df["Installation_Cost_USD"].values.astype(float)
    pops = df["Population_Coverage"].values.astype(float)
    energies = df["Energy_Capacity_kWh_day"].values.astype(float)

    total_cost = float(np.dot(x_arr, costs))
    total_population = float(np.dot(x_arr, pops))
    total_energy = float(np.dot(x_arr, energies))
    selected_sites = int(np.sum(x_arr))

    objective_linear = float(np.sum(costs * x_arr) - alpha * np.sum(pops * x_arr) - gamma * np.sum(energies * x_arr))
    budget_penalty = float(constraint_budget(
        x_arr,
        df,
        budget=budget,
        theta=theta,
        alpha=alpha,
        gamma=gamma,
        auto_calibrate_theta=auto_calibrate_theta,
    ))
    grid_penalty = float(mu * (np.sum(x_arr) - max_grids) ** 2)
    population_penalty = float(lambda_ * (min_population - np.sum(pops * x_arr)) ** 2)
    total_penalty = float(budget_penalty + grid_penalty + population_penalty)
    xqx = float(x_arr @ Q @ x_arr)

    budget_violation = max(0.0, total_cost - float(budget))
    grid_violation = max(0.0, selected_sites - int(max_grids))
    population_shortfall = max(0.0, float(min_population) - total_population)

    return {
        "xQx": xqx,
        "objective_linear": objective_linear,
        "budget_penalty": budget_penalty,
        "grid_penalty": grid_penalty,
        "population_penalty": population_penalty,
        "total_penalty": total_penalty,
        "total_cost": total_cost,
        "total_population": total_population,
        "total_energy": total_energy,
        "num_sites": selected_sites,
        "budget_violation": budget_violation,
        "grid_violation": float(grid_violation),
        "population_shortfall": population_shortfall,
        "is_hard_feasible": bool(
            budget_violation <= 1e-9 and grid_violation <= 0 and population_shortfall <= 1e-9
        ),
    }


def _default_results_path(mode_name):
    return os.path.join("results", f"unified_solver_results_{mode_name}.json")


def _serialize_for_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, dict):
        return {k: _serialize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_for_json(v) for v in value]
    return value

def nar_greedy_solver(df, budget, max_grids=10, record_progress=False):
    """
    NAR (Nearest Available Resource) greedy solver.
    Selects sites based on population-to-cost ratio within budget.
    If record_progress is True, saves progress to 'nar_progress.json'.
    """
    import json
    # Calculate population-to-cost ratio
    df_copy = df.copy()
    df_copy['pop_cost_ratio'] = df_copy['Population_Coverage'] / df_copy['Installation_Cost_USD']
    
    # Sort by ratio (descending)
    df_sorted = df_copy.sort_values('pop_cost_ratio', ascending=False)
    
    selected_indices = []
    total_cost = 0
    total_population = 0
    progress = []
    
    for idx, row in df_sorted.iterrows():
        if (total_cost + row['Installation_Cost_USD'] <= budget and 
            len(selected_indices) < max_grids):
            selected_indices.append(idx)
            total_cost += row['Installation_Cost_USD']
            total_population += row['Population_Coverage']
            # Record progress
            if record_progress:
                solution = [1 if i in selected_indices else 0 for i in range(len(df))]
                progress.append({
                    'step': len(selected_indices),
                    'solution': solution,
                    'total_cost': total_cost,
                    'total_population': total_population
                })
    # Save progress if requested
    if record_progress:
        with open('nar_progress.json', 'w') as f:
            json.dump(progress, f)
    # Create binary solution vector
    solution = np.zeros(len(df))
    for idx in selected_indices:
        solution[idx] = 1
    
    return solution, {
        'total_cost': total_cost,
        'total_population': total_population,
        'num_sites': len(selected_indices)
    }

def gurobi_solver(df, budget, max_grids=10, min_population=15000, qubo_kwargs=None):
    """
    Gurobi solver using the new data structure.
    """
    # Build QUBO
    Q, offset = build_qubo(df, budget, max_grids, min_population, **(qubo_kwargs or {}))
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as qubo_file:
        json.dump({'Q': Q.tolist()}, qubo_file)
        qubo_path = qubo_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as costs_file:
        json.dump(df['Installation_Cost_USD'].tolist(), costs_file)
        costs_path = costs_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as data_file:
        json.dump(df.to_dict('records'), data_file)
        data_path = data_file.name
    
    try:
        # Run Gurobi solver
        result = subprocess.run([
            'python3', 'gurobi_optimize.py',
            '--qubo_file', qubo_path,
            '--budget', str(budget),
            '--costs_file', costs_path,
            '--data_file', data_path
        ], capture_output=True, text=True, check=True)
        
        # Parse result
        output_lines = result.stdout.strip().split('\n')
        result_data = json.loads(output_lines[-1])
        
        # Create binary solution vector
        solution = np.zeros(len(df))
        for idx in result_data['selected_indices']:
            solution[idx] = 1
        
        return solution, result_data
        
    except subprocess.CalledProcessError as e:
        print(f"Gurobi solver error: {e}")
        print(f"stderr: {e.stderr}")
        return None, None
    finally:
        # Clean up temporary files
        for path in [qubo_path, costs_path, data_path]:
            try:
                os.unlink(path)
            except:
                pass

def sa_solver(df, budget, max_grids=10, min_population=15000, steps=10000, tmax=25000.0, tmin=0.001, record_progress=False, progress_interval=1, qubo_kwargs=None):
    """
    Simulated Annealing solver using the new data structure.
    """
    # Build QUBO
    Q, offset = build_qubo(df, budget, max_grids, min_population, **(qubo_kwargs or {}))
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as qubo_file:
        json.dump({'Q': Q.tolist()}, qubo_file)
        qubo_path = qubo_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as costs_file:
        json.dump(df['Installation_Cost_USD'].tolist(), costs_file)
        costs_path = costs_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as data_file:
        json.dump(df.to_dict('records'), data_file)
        data_path = data_file.name
    
    try:
        # Run SA solver
        cmd = [
            'python3', 'sa_optimize.py',
            '--qubo_file', qubo_path,
            '--budget', str(budget),
            '--costs_file', costs_path,
            '--data_file', data_path,
            '--steps', str(steps),
            '--tmax', str(tmax),
            '--tmin', str(tmin)
        ]
        if record_progress:
            cmd.append('--record_progress')
            cmd.extend(['--progress_interval', str(progress_interval)])
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse result
        output_lines = result.stdout.strip().split('\n')
        result_data = json.loads(output_lines[-1])
        
        # Create binary solution vector
        solution = np.zeros(len(df))
        for idx in result_data['selected_indices']:
            solution[idx] = 1
        
        return solution, result_data
        
    except subprocess.CalledProcessError as e:
        print(f"SA solver error: {e}")
        print(f"stderr: {e.stderr}")
        return None, None
    finally:
        # Clean up temporary files
        for path in [qubo_path, costs_path, data_path]:
            try:
                os.unlink(path)
            except:
                pass

def tabu_search_solver(df, budget, max_grids=10, min_population=15000, iterations=1000, tenure=10, qubo_kwargs=None):
    """
    Tabu Search solver using the new data structure.
    """
    # Build QUBO
    Q, offset = build_qubo(df, budget, max_grids, min_population, **(qubo_kwargs or {}))
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as qubo_file:
        json.dump({'Q': Q.tolist()}, qubo_file)
        qubo_path = qubo_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as data_file:
        json.dump(df.to_dict('records'), data_file)
        data_path = data_file.name
    
    try:
        # Run Tabu Search solver
        result = subprocess.run([
            'python3', 'tabu_search_optimize.py',
            '--qubo_file', qubo_path,
            '--data_file', data_path,
            '--iterations', str(iterations),
            '--tenure', str(tenure)
        ], capture_output=True, text=True, check=True)
        
        # Parse result
        output_lines = result.stdout.strip().split('\n')
        result_data = json.loads(output_lines[-1])
        
        # Create binary solution vector
        solution = np.zeros(len(df))
        for idx in result_data['selected_indices']:
            solution[idx] = 1
        
        return solution, result_data
        
    except subprocess.CalledProcessError as e:
        print(f"Tabu Search solver error: {e}")
        print(f"stderr: {e.stderr}")
        return None, None
    finally:
        # Clean up temporary files
        for path in [qubo_path, data_path]:
            try:
                os.unlink(path)
            except:
                pass

def quantum_solver(df, budget, max_grids=10, min_population=5000, reps=2, maxiter=300, ibm_backend=None, qubo_kwargs=None):
    """
    Quantum (QAOA) solver using the same QUBO as classical solvers.
    Calls quantum_optimize.py as a subprocess (requires the quantum venv).
    If ibm_backend is specified, runs on real IBM Quantum hardware.
    """
    # Build QUBO
    Q, offset = build_qubo(df, budget, max_grids, min_population, **(qubo_kwargs or {}))
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as qubo_file:
        json.dump({'Q': Q.tolist()}, qubo_file)
        qubo_path = qubo_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as data_file:
        json.dump(df.to_dict('records'), data_file)
        data_path = data_file.name
    
    try:
        # Run quantum solver -- uses the same Python that is running this script
        cmd = [
            sys.executable, 'quantum_optimize.py',
            '--qubo_file', qubo_path,
            '--data_file', data_path,
            '--reps', str(reps),
            '--maxiter', str(maxiter)
        ]
        if ibm_backend:
            cmd.extend(['--ibm_backend', ibm_backend])
        
        # Real hardware jobs can take minutes; use a generous timeout
        timeout = 600 if ibm_backend else 300
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=True, timeout=timeout)
        
        # Parse result
        output_lines = result.stdout.strip().split('\n')
        result_data = json.loads(output_lines[-1])
        
        # Create binary solution vector
        solution = np.zeros(len(df))
        for idx in result_data['selected_indices']:
            solution[idx] = 1
        
        return solution, result_data
        
    except subprocess.CalledProcessError as e:
        print(f"Quantum solver error: {e}")
        print(f"stderr: {e.stderr}")
        return None, None
    except subprocess.TimeoutExpired:
        print(f"Quantum solver timed out after {timeout}s")
        return None, None
    finally:
        # Clean up temporary files
        for path in [qubo_path, data_path]:
            try:
                os.unlink(path)
            except:
                pass

def main():
    parser = argparse.ArgumentParser(description='Unified microgrid optimization solver')
    parser.add_argument('--solver', choices=['nar', 'gurobi', 'sa', 'tabu', 'quantum', 'all'], 
                       default='all', help='Solver to use')
    parser.add_argument('--budget', type=float, default=900000, 
                       help='Budget constraint')
    parser.add_argument('--max_grids', type=int, default=10, 
                       help='Maximum number of grids')
    parser.add_argument('--min_population', type=int, default=15000, 
                       help='Minimum population coverage')
    parser.add_argument('--num_sites', type=int, default=50, 
                       help='Number of sites to generate')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--sa_steps', type=int, default=10000, 
                       help='Number of SA steps')
    parser.add_argument('--sa_tmax', type=float, default=25000.0, 
                       help='SA initial temperature')
    parser.add_argument('--sa_tmin', type=float, default=0.001, 
                       help='SA final temperature')
    parser.add_argument('--record_progress', action='store_true', help='Record progress for visualization')
    parser.add_argument('--progress_interval', type=int, default=1, help='Record progress every N steps (SA only)')
    parser.add_argument('--tabu_iterations', type=int, default=1000,
                       help='Tabu Search iterations')
    parser.add_argument('--tabu_tenure', type=int, default=10,
                       help='Tabu Search tenure')
    parser.add_argument('--qaoa_reps', type=int, default=2,
                       help='QAOA layers (quantum solver)')
    parser.add_argument('--qaoa_maxiter', type=int, default=300,
                       help='QAOA optimizer max iterations (quantum solver)')
    parser.add_argument('--ibm_backend', type=str, default=None,
                       help='IBM backend name for quantum solver (e.g. ibm_torino). '
                            'If omitted, quantum solver uses local StatevectorSampler.')
    parser.add_argument('--alpha', type=float, default=None,
                       help='Override QUBO population benefit weight alpha.')
    parser.add_argument('--gamma', type=float, default=None,
                       help='Override QUBO energy benefit weight gamma.')
    parser.add_argument('--theta', type=float, default=None,
                       help='Override QUBO budget-penalty weight theta.')
    parser.add_argument('--mu', type=float, default=None,
                       help='Override QUBO grid-count penalty weight mu.')
    parser.add_argument('--lambda_weight', type=float, default=None,
                       help='Override QUBO population-penalty weight lambda_.')
    parser.add_argument('--disable_auto_theta', action='store_true',
                       help='Disable adaptive theta calibration in QUBO builder.')
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--paper_mode', action='store_true',
                           help='Use fixed paper penalties with soft constraints (no hard projection).')
    mode_group.add_argument('--hard_budget_mode', action='store_true',
                           help='Apply hard budget projection after solver output for reported solution.')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Optional JSON output path for solver results and diagnostics.')
    
    args = parser.parse_args()
    
    mode_name = 'default_mode'
    qubo_kwargs = {}
    if args.paper_mode:
        mode_name = 'paper_mode'
        qubo_kwargs = {
            'theta': 1e-6,
            'mu': 2,
            'lambda_': 1e-4,
            'auto_calibrate_theta': False,
        }
    elif args.hard_budget_mode:
        mode_name = 'hard_constraints_mode'

    if args.alpha is not None:
        qubo_kwargs['alpha'] = args.alpha
    if args.gamma is not None:
        qubo_kwargs['gamma'] = args.gamma
    if args.theta is not None:
        qubo_kwargs['theta'] = args.theta
    if args.mu is not None:
        qubo_kwargs['mu'] = args.mu
    if args.lambda_weight is not None:
        qubo_kwargs['lambda_'] = args.lambda_weight
    if args.disable_auto_theta:
        qubo_kwargs['auto_calibrate_theta'] = False

    save_results_path = args.save_results or _default_results_path(mode_name)
    save_results_dir = os.path.dirname(save_results_path)
    if save_results_dir:
        os.makedirs(save_results_dir, exist_ok=True)

    # Generate dataset
    print("Generating Ethiopia dataset...")
    df = generate_ethiopia_dataset(args.num_sites, args.seed)
    print(f"Generated {len(df)} sites")
    print(f"Total potential cost: ${df['Installation_Cost_USD'].sum():,}")
    print(f"Total potential population: {df['Population_Coverage'].sum():,}")
    print(f"Total potential energy: {df['Energy_Capacity_kWh_day'].sum():.2f} kWh/day")
    print(f"Mode: {mode_name}")
    print()

    Q_reference, offset = build_qubo(
        df,
        args.budget,
        args.max_grids,
        args.min_population,
        **qubo_kwargs,
    )
    
    solvers = []
    if args.solver == 'all':
        solvers = ['nar', 'gurobi', 'sa', 'tabu', 'quantum']
    else:
        solvers = [args.solver]
    
    results = {}
    
    for solver_name in solvers:
        print(f"Running {solver_name.upper()} solver...")
        start_time = time.time()
        
        if solver_name == 'nar':
            solution, result = nar_greedy_solver(df, args.budget, args.max_grids, record_progress=True)
        elif solver_name == 'gurobi':
            solution, result = gurobi_solver(df, args.budget, args.max_grids, args.min_population, qubo_kwargs=qubo_kwargs)
        elif solver_name == 'sa':
            solution, result = sa_solver(df, args.budget, args.max_grids, args.min_population, 
                                       args.sa_steps, args.sa_tmax, args.sa_tmin, record_progress=args.record_progress, progress_interval=args.progress_interval, qubo_kwargs=qubo_kwargs)
        elif solver_name == 'tabu':
            solution, result = tabu_search_solver(df, args.budget, args.max_grids, args.min_population,
                                                  args.tabu_iterations, args.tabu_tenure, qubo_kwargs=qubo_kwargs)
        elif solver_name == 'quantum':
            solution, result = quantum_solver(df, args.budget, args.max_grids, args.min_population,
                                              args.qaoa_reps, args.qaoa_maxiter, args.ibm_backend, qubo_kwargs=qubo_kwargs)
        
        elapsed_time = time.time() - start_time
        
        if solution is not None:
            raw_solution = np.array(solution, dtype=float)
            projected_solution = np.array(raw_solution, dtype=float)
            if args.hard_budget_mode:
                projected_solution = enforce_hard_budget(raw_solution, df, args.budget)

            reported_solution = projected_solution if args.hard_budget_mode else raw_solution
            analysis = analyze_solution(reported_solution, df)
            raw_metrics = _evaluate_solution_metrics(
                raw_solution,
                df,
                Q_reference,
                args.budget,
                args.max_grids,
                args.min_population,
                qubo_kwargs=qubo_kwargs,
            )
            projected_metrics = _evaluate_solution_metrics(
                projected_solution,
                df,
                Q_reference,
                args.budget,
                args.max_grids,
                args.min_population,
                qubo_kwargs=qubo_kwargs,
            )

            results[solver_name] = {
                'solution': reported_solution,
                'raw_solution': raw_solution,
                'projected_solution': projected_solution,
                'result': result,
                'analysis': analysis,
                'diagnostics': {
                    'raw': raw_metrics,
                    'projected': projected_metrics,
                },
                'mode': mode_name,
                'time': elapsed_time
            }
            
            print(f"[OK] {solver_name.upper()} completed in {elapsed_time:.2f} seconds")
            print(f"   Selected sites: {analysis['num_sites']}")
            print(f"   Total cost: ${analysis['total_cost']:,}")
            print(f"   Total population: {analysis['total_population']:,}")
            print(f"   Total energy: {analysis['total_energy']:.2f} kWh/day")
            print(f"   xQx (raw): {raw_metrics['xQx']:.2f}")
            if args.hard_budget_mode:
                print(f"   xQx (projected): {projected_metrics['xQx']:.2f}")
        else:
            print(f"[FAIL] {solver_name.upper()} failed")
        
        print()
    
    # Summary
    if len(results) > 1:
        print("Summary:")
        print("-" * 80)
        print(f"{'Solver':<12} {'xQx':<14} {'Sites':<6} {'Cost':<12} {'Population':<12} {'Time':<8}")
        print("-" * 80)
        for solver_name, result in results.items():
            analysis = result['analysis']
            metrics_key = 'projected' if args.hard_budget_mode else 'raw'
            xqx_value = result['diagnostics'][metrics_key]['xQx']
            print(f"{solver_name.upper():<12} {xqx_value:<14.2f} {analysis['num_sites']:<6} "
                  f"${analysis['total_cost']:<11,} {analysis['total_population']:<12,} "
                  f"{result['time']:<8.2f}s")

    serializable_results = {
        'metadata': {
            'mode': mode_name,
            'budget': args.budget,
            'max_grids': args.max_grids,
            'min_population': args.min_population,
            'qubo_kwargs': qubo_kwargs,
            'qubo_offset': float(offset),
        },
        'results': _serialize_for_json(results),
    }
    with open(save_results_path, 'w', encoding='utf-8') as output_file:
        json.dump(serializable_results, output_file, indent=2)
    print(f"\nSaved diagnostics to {save_results_path}")
    
    return results

if __name__ == '__main__':
    main() 