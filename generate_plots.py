#!/usr/bin/env python3
# generate_plots.py
"""
Microgrid Optimization Solver Comparison & Visualization Tool
==============================================================

This script runs multiple optimization solvers (classical, quantum-inspired, and quantum hardware)
on the 50-site Ethiopia microgrid placement problem and generates publication-quality figures
for the Q-Energic research paper.

SOLVERS AVAILABLE
-----------------
  • NAR Greedy          : Greedy heuristic (population/cost ratio)
  • Simulated Annealing : Classical metaheuristic (100 restarts)
  • Tabu Search         : Classical metaheuristic (100 reads)
    • D-Wave QBSolv       : QBSolv-based solver (dwave-qbsolv)
  • QAOA (IBM Torino)   : Gate-based quantum optimization on IBM Quantum hardware

COMMAND-LINE ARGUMENTS
----------------------
  --budget <int>              : Maximum installation budget (default: 900,000 USD)
  --max_grids <int>           : Maximum number of microgrids (default: 10)
  --min_population <int>      : Minimum population coverage target (default: 5,000)
  
  --ibm_backend <name>        : Run QAOA on IBM Quantum hardware (e.g., 'ibm_torino')
  --qaoa_local_sites <int>    : Number of sites for local QAOA simulation (default: 20)
  --no_qaoa_local             : Skip local QAOA simulation entirely
  
  --load <path.json>          : Load saved results from JSON file (legacy format)
  --from_csv <path.csv>       : Load saved results from CSV file (recommended)
  --save_results <path>       : Save results to JSON + CSV (auto-generates both formats)
  
  --solvers "Name1,Name2"     : Filter which solvers appear in plots (comma-separated)
  --rerun "Solver Name"       : Re-run specific solver(s) while loading others from CSV
  
  --record_progress           : Save per-iteration progress (for convergence plots)

USAGE EXAMPLES
--------------
  1. Run all solvers locally (no quantum hardware):
     $ python generate_plots.py

  2. Run with quantum hardware (requires IBM Quantum credentials in .env.local):
     $ python generate_plots.py --ibm_backend ibm_torino

  3. Load cached results from CSV and regenerate all figures instantly:
     $ python generate_plots.py --from_csv figures/solver_results.csv

  4. Re-run only D-Wave QBSolv while loading other solvers from cache:
      $ python generate_plots.py --from_csv figures/solver_results.csv --rerun "D-Wave QBSolv"

  5. Generate plots for only selected solvers:
     $ python generate_plots.py --from_csv figures/solver_results.csv --solvers "NAR Greedy,Sim. Annealing,QAOA (IBM Torino)"

  6. Save results to custom location:
     $ python generate_plots.py --save_results my_experiments/run_01

  7. Skip local QAOA simulation (faster, only hardware QAOA if specified):
     $ python generate_plots.py --ibm_backend ibm_torino --no_qaoa_local

OUTPUT FILES
------------
  JSON + CSV : Solver results with metrics and binary solution vectors
  Figures    : 15 publication-quality plots (PNG + PDF formats)
               - 01_solver_comparison.png/pdf (3-panel bar chart)
               - 09_pareto_objective_tradeoffs.png/pdf (2-panel scatter)
               - 10_objective_achievement.png/pdf (2-panel bars)
               - 15_energy_produced.png/pdf (horizontal bar chart)
               - Plus 11 additional analysis figures (maps, heatmaps, radar, etc.)

NOTES
-----
  • CSV format is recommended for caching (instant reload, no re-computation)
  • Quantum hardware requires valid IBM_QUANTUM_INSTANCE_CRN in .env.local
  • QAOA hardware experiments may take 3-5 minutes per run (queue + execution time)
  • All figures use colorblind-friendly palettes
  • Budget/population constraints are soft penalties in QUBO formulation

For more details, see README.md and paper.tex in the repository root.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generator import generate_ethiopia_dataset
from qubo_builder import build_qubo, analyze_solution

# ── constants ────────────────────────────────────────────────────────────────
BUDGET = 900_000
MAX_GRIDS = 10
MIN_POPULATION = 5_000
NUM_SITES = 50
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# Consistent solver colors (colorblind-friendly palette)
SOLVER_COLORS = {
    "NAR Greedy":       "#4477AA",
    "Sim. Annealing":   "#228833",
    "Tabu Search":      "#CCBB44",
    "D-Wave QBSolv":      "#66CCEE",
    "QAOA (IBM Torino)": "#AA3377",
}

SOLVER_ORDER = list(SOLVER_COLORS.keys())

# ── matplotlib global styling ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.15,
})

# ═════════════════════════════════════════════════════════════════════════════
#  SOLVER RUNNERS
# ═════════════════════════════════════════════════════════════════════════════

def run_nar(df):
    """NAR greedy (in-process)."""
    df_copy = df.copy()
    df_copy["ratio"] = df_copy["Population_Coverage"] / df_copy["Installation_Cost_USD"]
    df_sorted = df_copy.sort_values("ratio", ascending=False)
    selected, cost = [], 0
    for idx, row in df_sorted.iterrows():
        if cost + row["Installation_Cost_USD"] <= BUDGET and len(selected) < MAX_GRIDS:
            selected.append(idx)
            cost += row["Installation_Cost_USD"]
    x = np.zeros(len(df)); 
    for i in selected: x[i] = 1
    return x


def run_sa(df, Q):
    """Simulated annealing via simanneal (in-process)."""
    from simanneal import Annealer

    class _SA(Annealer):
        def __init__(self, Q, state):
            self.Q = Q
            super().__init__(state)
        def move(self):
            i = np.random.randint(len(self.state))
            self.state[i] = 1 - self.state[i]
        def energy(self):
            x = np.array(self.state)
            return float(x @ self.Q @ x)

    # 100 independent restarts; return best solution found across all runs.
    # Matches the multi-read budget of D-Wave QBSolv for a fair comparison.
    rng = np.random.RandomState(42)
    best_energy = float("inf")
    best_state = None
    # Calibrate schedule once from a single trial run
    init0 = rng.randint(2, size=len(df)).tolist()
    sa0 = _SA(Q, init0)
    sa0.updates = 0
    schedule = sa0.auto(minutes=0.05)   # quick calibration probe
    for _ in range(100):
        init = rng.randint(2, size=len(df)).tolist()
        sa = _SA(Q, init)
        sa.set_schedule(schedule)
        sa.updates = 0          # suppress per-run progress table
        sa.copy_strategy = "slice"
        state, energy = sa.anneal()
        if energy < best_energy:
            best_energy = energy
            best_state = state
    return np.array(best_state, dtype=float)


def run_dwave_qbsolv(
    df,
    Q,
    max_iterations=100,
    record_progress=False,
    subproblem_size=None,
    num_reads=None,
    timeout_ms=None,
    progress_out_path=None,
    clear_subsamples=True,
):
    """D-Wave QBSolv-like runner using `dwave-hybrid` Tabu decomposition.

    Parameters
    - df, Q: dataset and QUBO matrix
    - max_iterations: number of hybrid decomposition iterations to run
    - record_progress: if True, save per-iteration metrics and draw trajectory

    Returns a binary solution vector (numpy array) or `None` if hybrid
    primitives are unavailable.
    """
    # Try to import hybrid primitives (optional dependency)
    try:
        from hybrid import EnergyImpactDecomposer, TabuSubproblemSampler, SplatComposer
        from hybrid.core import State, Runnable, SampleSet
    except Exception:
        return None

    import dimod

    n = Q.shape[0]
    # Build a dimod BinaryQuadraticModel from the numpy Q matrix (BINARY vartype)
    linear = {i: float(Q[i, i]) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            c = float(Q[i, j] + Q[j, i])
            if c != 0:
                quadratic[(i, j)] = c
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

    # Hybrid decomposition / sampling parameters (tunable)
    # Default (fast) settings; can be overridden via function args.
    default_subproblem_size = min(max(6, n // 3), n)
    default_num_reads = 4
    default_timeout_ms = 100

    if subproblem_size is None:
        subproblem_size = default_subproblem_size
    if num_reads is None:
        num_reads = default_num_reads
    if timeout_ms is None:
        timeout_ms = default_timeout_ms

    # Small helper to clear subsamples after splatting so subsequent
    # decomposer iterations don't reuse incompatible subsamples.
    class _ClearSubsamples(Runnable):
        def next(self, state, **runopts):
            return state.updated(subsamples=None)

    # Branch: choose high-impact variables, solve subproblems with Tabu,
    # then splat subsolutions back into global sampleset. Optionally clear
    # subsamples between iterations to avoid `initial_states` mismatches.
    branch = (
        EnergyImpactDecomposer(size=subproblem_size)
        | TabuSubproblemSampler(num_reads=num_reads, timeout=timeout_ms)
        | SplatComposer()
    )
    if clear_subsamples:
        branch = branch | _ClearSubsamples()

    # Initial state (uses hybrid.utils.min_sample by default)
    state = State.from_problem(bqm)
    best_sampleset = state.samples
    try:
        initial_energy = float(state.samples.first.energy)
    except Exception:
        initial_energy = None
    try:
        best_energy = float(best_sampleset.first.energy)
    except Exception:
        best_energy = float("inf")

    progress = []
    prev_cost = None
    debug_log = os.path.join(FIGURES_DIR, "dwave_qbsolv_debug.log")
    out_path = progress_out_path if progress_out_path else os.path.join(FIGURES_DIR, "dwave_qbsolv_progress.json")

    def _atomic_write_json(data, path):
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            os.replace(tmp, path)
        except Exception as e:
            try:
                with open(debug_log, "a") as _f:
                    _f.write(f"atomic write failed: {e}\n")
            except Exception:
                pass

    # energy range trackers for normalization
    energy_min = initial_energy if initial_energy is not None else float("inf")
    energy_max = initial_energy if initial_energy is not None else float("-inf")

    # write a short header for each run
    try:
        with open(debug_log, "a") as _f:
            _f.write(f"\n--- run_dwave_qbsolv run start: {time.asctime()} max_iter={max_iterations}\n")
    except Exception:
        pass

    # Iteratively run the branch to improve the global sample using subproblem
    # tabu search. Keep the best sample seen across iterations and record
    # per-iteration metrics if requested. We ensure progress is persisted to
    # disk on every iteration (atomic write) and once more on exit so partial
    # runs are preserved.
    try:
        for it in range(1, max_iterations + 1):
            try:
                state = branch.next(state)
            except Exception as e:
                # Log exception and attempt one retry before giving up
                import traceback
                tb = traceback.format_exc()
                msg = f"branch.next failed at iter {it}: {e}\n{tb}\n"
                print(msg)
                try:
                    with open(debug_log, "a") as f:
                        f.write(msg)
                except Exception:
                    pass
                # one retry
                try:
                    state = branch.next(state)
                except Exception as e2:
                    tb2 = traceback.format_exc()
                    msg2 = f"retry failed at iter {it}: {e2}\n{tb2}\n"
                    print(msg2)
                    try:
                        with open(debug_log, "a") as f:
                            f.write(msg2)
                    except Exception:
                        pass
                    break

            sampleset = state.samples
            try:
                energy = float(sampleset.first.energy)
            except Exception as e:
                # Log and continue to next iteration
                import traceback
                tb = traceback.format_exc()
                msg = f"failed to read energy at iter {it}: {e}\n{tb}\n"
                print(msg)
                try:
                    with open(debug_log, "a") as f:
                        f.write(msg)
                except Exception:
                    pass
                continue

            if energy < best_energy:
                best_energy = energy
                best_sampleset = sampleset

            # Extract the current best sample and compute analysis metrics
            sample_map = sampleset.first.sample
            x = np.array([sample_map.get(i, 0) for i in range(n)], dtype=float)
            analysis = analyze_solution(x, df)
            total_cost = float(analysis.get("total_cost", 0.0))
            total_population = float(analysis.get("total_population", 0.0))
            num_sites = int(analysis.get("num_sites", 0))
            budget_used_pct = total_cost / BUDGET * 100

            cost_change = abs(total_cost - prev_cost) if prev_cost is not None else 0.0
            prev_cost = total_cost

            # Dynamic energy normalization using min/max energies seen so far.
            #  - 0 corresponds to the worst-seen (highest) energy
            #  - 1 corresponds to the best-seen (lowest) energy
            # Update running min/max before normalizing.
            if energy < energy_min:
                energy_min = energy
            if energy > energy_max:
                energy_max = energy

            denom = energy_max - energy_min
            if denom > 1e-12:
                energy_norm = (energy_max - energy) / denom
                energy_norm = max(0.0, min(1.0, energy_norm))
            else:
                energy_norm = 0.0

            progress.append({
                "iteration": it,
                "energy": energy,
                "energy_norm": energy_norm,
                "total_cost": total_cost,
                "total_population": total_population,
                "num_sites": num_sites,
                "budget_used_pct": budget_used_pct,
                "cost_change": cost_change,
                "solution": [int(v) for v in x.tolist()],
            })

            # lightweight progress print for debugging
            try:
                print(f"  iter {it}: energy={energy:.2f}, cost=${total_cost:,.0f}, pop={total_population}")
            except Exception:
                pass

            # Persist progress atomically each iteration so partial runs are saved
            if record_progress:
                try:
                    _atomic_write_json(progress, out_path)
                except Exception as e:
                    try:
                        with open(debug_log, "a") as f:
                            f.write(f"failed per-iter write at iter {it}: {e}\n")
                    except Exception:
                        pass
    except KeyboardInterrupt:
        # Allow graceful interruption and fall through to final write
        try:
            with open(debug_log, "a") as f:
                f.write("run interrupted by KeyboardInterrupt\n")
        except Exception:
            pass
        print("run interrupted by user (KeyboardInterrupt)")
    finally:
        # Final persistence and figure generation (best-effort)
        if record_progress and progress:
            try:
                _atomic_write_json(progress, out_path)
                msg = f"wrote {len(progress)} progress entries to {out_path}\n"
                print(msg)
                try:
                    with open(debug_log, "a") as f:
                        f.write(msg)
                except Exception:
                    pass
            except Exception as e:
                emsg = f"failed to write progress JSON in finally: {e}\n"
                print(emsg)
                try:
                    with open(debug_log, "a") as f:
                        f.write(emsg)
                except Exception:
                    pass
            # Draw the trajectory figure (non-fatal on failure)
            try:
                fig_dwave_trajectory(progress, name="D-Wave QBSolv")
            except Exception as e:
                emsg = f"fig_dwave_trajectory failed: {e}\n"
                print(emsg)
                try:
                    with open(debug_log, "a") as f:
                        f.write(emsg)
                except Exception:
                    pass

    # Optionally record progress to disk and draw trajectory plot
    if record_progress and progress:
        out_path = os.path.join(FIGURES_DIR, "dwave_qbsolv_progress.json")
        try:
            with open(out_path, "w") as f:
                json.dump(progress, f, indent=2)
            msg = f"wrote {len(progress)} progress entries to {out_path}\n"
            print(msg)
            try:
                with open(debug_log, "a") as f:
                    f.write(msg)
            except Exception:
                pass
            # Draw the trajectory figure (non-fatal on failure)
            try:
                fig_dwave_trajectory(progress, name="D-Wave QBSolv")
            except Exception as e:
                emsg = f"fig_dwave_trajectory failed: {e}\n"
                print(emsg)
                try:
                    with open(debug_log, "a") as f:
                        f.write(emsg)
                except Exception:
                    pass
        except Exception as e:
            emsg = f"failed to write progress JSON: {e}\n"
            print(emsg)
            try:
                with open(debug_log, "a") as f:
                    f.write(emsg + "\n")
            except Exception:
                pass

    # Extract best sample (ensure variables ordered 0..n-1)
    sample = best_sampleset.first.sample
    return np.array([sample.get(i, 0) for i in range(n)], dtype=float)


def run_qaoa_local(df, Q_small, num_sites_local=12):
    """QAOA on local StatevectorSampler (reduced problem size)."""
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_algorithms.utils import algorithm_globals
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    algorithm_globals.random_seed = 42
    n = Q_small.shape[0]
    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var(name=f"x{i}")
    linear = {f"x{i}": float(Q_small[i, i]) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            c = float(Q_small[i, j] + Q_small[j, i])
            if c != 0:
                quadratic[(f"x{i}", f"x{j}")] = c
    qp.minimize(linear=linear, quadratic=quadratic)

    sampler = StatevectorSampler(seed=42)
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=300), reps=2)
    result = MinimumEigenOptimizer(qaoa).solve(qp)
    x = np.array([int(result.x[i]) for i in range(n)])
    # Pad to full 50-site vector
    x_full = np.zeros(NUM_SITES)
    x_full[:n] = x
    return x_full


def run_qaoa_hardware(df, Q, backend_name):
    """QAOA on real IBM Quantum hardware -- manual circuit management.

    IBM backends require ISA (Instruction-Set Architecture) circuits, so we:
      1. Convert QUBO -> Ising operator
      2. Build QAOAAnsatz (50 virtual qubits, 2 parameters for reps=1)
      3. Transpile to ISA (133 physical qubits on ibm_torino)
      4. Record virtual-to-physical qubit layout for result decoding
      5. Run a COBYLA optimisation loop via Session + SamplerV2
      6. Return the best binary solution found
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.translators import to_ising
    from scipy.optimize import minimize as sp_minimize

    from ibm_quantum_config import get_ibm_instance_crn, get_qiskit_ibm_token

    MAXITER = 15        # COBYLA iterations (each = 1 hardware job)
    SHOTS   = 4096

    n = Q.shape[0]

    # ── 1. QUBO -> Ising ────────────────────────────────────────────────────
    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var(name=f"x{i}")
    linear = {f"x{i}": float(Q[i, i]) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            c = float(Q[i, j] + Q[j, i])
            if c != 0:
                quadratic[(f"x{i}", f"x{j}")] = c
    qp.minimize(linear=linear, quadratic=quadratic)
    operator, offset = to_ising(qp)

    # ── 2. Build QAOA ansatz ────────────────────────────────────────────────
    ansatz = QAOAAnsatz(cost_operator=operator, reps=1)
    ansatz.measure_all()
    param_names = [p.name for p in ansatz.parameters]
    print(f"  QAOA ansatz: {ansatz.num_qubits} qubits, "
          f"{ansatz.num_parameters} params ({param_names})")

    # ── 3. Connect & transpile ──────────────────────────────────────────────
    print(f"  Connecting to IBM Quantum ({backend_name})...")
    instance = get_ibm_instance_crn(required=True)
    token = get_qiskit_ibm_token()
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        instance=instance,
        token=token,
    )
    backend = service.backend(backend_name)
    print(f"  Backend: {backend.num_qubits} qubits")

    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
    isa_circuit = pm.run(ansatz)
    print(f"  Transpiled: {isa_circuit.num_qubits} physical qubits, "
          f"depth={isa_circuit.depth()}, gates={isa_circuit.size()}")

    # ── 4. Layout for decoding ──────────────────────────────────────────────
    virtual_to_physical = isa_circuit.layout.final_index_layout(
        filter_ancillas=True)

    def decode_counts(counts_dict):
        """Convert hardware counts -> (avg QUBO cost, best solution vector)."""
        best_c = float("inf")
        best_x = None
        avg = 0.0
        total = sum(counts_dict.values())
        for bitstring, cnt in counts_dict.items():
            # bitstring is MSB-first; reverse for positional indexing
            phys_bits = [int(b) for b in reversed(bitstring)]
            x_vec = np.zeros(n, dtype=float)
            for v in range(n):
                p = virtual_to_physical[v]
                if p < len(phys_bits):
                    x_vec[v] = phys_bits[p]
            cost = float(x_vec @ Q @ x_vec)
            avg += cost * cnt / total
            if cost < best_c:
                best_c = cost
                best_x = x_vec.copy()
        return avg, best_c, best_x

    # ── 5. COBYLA optimisation loop inside a Session ────────────────────────
    best_cost_overall = float("inf")
    best_solution_overall = None
    iteration = [0]

    print(f"  Running COBYLA ({MAXITER} iters, {SHOTS} shots each)...")

    # Direct backend mode (Session not available on open plan)
    sampler = SamplerV2(mode=backend)

    def objective(params):
        nonlocal best_cost_overall, best_solution_overall
        bound = isa_circuit.assign_parameters(
            dict(zip(isa_circuit.parameters, params)))
        job = sampler.run([bound], shots=SHOTS)
        result = job.result()
        counts = result[0].data.meas.get_counts()

        avg_c, best_c, best_x = decode_counts(counts)
        if best_c < best_cost_overall:
            best_cost_overall = best_c
            best_solution_overall = best_x
        iteration[0] += 1
        print(f"    Iter {iteration[0]:2d}/{MAXITER}: "
              f"avg={avg_c:,.0f}  best_this={best_c:,.0f}  "
              f"best_overall={best_cost_overall:,.0f}")
        return avg_c

    rng = np.random.default_rng(42)
    x0 = rng.uniform(0, np.pi, len(isa_circuit.parameters))
    sp_minimize(objective, x0, method="COBYLA",
                options={"maxiter": MAXITER, "rhobeg": 0.5})

    print(f"  Hardware QAOA finished. Best QUBO cost = {best_cost_overall:,.0f}")
    return best_solution_overall


def run_tabu(df, Q):
    """D-Wave Tabu sampler."""
    import dimod
    from dwave.samplers import TabuSampler
    n = Q.shape[0]
    linear = {i: float(Q[i, i]) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            c = float(Q[i, j] + Q[j, i])
            if c != 0:
                quadratic[(i, j)] = c
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)
    ss = TabuSampler().sample(bqm, num_reads=100)
    best = ss.first
    return np.array([best.sample[i] for i in range(n)], dtype=float)


# ═════════════════════════════════════════════════════════════════════════════
#  MASTER RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def _save_csv(results, path):
    """Save solver results to CSV so figures can be regenerated without re-running solvers."""
    rows = []
    for name, data in results.items():
        a = data["analysis"]
        rows.append({
            "solver":           name,
            "num_sites":        a["num_sites"],
            "total_cost":       a["total_cost"],
            "total_population": a["total_population"],
            "total_energy":     a["total_energy"],
            "budget_used_pct":  round(a["total_cost"] / BUDGET * 100, 2),
            "time_sec":         round(data["time"], 4),
            "solution":         " ".join(str(int(v)) for v in data["solution"]),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Results also saved to {path}")


def _load_csv(path, df):
    """Load solver results from CSV and reconstruct analysis from the solution vector."""
    results = {}
    frame = pd.read_csv(path)
    for _, row in frame.iterrows():
        x = np.array([float(v) for v in str(row["solution"]).split()])
        analysis = analyze_solution(x, df)
        results[str(row["solver"])] = {
            "solution": x,
            "analysis": analysis,
            "time":     float(row["time_sec"]),
        }
    return results


def run_all_solvers(df, Q, ibm_backend=None, qaoa_local_sites=12, run_qaoa_local=True):
    """Run every available solver and return {name: {solution, analysis, time}}."""
    results = {}

    runners = [
        ("NAR Greedy",     lambda: run_nar(df)),
        ("Sim. Annealing", lambda: run_sa(df, Q)),
        ("Tabu Search",    lambda: run_tabu(df, Q)),
        ("D-Wave QBSolv",    lambda: run_dwave_qbsolv(df, Q)),
    ]

    for name, fn in runners:
        print(f"  Running {name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            x = fn()
            elapsed = time.time() - t0
            if x is None:
                print("SKIPPED (not installed)")
                continue
            analysis = analyze_solution(x, df)
            results[name] = {"solution": x, "analysis": analysis, "time": elapsed}
            print(f"{elapsed:.2f}s  [{analysis['num_sites']} sites, "
                  f"${analysis['total_cost']:,}]")
        except Exception as e:
            print(f"FAILED: {e}")

    # QAOA local (reduced problem) — skipped when run_qaoa_local=False
    if run_qaoa_local:
        print(f"  Running QAOA (local, {qaoa_local_sites} sites)...", end=" ", flush=True)
        t0 = time.time()
        try:
            # Use the first N rows of the original df so site indices align
            # correctly when the solution is evaluated against the full dataset.
            df_small = df.iloc[:qaoa_local_sites].reset_index(drop=True)
            Q_small, _ = build_qubo(df_small, BUDGET, MAX_GRIDS, MIN_POPULATION)
            x = run_qaoa_local(df, Q_small, qaoa_local_sites)
            elapsed = time.time() - t0
            analysis = analyze_solution(x, df)
            results["QAOA (local)"] = {"solution": x, "analysis": analysis,
                                       "time": elapsed,
                                       "note": f"{qaoa_local_sites}-site subset"}
            print(f"{elapsed:.2f}s  [{analysis['num_sites']} sites, "
                  f"${analysis['total_cost']:,}]")
        except Exception as e:
            print(f"FAILED: {e}")

    # QAOA hardware (full 50 sites)
    if ibm_backend:
        print(f"  Running QAOA (hardware: {ibm_backend})...", flush=True)
        t0 = time.time()
        try:
            x = run_qaoa_hardware(df, Q, ibm_backend)
            elapsed = time.time() - t0
            analysis = analyze_solution(x, df)
            results["QAOA (IBM Torino)"] = {"solution": x, "analysis": analysis,
                                          "time": elapsed,
                                          "backend": ibm_backend}
            print(f"  {elapsed:.2f}s  [{analysis['num_sites']} sites, "
                  f"${analysis['total_cost']:,}]")
        except Exception as e:
            print(f"  FAILED: {e}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  HELPER
# ═════════════════════════════════════════════════════════════════════════════

def _save(fig, name):
    """Save figure as PNG and PDF."""
    for ext in ("png", "pdf"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path)
    plt.close(fig)
    print(f"    saved figures/{name}.png/pdf")


def _get_ordered(results):
    """Return (names, colors) in consistent order, only for solvers with results."""
    names = [n for n in SOLVER_ORDER if n in results]
    colors = [SOLVER_COLORS[n] for n in names]
    return names, colors


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE GENERATORS
# ═════════════════════════════════════════════════════════════════════════════

def fig01_solver_comparison(results, df):
    """3-panel bar chart reflecting paper QUBO objectives: cost, population, energy."""
    names, colors = _get_ordered(results)
    metrics = [
        ("Total Installation Cost (USD)", "total_cost",       "${x:,.0f}"),
        ("Population Coverage",           "total_population", "{x:,.0f}"),
        ("Energy Capacity (kWh/day)",     "total_energy",     "{x:,.0f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (title, key, fmt) in zip(axes, metrics):
        vals = [results[n]["analysis"][key] for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="black",
                      linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _, f=fmt: f.format(x=x)))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    fmt.format(x=v), ha="center", va="bottom", fontsize=7)
    fig.suptitle("Solver Comparison -- Ethiopia Microgrid Optimization",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "01_solver_comparison")


def fig02_solve_time(results):
    """Bar chart of solve times (log scale)."""
    names, colors = _get_ordered(results)
    times = [results[n]["time"] for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names)), times, color=colors, edgecolor="black",
                  linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Time (seconds, log scale)")
    ax.set_title("Solver Execution Time Comparison", fontweight="bold")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.15,
                f"{t:.2f}s", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, "02_solve_time")


def fig03_geographic_all(df):
    """Map of all 50 candidate sites."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(df["X_coord"], df["Y_coord"],
                    s=df["Population_Coverage"] / 15,
                    c=df["Solar_Potential_kWh_m2_day"],
                    cmap="YlOrRd", edgecolors="black", linewidth=0.5,
                    alpha=0.85, zorder=3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("Solar Potential (kWh/m\u00b2/day)")
    for _, row in df.iterrows():
        ax.annotate(row["Site_ID"], (row["X_coord"], row["Y_coord"]),
                    fontsize=5, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Ethiopia -- 50 Candidate Microgrid Sites\n"
                 "(marker size = population coverage)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "03_geographic_all_sites")


def fig04_geographic_selected(results, df):
    """Multi-panel map: selected sites per solver."""
    names, colors = _get_ordered(results)
    ncols = min(3, len(names))
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    for i, name in enumerate(names):
        ax = axes[i]
        x = results[name]["solution"]
        sel = x == 1
        ax.scatter(df.loc[~sel, "X_coord"], df.loc[~sel, "Y_coord"],
                   s=30, c="lightgray", edgecolors="gray", linewidth=0.3,
                   label="Not selected", zorder=2)
        ax.scatter(df.loc[sel, "X_coord"], df.loc[sel, "Y_coord"],
                   s=df.loc[sel, "Population_Coverage"] / 15,
                   c=SOLVER_COLORS[name], edgecolors="black", linewidth=0.6,
                   label="Selected", zorder=3)
        a = results[name]["analysis"]
        ax.set_title(f"{name}\n{a['num_sites']} sites, "
                     f"${a['total_cost']:,}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.grid(True, alpha=0.2)
    # Hide extra axes
    for j in range(len(names), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Selected Sites by Solver", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "04_geographic_selected")


def fig05_consensus(results, df):
    """Site selection consensus heatmap."""
    names, _ = _get_ordered(results)
    consensus = np.zeros(len(df))
    for n in names:
        consensus += results[n]["solution"]
    order = np.argsort(consensus)[::-1]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors_bar = plt.cm.RdYlGn(consensus[order] / max(consensus.max(), 1))
    ax.bar(range(len(df)), consensus[order], color=colors_bar,
           edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([df.iloc[i]["Site_ID"] for i in order],
                       rotation=90, fontsize=6)
    ax.set_ylabel(f"Selected by N solvers (out of {len(names)})")
    ax.set_title("Site Selection Consensus Across Solvers", fontweight="bold")
    ax.set_ylim(0, len(names) + 0.5)
    fig.tight_layout()
    _save(fig, "05_consensus")


def fig06_sa_convergence(df, Q):
    """SA convergence curve (run a fresh SA with progress tracking)."""
    from simanneal import Annealer

    class _SATrack(Annealer):
        def __init__(self, Q, state):
            self.Q = Q
            self.history = []
            super().__init__(state)
        def move(self):
            i = np.random.randint(len(self.state))
            self.state[i] = 1 - self.state[i]
        def energy(self):
            x = np.array(self.state)
            return float(x @ self.Q @ x)
        def update(self, step, T, E, acceptance, improvement):
            if step % 50 == 0:
                self.history.append({"step": step, "energy": E, "T": T})

    init = np.random.RandomState(42).randint(2, size=Q.shape[0])
    sa = _SATrack(Q, init)
    sa.steps = 15_000
    sa.Tmax = 25_000.0
    sa.Tmin = 0.001
    sa.anneal()

    steps = [h["step"] for h in sa.history]
    energies = [h["energy"] for h in sa.history]
    temps = [h["T"] for h in sa.history]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(steps, energies, color="#228833", linewidth=1, label="QUBO Energy")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("QUBO Energy", color="#228833")
    ax1.tick_params(axis="y", labelcolor="#228833")

    ax2 = ax1.twinx()
    ax2.plot(steps, temps, color="#EE6677", linewidth=0.8, alpha=0.6,
             label="Temperature")
    ax2.set_ylabel("Temperature", color="#EE6677")
    ax2.tick_params(axis="y", labelcolor="#EE6677")
    ax2.set_yscale("log")

    ax1.set_title("Simulated Annealing Convergence", fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    _save(fig, "06_sa_convergence")


def fig_dwave_trajectory(progress, name="D-Wave QBSolv"):
    """Plot per-iteration trajectory for the D-Wave hybrid QBSolv runner.

    Expects `progress` as a list of dicts with keys: iteration, energy,
    energy_norm, total_cost, total_population, cost_change, ...
    All series are normalized to [0,1] for overlaying.
    """
    # If no progress data, generate a small placeholder image
    if not progress:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No progress data", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        _save(fig, "dwave_qbsolv_trajectory")
        return

    dfp = pd.DataFrame(progress)
    if dfp.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No progress data", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        _save(fig, "dwave_qbsolv_trajectory")
        return

    it = dfp["iteration"].astype(int)

    def _norm(col):
        s = col.astype(float)
        mn, mx = s.min(), s.max()
        if mx == mn:
            return np.zeros_like(s)
        return (s - mn) / (mx - mn)

    # Use full-series normalization of raw energy for a stable visual shape.
    # Keep direct min-max normalization to match prior trajectory visuals.
    if "energy" in dfp.columns:
        energy_norm = _norm(dfp["energy"]).to_numpy(dtype=float)
    elif "energy_norm" in dfp.columns:
        energy_norm = dfp["energy_norm"].astype(float).to_numpy()
    else:
        energy_norm = np.zeros(len(dfp), dtype=float)

    budget_norm = _norm(dfp["total_cost"]) if "total_cost" in dfp.columns else np.zeros(len(dfp))
    cost_change_norm = _norm(dfp["cost_change"]) if "cost_change" in dfp.columns else np.zeros(len(dfp))
    pop_norm = _norm(dfp["total_population"]) if "total_population" in dfp.columns else np.zeros(len(dfp))

    fig, ax = plt.subplots(figsize=(10, 5))
    c_energy = SOLVER_COLORS.get(name, "#66CCEE")
    # Continuous lines only (no shapes/markers) per user request
    ax.plot(it, energy_norm, label="Energy (norm)", color=c_energy, linewidth=2)
    ax.plot(it, budget_norm, label="Budget (norm)", color="#EE6677", linewidth=1.5)
    ax.plot(it, cost_change_norm, label="Cost change (norm)", color="#CCBB44", linewidth=1.5)
    ax.plot(it, pop_norm, label="Population (norm)", color="#4477AA", linewidth=1.5)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized value (0-1)")
    ax.grid(True, alpha=0.2)
    # Ensure the y-axis shows the full 0..1 band even for single points
    ax.set_ylim(-0.05, 1.05)
    # Ensure x-axis has sensible padding for single-point series
    if len(it) == 1:
        ax.set_xlim(it.iloc[0] - 1, it.iloc[0] + 1)
    else:
        ax.set_xlim(it.min() - 0.5, it.max() + 0.5)

    # Place legend at the top-right (semi-transparent) to avoid hiding data
    ax.legend(loc="upper right", frameon=True, fontsize=9, framealpha=0.85)
    # Use default tight layout
    fig.tight_layout()
    _save(fig, "dwave_qbsolv_trajectory")


def fig07_qubo_heatmap(Q):
    """QUBO matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = np.percentile(np.abs(Q), 99)
    im = ax.imshow(Q, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("QUBO Coefficient")
    ax.set_xlabel("Variable Index")
    ax.set_ylabel("Variable Index")
    ax.set_title(f"QUBO Matrix ({Q.shape[0]}x{Q.shape[0]})", fontweight="bold")
    fig.tight_layout()
    _save(fig, "07_qubo_matrix")


def fig08_budget_utilization(results):
    """Budget utilization comparison."""
    names, colors = _get_ordered(results)
    costs = [results[n]["analysis"]["total_cost"] for n in names]
    remaining = [BUDGET - c for c in costs]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(names)), costs, color=colors, edgecolor="black",
            linewidth=0.5, label="Spent")
    ax.barh(range(len(names)), remaining, left=costs, color="whitesmoke",
            edgecolor="gray", linewidth=0.3, label="Remaining")
    ax.axvline(BUDGET, color="red", linestyle="--", linewidth=1.2,
               label=f"Budget (${BUDGET:,})")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("USD")
    ax.set_title("Budget Utilization by Solver", fontweight="bold")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    _save(fig, "08_budget_utilization")


def fig09_pareto(results):
    """Pareto analysis: cost vs population and cost vs energy (objective trade-offs)."""
    names, colors = _get_ordered(results)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Cost vs Population
    ax = axes[0]
    for name in names:
        a = results[name]["analysis"]
        ax.scatter(a["total_cost"], a["total_population"],
                   s=200, c=SOLVER_COLORS[name], edgecolors="black",
                   linewidth=1, zorder=3, label=name)
        # Place label to the left of the marker for all solvers except NAR Greedy
        if name == "NAR Greedy":
            ha = "left"
            xytext = (8, 4)
        else:
            ha = "right"
            xytext = (-8, 4)
        ax.annotate(name, (a["total_cost"], a["total_population"]),
                fontsize=12, ha=ha, va="bottom",
                xytext=xytext, textcoords="offset points")
    ax.axvline(BUDGET, color="red", linestyle="--", alpha=0.5,
               label=f"Budget = ${BUDGET:,}")
    ax.set_xlabel("Total Cost (USD)", fontsize=14)
    ax.set_ylabel("Total Population Coverage", fontsize=14)
    ax.set_title("Cost vs Population Coverage", fontweight="bold", fontsize=16)
    ax.legend(fontsize=12, loc="lower right")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"))
    ax.grid(True, alpha=0.3)

    # Panel 2: Cost vs Energy
    ax = axes[1]
    for name in names:
        a = results[name]["analysis"]
        ax.scatter(a["total_cost"], a["total_energy"],
                   s=200, c=SOLVER_COLORS[name], edgecolors="black",
                   linewidth=1, zorder=3, label=name)
        # Place label to the left of the marker for all solvers except NAR Greedy
        if name == "NAR Greedy":
            ha = "left"
            xytext = (8, 4)
        else:
            ha = "right"
            xytext = (-8, 4)
        ax.annotate(name, (a["total_cost"], a["total_energy"]),
                fontsize=12, ha=ha, va="bottom",
                xytext=xytext, textcoords="offset points")
    ax.axvline(BUDGET, color="red", linestyle="--", alpha=0.5,
               label=f"Budget = ${BUDGET:,}")
    ax.set_xlabel("Total Cost (USD)", fontsize=14)
    ax.set_ylabel("Total Energy Capacity (kWh/day)", fontsize=14)
    ax.set_title("Cost vs Energy Capacity", fontweight="bold", fontsize=16)
    ax.legend(fontsize=12, loc="lower right")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"))
    ax.grid(True, alpha=0.3)

    # Removed suptitle per user request (title was moved to panel titles)
    fig.tight_layout()
    _save(fig, "09_pareto_objective_tradeoffs")


def fig10_objective_achievement(results):
    """Objective achievement: cost minimisation and population maximisation."""
    names, colors = _get_ordered(results)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Cost Minimisation (objective: minimise)
    ax = axes[0]
    costs = [results[n]["analysis"]["total_cost"] for n in names]
    ax.bar(range(len(names)), costs, color=colors, edgecolor="black",
           linewidth=0.5)
    ax.axhline(BUDGET, color="red", linestyle="--", linewidth=1.5,
               label=f"Budget ${BUDGET:,}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_title("Cost Minimisation", fontweight="bold")
    ax.set_ylabel("Total Installation Cost (USD)")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"))

    # Panel 2: Population Coverage Maximisation (objective: maximise)
    ax = axes[1]
    pops = [results[n]["analysis"]["total_population"] for n in names]
    ax.bar(range(len(names)), pops, color=colors, edgecolor="black",
           linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_title("Population Coverage Maximisation", fontweight="bold")
    ax.set_ylabel("Total Population Covered")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}"))

    fig.suptitle("Objective Achievement by Solver", fontsize=14,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "10_objective_achievement")


def fig11_site_distributions(df):
    """Site attribute distributions (box plots)."""
    cols = [("Installation_Cost_USD", "Installation Cost (USD)"),
            ("Population_Coverage", "Population Coverage"),
            ("Energy_Capacity_kWh_day", "Energy Capacity (kWh/day)"),
            ("Solar_Potential_kWh_m2_day", "Solar Potential (kWh/m\u00b2/day)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, (col, label) in zip(axes, cols):
        data = df[col].values
        bp = ax.boxplot(data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#4477AA", alpha=0.6),
                        medianprops=dict(color="red", linewidth=2))
        ax.set_ylabel(label)
        ax.set_title(f"Distribution of {label}", fontweight="bold")
        ax.set_xticks([])
        # Add individual points
        jitter = np.random.RandomState(42).normal(0, 0.02, len(data))
        ax.scatter(1 + jitter, data, s=15, c="#EE6677", alpha=0.5, zorder=3)
    fig.suptitle("Site Attribute Distributions (50 Ethiopia Sites)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "11_site_distributions")


def fig12_efficiency(results, df):
    """Population per dollar and energy per dollar."""
    names, colors = _get_ordered(results)
    pop_per_dollar = []
    energy_per_dollar = []
    for n in names:
        a = results[n]["analysis"]
        cost = max(a["total_cost"], 1)
        pop_per_dollar.append(a["total_population"] / cost * 1000)
        energy_per_dollar.append(a["total_energy"] / cost * 1000)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(range(len(names)), pop_per_dollar, color=colors,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Population per $1,000")
    ax.set_title("Population Efficiency", fontweight="bold")

    ax = axes[1]
    ax.bar(range(len(names)), energy_per_dollar, color=colors,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Energy (kWh/day) per $1,000")
    ax.set_title("Energy Efficiency", fontweight="bold")

    fig.suptitle("Efficiency Metrics Comparison", fontsize=14,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "12_efficiency")


def fig13_radar(results):
    """Radar/spider chart: multi-axis comparison."""
    names, colors = _get_ordered(results)
    axes_labels = ["Cost Eff.", "Population", "Energy",
                   "Budget Use", "Solve Speed"]

    # Compute raw values
    raw = {}
    for n in names:
        a = results[n]["analysis"]
        cost = max(a["total_cost"], 1)
        raw[n] = [
            a["total_population"] / cost * 1000,  # pop per $1k
            a["total_population"],
            a["total_energy"],
            min(a["total_cost"] / BUDGET, 1.0),   # budget utilization ratio
            1.0 / max(results[n]["time"], 0.001),  # speed (inverse time)
        ]

    # Normalize each axis to [0, 1]
    all_vals = np.array(list(raw.values()))
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1

    angles = np.linspace(0, 2 * np.pi, len(axes_labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name in names:
        normed = (np.array(raw[name]) - mins) / ranges
        values = normed.tolist() + normed[:1].tolist()
        ax.plot(angles, values, linewidth=1.5, label=name,
                color=SOLVER_COLORS[name])
        ax.fill(angles, values, alpha=0.1, color=SOLVER_COLORS[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=10)
    ax.set_title("Multi-Objective Solver Performance", fontweight="bold",
                 pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    _save(fig, "13_radar")


def fig14_solution_heatmap(results, df):
    """Binary solution matrix: solvers x sites."""
    names, _ = _get_ordered(results)
    matrix = np.array([results[n]["solution"] for n in names])
    fig, ax = plt.subplots(figsize=(16, 4 + 0.3 * len(names)))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", interpolation="none")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([df.iloc[i]["Site_ID"] for i in range(len(df))],
                       rotation=90, fontsize=5)
    ax.set_title("Solution Matrix (blue = selected)", fontweight="bold")
    ax.set_xlabel("Sites")
    fig.tight_layout()
    _save(fig, "14_solution_heatmap")


def fig15_energy_produced(results, df):
    """Daily energy capacity secured by each solver (objective: maximise)."""
    names, colors = _get_ordered(results)
    energies = [results[n]["analysis"]["total_energy"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    # Explicit per-bar loop so every bar gets exactly its solver colour.
    for i, (name, energy, color) in enumerate(zip(names, energies, colors)):
        ax.barh(i, energy, color=color, edgecolor="black", linewidth=0.5)
        ax.text(energy + max(energies) * 0.01, i, f"{energy:,.0f} kWh/day",
                va="center", fontsize=8)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Total Energy Capacity (kWh/day)")
    ax.set_title("Daily Energy Capacity Secured by Solver", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}"))
    ax.set_xlim(left=0)
    fig.tight_layout()
    _save(fig, "15_energy_produced")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality plots for the research paper"
    )
    parser.add_argument("--ibm_backend", type=str, default=None,
                        help="IBM backend for real hardware QAOA (e.g. ibm_torino)")
    parser.add_argument("--load", type=str, default=None,
                        help="Load results from a JSON file instead of re-running solvers")
    parser.add_argument("--from_csv", type=str, default=None,
                        help="Load results from a CSV file instead of re-running solvers")
    parser.add_argument("--solvers", type=str, default=None,
                        help="Comma-separated solver names to include in figures when loading "
                             "from CSV or JSON (e.g. \"NAR Greedy,Sim. Annealing\"). "
                             "Omit to include all saved solvers.")
    parser.add_argument("--save_results", "--save_result", type=str, default="solver_results.json",
                        help="Save solver results to this JSON file (or inside a directory path)")
    parser.add_argument("--qaoa_local_sites", type=int, default=12,
                        help="Number of sites for local QAOA simulation")
    parser.add_argument("--no_qaoa_local", action="store_true",
                        help="Skip local QAOA simulation (useful when only hardware QAOA is needed)")
    parser.add_argument("--rerun", type=str, default=None,
                        help="Comma-separated solver names to re-run fresh (must be used with "
                             "--from_csv or --load to load the rest from cache). "
                            'e.g. --rerun \'D-Wave QBSolv\'' )
    parser.add_argument("--record_progress", action="store_true",
                        help="Record per-iteration progress for iterative solvers (D-Wave)")
    parser.add_argument("--dwave_max_iter", type=int, default=20,
                        help="Maximum hybrid iterations for D-Wave QBSolv runner")
    parser.add_argument("--dwave_subproblem_size", type=int, default=None,
                        help="Subproblem size for D-Wave QBSolv runner (default: auto)")
    parser.add_argument("--dwave_num_reads", type=int, default=None,
                        help="Tabu num_reads for D-Wave QBSolv runner (default: auto)")
    parser.add_argument("--dwave_timeout_ms", type=int, default=None,
                        help="Tabu timeout in ms for D-Wave QBSolv runner (default: auto)")
    parser.add_argument("--dwave_no_clear_subsamples", action="store_true",
                        help="Disable subsample clearing between D-Wave hybrid iterations")
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)

    # ── Generate dataset and QUBO ────────────────────────────────────────
    print("=" * 60)
    print("  MICROGRID OPTIMIZATION -- RESEARCH PAPER FIGURES")
    print("=" * 60)
    df = generate_ethiopia_dataset(NUM_SITES)
    Q, offset = build_qubo(df, BUDGET, MAX_GRIDS, MIN_POPULATION)
    print(f"Dataset: {len(df)} sites | QUBO: {Q.shape} | Offset: {offset:.2f}")
    print()

    # ── Run or load solvers ──────────────────────────────────────────────
    # Map of solver name → callable, used by --rerun
    _SOLVER_FNS = {
        "NAR Greedy":     lambda: run_nar(df),
        "Sim. Annealing": lambda: run_sa(df, Q),
        "Tabu Search":    lambda: run_tabu(df, Q),
        "D-Wave QBSolv":  lambda: run_dwave_qbsolv(df, Q,
                                                  max_iterations=args.dwave_max_iter,
                                                  record_progress=args.record_progress,
                                                  subproblem_size=args.dwave_subproblem_size,
                                                  num_reads=args.dwave_num_reads,
                                                  timeout_ms=args.dwave_timeout_ms,
                                                  clear_subsamples=not args.dwave_no_clear_subsamples),
        "QAOA (IBM Torino)": lambda: run_qaoa_hardware(df, Q, args.ibm_backend),
    }

    def _run_one(name):
        """Run a single named solver and return its results dict entry."""
        fn = _SOLVER_FNS.get(name)
        if fn is None:
            print(f"  Unknown solver '{name}' -- skipping")
            return None
        print(f"  Re-running {name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            x = fn()
            elapsed = time.time() - t0
            if x is None:
                print("SKIPPED (not installed)")
                return None
            analysis = analyze_solution(x, df)
            print(f"{elapsed:.2f}s  [{analysis['num_sites']} sites, ${analysis['total_cost']:,}]")
            return {"solution": x, "analysis": analysis, "time": elapsed}
        except Exception as e:
            print(f"FAILED: {e}")
            return None

    if args.from_csv and os.path.exists(args.from_csv):
        print(f"Loading results from {args.from_csv}...")
        results = _load_csv(args.from_csv, df)
        # --rerun: replace specific solvers with a fresh run
        if args.rerun:
            for name in [s.strip() for s in args.rerun.split(",")]:
                entry = _run_one(name)
                if entry is not None:
                    results[name] = entry
        # --ibm_backend: run hardware QAOA and merge
        if args.ibm_backend and "QAOA (IBM Torino)" not in results:
            entry = _run_one("QAOA (IBM Torino)")
            if entry is not None:
                results["QAOA (IBM Torino)"] = entry
    elif args.load and os.path.exists(args.load):
        print(f"Loading results from {args.load}...")
        with open(args.load, "r") as f:
            saved = json.load(f)
        results = {}
        for name, data in saved.items():
            results[name] = {
                "solution": np.array(data["solution"]),
                "analysis": data["analysis"],
                "time":     data["time"],
            }
        # --rerun: replace specific solvers with a fresh run
        if args.rerun:
            for name in [s.strip() for s in args.rerun.split(",")]:
                entry = _run_one(name)
                if entry is not None:
                    results[name] = entry
        # --ibm_backend: run hardware QAOA and merge
        if args.ibm_backend and "QAOA (IBM Torino)" not in results:
            entry = _run_one("QAOA (IBM Torino)")
            if entry is not None:
                results["QAOA (IBM Torino)"] = entry
    else:
        print("Running solvers...")
        results = run_all_solvers(df, Q, ibm_backend=args.ibm_backend,
                                  qaoa_local_sites=args.qaoa_local_sites,
                                  run_qaoa_local=not args.no_qaoa_local)

    # ── Apply --solvers filter ───────────────────────────────────────────
    if args.solvers:
        keep = {s.strip() for s in args.solvers.split(",")}
        removed = [n for n in list(results) if n not in keep]
        for n in removed:
            del results[n]
        if removed:
            print(f"  Filtered out: {', '.join(removed)}")
        print(f"  Plotting {len(results)} solver(s): {', '.join(results)}")

    # ── Save / re-save results ───────────────────────────────────────────
    serializable = {}
    for name, data in results.items():
        serializable[name] = {
            "solution":  data["solution"].tolist() if hasattr(data["solution"], "tolist")
                         else data["solution"],
            "analysis":  {k: (v if not isinstance(v, (np.integer, np.floating))
                          else float(v))
                          for k, v in data["analysis"].items()
                          if k != "selected_sites"},
            "time":      data["time"],
        }
    save_target = Path(args.save_results)
    if save_target.exists() and save_target.is_dir():
        save_path = save_target / "solver_results.json"
    else:
        save_path = save_target
        if save_path.parent and not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {save_path}")
        # Also write CSV alongside the JSON
        csv_path = save_path.with_suffix(".csv")
        _save_csv(results, csv_path)
    except OSError as e:
        print(f"\nWARNING: Could not save results to {save_path}: {e}")

    print(f"\n{len(results)} solvers completed. Generating figures...\n")

    # ── Generate all figures ─────────────────────────────────────────────
    if len(results) >= 2:
        fig01_solver_comparison(results, df)
        fig02_solve_time(results)
        fig04_geographic_selected(results, df)
        fig05_consensus(results, df)
        fig08_budget_utilization(results)
        fig09_pareto(results)
        fig10_objective_achievement(results)
        fig12_efficiency(results, df)
        fig13_radar(results)
        fig14_solution_heatmap(results, df)
        fig15_energy_produced(results, df)

    # These don't need multiple solvers
    fig03_geographic_all(df)
    fig06_sa_convergence(df, Q)
    fig07_qubo_heatmap(Q)
    fig11_site_distributions(df)

    print(f"\nDone! {len(os.listdir(FIGURES_DIR))} files in figures/")


if __name__ == "__main__":
    main()
