#!/usr/bin/env python3
"""Run the D-Wave hybrid QBSolv runner for 100 iterations using the
previously selected 'moderate' parameters and save per-iteration progress.

Usage: python tools/run_dwave_100.py
"""
from pathlib import Path
import sys
import json
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import generate_plots as gp

def main():
    df = gp.generate_ethiopia_dataset(gp.NUM_SITES)
    Q, offset = gp.build_qubo(df, gp.BUDGET, gp.MAX_GRIDS, gp.MIN_POPULATION)

    subproblem_size = max(8, gp.NUM_SITES // 3)
    num_reads = 20
    timeout_ms = 500
    out_path = str(Path(gp.FIGURES_DIR) / "dwave_qbsolv_progress_100.json")

    print("Running D-Wave QBSolv (moderate) for 100 iterations...")
    t0 = time.time()
    gp.run_dwave_qbsolv(
        df,
        Q,
        max_iterations=100,
        record_progress=True,
        subproblem_size=subproblem_size,
        num_reads=num_reads,
        timeout_ms=timeout_ms,
        progress_out_path=out_path,
        clear_subsamples=True,
    )
    print(f"Run finished in {time.time()-t0:.1f}s; progress -> {out_path}")

if __name__ == "__main__":
    main()
