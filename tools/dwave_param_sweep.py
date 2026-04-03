#!/usr/bin/env python3
"""Parameter sweep for D-Wave hybrid QBSolv runner.

Runs a small sweep over a few Tabu/decomposition parameter sets, selects
the best iteration per config (cost vs population tradeoff), and writes a
summary + updates `solver_results.csv` / `solver_results.json` with the
best overall found across configs.

Usage: python tools/dwave_param_sweep.py
"""
from pathlib import Path
import sys
import os
import json
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import generate_plots as gp

df = gp.generate_ethiopia_dataset(gp.NUM_SITES)
Q, offset = gp.build_qubo(df, gp.BUDGET, gp.MAX_GRIDS, gp.MIN_POPULATION)

# Sweep configs (labels, subproblem_size, num_reads, timeout_ms, max_iter, clear_subsamples)
configs = [
    {"label": "fast",     "subproblem_size": max(6, gp.NUM_SITES // 4), "num_reads": 6,  "timeout_ms": 200,  "max_iter": 30, "clear_subsamples": True},
    {"label": "moderate", "subproblem_size": max(8, gp.NUM_SITES // 3), "num_reads": 20, "timeout_ms": 500,  "max_iter": 30, "clear_subsamples": True},
    {"label": "strong",   "subproblem_size": max(12, gp.NUM_SITES // 2), "num_reads": 60, "timeout_ms": 1000, "max_iter": 30, "clear_subsamples": False},
]

summary = []
best_overall = None
best_score = float("inf")

for cfg in configs:
    label = cfg["label"]
    out_path = str(Path(gp.FIGURES_DIR) / f"dwave_qbsolv_progress_{label}.json")
    print(f"Running config '{label}' ({cfg['subproblem_size']=}, {cfg['num_reads']=}, {cfg['timeout_ms']=}) ...")
    t0 = time.time()
    sol = gp.run_dwave_qbsolv(
        df, Q,
        max_iterations=cfg["max_iter"],
        record_progress=True,
        subproblem_size=cfg["subproblem_size"],
        num_reads=cfg["num_reads"],
        timeout_ms=cfg["timeout_ms"],
        progress_out_path=out_path,
        clear_subsamples=cfg.get("clear_subsamples", True),
    )
    elapsed = time.time() - t0
    # load progress
    with open(out_path, "r") as f:
        progress = json.load(f)
    dfp = pd.DataFrame(progress)
    # compute normalized score using cost (min) and population (max)
    costs = dfp["total_cost"].astype(float)
    pops = dfp["total_population"].astype(float)
    cn_min, cn_max = costs.min(), costs.max()
    pn_min, pn_max = pops.min(), pops.max()
    cn_denom = cn_max - cn_min if cn_max != cn_min else 1.0
    pn_denom = pn_max - pn_min if pn_max != pn_min else 1.0
    norm_cost = (costs - cn_min) / cn_denom
    norm_pop = (pops - pn_min) / pn_denom
    scores = norm_cost + (1.0 - norm_pop)
    best_idx = int(np.argmin(scores.values))
    row = dfp.loc[best_idx]
    chosen = {
        "label": label,
        "iteration": int(row["iteration"]),
        "total_cost": float(row["total_cost"]),
        "total_population": float(row["total_population"]),
        "num_sites": int(row["num_sites"]),
        "time_sec": elapsed,
        "solution": [int(v) for v in row["solution"]],
    }
    summary.append(chosen)
    if float(scores.iloc[best_idx]) < best_score:
        best_score = float(scores.iloc[best_idx])
        best_overall = chosen.copy()
        best_overall["config"] = cfg
    print(f"  -> chosen iter {chosen['iteration']} cost=${chosen['total_cost']:,.0f} pop={int(chosen['total_population'])}")

# write sweep summary
out_summary = Path(gp.FIGURES_DIR) / "dwave_param_sweep_summary.json"
out_summary.write_text(json.dumps(summary, indent=2))
print(f"Saved summary to {out_summary}")

if best_overall:
    print("Best overall:", best_overall["label"], "iter", best_overall["iteration"],
          f"cost=${best_overall['total_cost']:,.0f}", "pop", int(best_overall['total_population']))
    # Update solver CSV (backup then write)
    csv_path = ROOT / "solver_results.csv"
    if csv_path.exists():
        bak = csv_path.with_suffix('.bak.csv')
        bak.write_bytes(csv_path.read_bytes())
        frame = pd.read_csv(csv_path)
        mask = frame["solver"] == "D-Wave QBSolv"
        if mask.any():
            i = frame.index[mask][0]
            frame.at[i, "num_sites"] = best_overall["num_sites"]
            frame.at[i, "total_cost"] = best_overall["total_cost"]
            frame.at[i, "total_population"] = best_overall["total_population"]
            frame.at[i, "solution"] = " ".join(str(int(v)) for v in best_overall["solution"])
            frame.to_csv(csv_path, index=False)
            print(f"Updated CSV {csv_path} (backup -> {bak.name})")
    # Update JSON results
    json_res = ROOT / "solver_results.json"
    try:
        saved = json.loads(json_res.read_text()) if json_res.exists() else {}
    except Exception:
        saved = {}
    from qubo_builder import analyze_solution
    analysis = analyze_solution(np.array(best_overall["solution"], dtype=float), df)
    clean = {k: (v.item() if hasattr(v, 'item') else v) for k, v in analysis.items()}
    saved["D-Wave QBSolv"] = {"solution": best_overall["solution"], "analysis": clean, "time": best_overall["time_sec"]}
    json_res.write_text(json.dumps(saved, indent=2))
    print(f"Updated JSON results at {json_res}")

    # Regenerate figures from updated CSV
    print("Regenerating figures from updated CSV...")
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "generate_plots.py"), "--from_csv", str(csv_path)])

print("Sweep complete.")
