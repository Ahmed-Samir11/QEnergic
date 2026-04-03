#!/usr/bin/env python3
"""Select best D-Wave iteration from progress JSON and update solver results.

Strategy: load per-iteration progress (must include 'solution'), compute a
simple normalized score balancing cost (min) and population (max):
  score = norm_cost + (1 - norm_population)
Lower is better. The script picks the iteration with minimal score, writes a
backup of CSV/JSON, updates `solver_results.csv` and `solver_results.json` for
the `D-Wave QBSolv` row, and writes a helper JSON with the chosen iteration.

Usage: python tools/select_best_dwave_iter.py
"""
import json
import os
from pathlib import Path
import sys
import math

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "dwave_qbsolv_progress.json"
CSV = ROOT / "solver_results.csv"
JSON_RES = ROOT / "solver_results.json"

if not FIG.exists():
    print(f"Progress file not found: {FIG}")
    sys.exit(1)

with open(FIG, "r") as f:
    progress = json.load(f)
if not progress:
    print("No progress entries found")
    sys.exit(1)

import numpy as np
import pandas as pd
# Ensure repository root is on sys.path so local modules import correctly
import sys as _sys
_sys.path.insert(0, str(ROOT))
from data_generator import generate_ethiopia_dataset
from qubo_builder import analyze_solution

df = generate_ethiopia_dataset(50)

dfp = pd.DataFrame(progress)

# Ensure solution exists
if "solution" not in dfp.columns:
    print("Progress JSON does not contain per-iteration 'solution' entries.")
    sys.exit(1)

# Use total_cost and total_population to compute normalized score
costs = dfp["total_cost"].astype(float)
pops = dfp["total_population"].astype(float)
cn_min, cn_max = costs.min(), costs.max()
pn_min, pn_max = pops.min(), pops.max()
cn_denom = cn_max - cn_min if cn_max != cn_min else 1.0
pn_denom = pn_max - pn_min if pn_max != pn_min else 1.0

norm_cost = (costs - cn_min) / cn_denom
norm_pop = (pops - pn_min) / pn_denom

# Score: lower is better (balance cost low and population high)
scores = norm_cost + (1.0 - norm_pop)
best_idx = int(np.argmin(scores.values))
best_row = dfp.loc[best_idx]

chosen = {
    "iteration": int(best_row["iteration"]),
    "energy": float(best_row["energy"]),
    "total_cost": float(best_row["total_cost"]),
    "total_population": float(best_row["total_population"]),
    "num_sites": int(best_row["num_sites"]),
    "solution": [int(v) for v in best_row["solution"]],
}

print("Chosen iteration:", chosen["iteration"],
      f"cost=${chosen['total_cost']:,.0f}",
      f"pop={int(chosen['total_population'])}")

# Update CSV: make a backup first
if CSV.exists():
    CSV_bak = CSV.with_suffix(".bak.csv")
    CSV_bak.write_bytes(CSV.read_bytes())
    frame = pd.read_csv(CSV)
    # Find D-Wave QBSolv row
    mask = frame["solver"] == "D-Wave QBSolv"
    if mask.any():
        i = frame.index[mask][0]
        frame.at[i, "num_sites"] = chosen["num_sites"]
        frame.at[i, "total_cost"] = chosen["total_cost"]
        frame.at[i, "total_population"] = chosen["total_population"]
        # total_energy left as-is (analysis will update below)
        frame.at[i, "solution"] = " ".join(str(int(v)) for v in chosen["solution"])
        frame.to_csv(CSV, index=False)
        print(f"Updated CSV {CSV} (backup -> {CSV_bak.name})")
    else:
        print("No 'D-Wave QBSolv' row found in CSV; not updating CSV.")

# Update JSON results if present
if JSON_RES.exists():
    try:
        with open(JSON_RES, "r") as f:
            saved = json.load(f)
    except Exception:
        saved = {}
    # Compute analysis using analyze_solution to ensure consistency
    x = np.array(chosen["solution"], dtype=float)
    analysis = analyze_solution(x, df)
    # Convert numpy types to native Python types for JSON serialization
    clean_analysis = {}
    for k, v in analysis.items():
        try:
            if hasattr(v, 'item'):
                clean_analysis[k] = v.item()
            else:
                clean_analysis[k] = v
        except Exception:
            clean_analysis[k] = v

    saved["D-Wave QBSolv"] = {
        "solution": chosen["solution"],
        "analysis": clean_analysis,
        "time": float(saved.get("D-Wave QBSolv", {}).get("time", 0.0)),
    }
    with open(JSON_RES, "w") as f:
        json.dump(saved, f, indent=2)
    print(f"Updated JSON results at {JSON_RES}")

# Save chosen iteration helper
out = ROOT / "figures" / "dwave_qbsolv_chosen_iteration.json"
with open(out, "w") as f:
    json.dump(chosen, f, indent=2)
print(f"Wrote chosen iteration to {out}")

print("Done.")
