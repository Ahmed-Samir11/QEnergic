#!/usr/bin/env python3
"""Apply a chosen D-Wave progress iteration to saved solver result artifacts.

This updates:
- figures/solver_results.csv
- figures/solver_results.json

It is useful when a rerun trajectory contains the paper-matching iteration,
but the final returned sample differs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_generator import generate_ethiopia_dataset
from qubo_builder import analyze_solution

BUDGET = 900_000


def _load_progress(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Progress file is not a list: {path}")
    return data


def _find_iteration(progress: list[dict[str, Any]], iteration: int) -> dict[str, Any]:
    for row in progress:
        if int(row.get("iteration", -1)) == iteration:
            return row
    raise ValueError(f"Iteration {iteration} not found in progress JSON")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a D-Wave iteration to saved results")
    parser.add_argument("--progress", type=str, default="figures/dwave_qbsolv_progress.json")
    parser.add_argument("--csv", type=str, default="figures/solver_results.csv")
    parser.add_argument("--json", type=str, default="figures/solver_results.json")
    parser.add_argument("--iteration", type=int, default=27)
    parser.add_argument("--solver_name", type=str, default="D-Wave QBSolv")
    parser.add_argument("--time", type=float, default=None,
                        help="Optional override for D-Wave runtime in seconds")
    args = parser.parse_args()

    progress_path = ROOT / args.progress
    csv_path = ROOT / args.csv
    json_path = ROOT / args.json

    progress = _load_progress(progress_path)
    row = _find_iteration(progress, args.iteration)

    solution = [int(v) for v in row["solution"]]
    x = np.array(solution, dtype=float)
    df = generate_ethiopia_dataset(50)
    analysis = analyze_solution(x, df)

    total_cost = float(analysis["total_cost"])
    total_population = float(analysis["total_population"])
    total_energy = float(analysis["total_energy"])
    num_sites = int(analysis["num_sites"])
    budget_used_pct = round(total_cost / BUDGET * 100.0, 2)

    # Update CSV row
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = f.readline

    found = False
    for r in rows:
        if r.get("solver") == args.solver_name:
            r["num_sites"] = str(num_sites)
            r["total_cost"] = str(int(total_cost))
            r["total_population"] = str(int(total_population))
            r["total_energy"] = str(int(total_energy))
            r["budget_used_pct"] = f"{budget_used_pct:.2f}"
            r["solution"] = " ".join(str(v) for v in solution)
            if args.time is not None:
                r["time_sec"] = f"{args.time:.4f}"
            found = True
            break

    if not found:
        raise ValueError(f"Solver row '{args.solver_name}' not found in CSV")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Update JSON row
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            saved = json.load(f)
    else:
        saved = {}

    prev_time = None
    if isinstance(saved, dict) and args.solver_name in saved:
        prev_time = saved[args.solver_name].get("time")

    saved[args.solver_name] = {
        "solution": [float(v) for v in solution],
        "analysis": {
            "total_cost": total_cost,
            "total_population": total_population,
            "total_energy": total_energy,
            "num_sites": num_sites,
        },
        "time": float(args.time if args.time is not None else (prev_time if prev_time is not None else 0.0)),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2)

    print(
        f"Applied {args.solver_name} iteration {args.iteration}: "
        f"cost=${int(total_cost):,}, sites={num_sites}, "
        f"population={int(total_population):,}, energy={int(total_energy):,}."
    )


if __name__ == "__main__":
    main()
