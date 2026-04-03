#!/usr/bin/env python3
"""Quick redraw of the D-Wave trajectory figure from saved progress JSON.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import generate_plots as gp

def main():
    p100 = Path(gp.FIGURES_DIR) / "dwave_qbsolv_progress_100.json"
    pdef = Path(gp.FIGURES_DIR) / "dwave_qbsolv_progress.json"
    if p100.exists():
        p = p100
    elif pdef.exists():
        p = pdef
    else:
        print("No progress JSON found in figures/ to redraw trajectory.")
        return
    progress = json.loads(p.read_text())
    gp.fig_dwave_trajectory(progress, name="D-Wave QBSolv")
    print(f"Redrew trajectory from {p}")

if __name__ == '__main__':
    main()
