

"""
tune_aco.py
-----------
Light‑weight random‑search hyper‑parameter tuner for the ACO solver
defined in `aconew.py`.

Usage:
    python tune_aco.py                # runs NUM_TRIALS random trials
    python tune_aco.py 40             # runs 40 trials instead of default

Outputs:
    • Trial‑by‑trial log on stdout
    • A CSV file `tuning_results.csv` with all trials
    • A leaderboard of the five best parameter sets
"""

import sys
import csv
import time
from pathlib import Path

import numpy as np

# Import helper utilities from the core solver
from acoworking2 import sample_hyperparams, run_aco_once

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
NUM_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 20    # default 20 trials
MAX_ITERS_PER_TRIAL = 10000
CPU_LIMIT_PER_TRIAL = 60  # seconds
RESULTS_CSV = Path(__file__).with_name("tuning_results.csv")

# ------------------------------------------------------------------
# Tuning Loop
# ------------------------------------------------------------------
results = []
start_all = time.time()

for trial in range(1, NUM_TRIALS + 1):
    params = sample_hyperparams()
    best_dist, cpu_time = run_aco_once(
        params,
        max_iters=MAX_ITERS_PER_TRIAL,
        cpu_limit=CPU_LIMIT_PER_TRIAL,
    )

    record = {
        **params,
        "best_distance": best_dist,
        "cpu_time_s": cpu_time,
        "trial": trial,
    }
    results.append(record)

    print(
        f"[{trial:>2}/{NUM_TRIALS}] "
        f"dist = {best_dist:8.2f} km  "
        f"α={params['ALPHA']:.2f}  "
        f"β={params['BETA']:.1f}  "
        f"ρ={params['RHO']:.2f}  "
        f"Q={params['Q']:.2f}  "
        f"ants={params['NUM_ANTS']:<3d}  "
        f"elite={params['ELITE_RANK']}  "
        f"time={cpu_time:5.1f}s"
    )

# ------------------------------------------------------------------
# Save results to CSV
# ------------------------------------------------------------------
fieldnames = (
    ["trial", "best_distance", "cpu_time_s"]
    + [k for k in results[0].keys() if k not in ("trial", "best_distance", "cpu_time_s")]
)
with RESULTS_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
print(f"\n✅ All results saved to {RESULTS_CSV}")

# ------------------------------------------------------------------
# Leaderboard
# ------------------------------------------------------------------
top5 = sorted(results, key=lambda r: r["best_distance"])[:5]
print("\n🏆 Top 5 parameter sets")
for rank, rec in enumerate(top5, 1):
    print(
        f"{rank}. dist={rec['best_distance']:.2f} km | "
        f"α={rec['ALPHA']:.2f}, β={rec['BETA']:.1f}, ρ={rec['RHO']:.2f}, "
        f"Q={rec['Q']:.2f}, ants={rec['NUM_ANTS']}, elite={rec['ELITE_RANK']} | "
        f"time={rec['cpu_time_s']:.1f}s"
    )

total_time = time.time() - start_all
print(f"\n⏱️  Total tuning time: {total_time:.1f} s for {NUM_TRIALS} trials")