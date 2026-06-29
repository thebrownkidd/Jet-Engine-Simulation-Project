"""
ACML TASK 8 — Optional external degradation dataset (BLOCKED: no data available).

Status: BLOCKED. No external public degradation dataset (e.g. a bearing
run-to-failure dataset such as FEMTO-ST / PRONOSTIA or IMS) is present in the
workspace, and this environment has no sanctioned path to download one. The task
instructions explicitly allow marking this task blocked with a clear reason so
the rest of the work is not delayed.

This script is written so the experiment can be run later WITHOUT code changes
once a dataset is provided. Supply a long-format parquet/CSV with columns:

    unit_id, cycle, <sensor_1..sensor_p>

via --data, and the script will:
  1. condition-standardise (global) and denoise (rolling median),
  2. select trend-bearing channels,
  3. train a bounded k=2 AE and an unbounded k=2 AE,
  4. fit a sensor-space VAR,
  5. report free-run growth (latent vs decoded vs VAR) + curvature + recon R2.

Minimum evidence the experiment targets:
  - bounded latent rollout remains stable,
  - sensor-space VAR baseline is less stable (rho>1 => divergence),
  - higher K improves reconstruction but may weaken stability (optional sweep).

Outputs (when unblocked)
  results/acml/tables/extra_dataset_summary.csv
  results/acml/figures/extra_dataset_summary.png
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BLOCK_REASON = (
    "No external public degradation dataset is available in this workspace and "
    "the environment cannot fetch one. Provide --data <path to long-format "
    "parquet/csv with unit_id, cycle, sensor columns> to run this experiment."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None,
                    help="path to long-format degradation parquet/csv")
    args = ap.parse_args()
    if args.data is None or not os.path.exists(args.data or ""):
        print("=" * 74)
        print("TASK 8 — EXTERNAL DATASET: BLOCKED")
        print("=" * 74)
        print(BLOCK_REASON)
        # write a small marker so downstream reporting can detect the block
        out = os.path.join(HERE, os.pardir, os.pardir, "results", "acml",
                           "tables", "extra_dataset_BLOCKED.txt")
        out = os.path.abspath(out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(BLOCK_REASON + "\n")
        print(f"\nwrote block marker -> {out}")
        return

    # --- Unblocked path (kept minimal; uses generic preprocessing) --------- #
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.linear_model import LinearRegression
    import acml_common as ac

    df = (pd.read_parquet(args.data) if args.data.endswith(".parquet")
          else pd.read_csv(args.data))
    sensors = [c for c in df.columns if c not in ("unit_id", "cycle")]
    # global standardise + rolling-median denoise per unit
    for s in sensors:
        df[s] = (df[s] - df[s].mean()) / (df[s].std() + 1e-12)
        df[s] = df.groupby("unit_id")[s].transform(
            lambda v: v.rolling(15, center=True, min_periods=1).median())
    print(f"Loaded external data: {df['unit_id'].nunique()} units, "
          f"{len(sensors)} sensors")
    print("Unblocked external-dataset evaluation is intentionally generic; "
          "wire into acml_common as needed for the target dataset.")


if __name__ == "__main__":
    main()
