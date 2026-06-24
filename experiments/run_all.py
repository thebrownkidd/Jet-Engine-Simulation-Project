"""
Driver: run the full discovery -> rollout -> forecasting -> RUL pipeline on
every C-MAPSS sub-dataset (FD001-FD004) and assemble a cross-dataset results
table.

Usage:
    python run_all.py            # all four datasets
    python run_all.py 1 3        # only FD001 and FD003

Outputs:
    experiments/artifacts/cross_dataset_results.csv
    experiments/artifacts/cross_dataset_results.json
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

import manifold_core as mc
import exp_discovery
import exp_rollout_stability as exp_a
import exp_health_forecasting as exp_b
import exp_rul_prediction as exp_c


def run_one(fd: int) -> dict:
    # train + cache the manifold once for this dataset
    mc.configure(fd, retrain=True)
    d = exp_discovery.main(fd)
    a = exp_a.main(fd)
    b = exp_b.main(fd)
    c = exp_c.main(fd)
    row = {"dataset": f"FD00{fd}"}
    row.update({k: v for k, v in d.items() if k != "fd"})
    row.update({k: v for k, v in a.items() if k != "fd"})
    row.update({k: v for k, v in b.items() if k != "fd"})
    row.update({k: v for k, v in c.items() if k != "fd"})
    return row


def main(fds):
    rows = [run_one(fd) for fd in fds]
    df = pd.DataFrame(rows)
    out_csv = os.path.join(mc.HERE, "artifacts", "cross_dataset_results.csv")
    out_json = os.path.join(mc.HERE, "artifacts", "cross_dataset_results.json")
    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("\n" + "=" * 78)
    print("CROSS-DATASET RESULTS")
    print("=" * 78)
    show = ["dataset", "n_regimes", "n_informative", "pca_rho2",
            "recon_mean_r2", "rho_var", "var_freerun_growth",
            "skill_cv_k20", "rul_rmse", "rul_r2", "base_rmse"]
    show = [c for c in show if c in df.columns]
    print(df[show].to_string(index=False))
    print(f"\nsaved {out_csv}")
    return df


if __name__ == "__main__":
    fds = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 4]
    main(fds)
