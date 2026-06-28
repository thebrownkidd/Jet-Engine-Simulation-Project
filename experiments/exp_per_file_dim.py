"""
RESEARCH EXPERIMENT E5 -- PER-FILE LATENT DIMENSION
===================================================

E1a showed the k=2 bottleneck is lossy on FD002 (needs ~4 dims) and mildly so
on FD004 (~3).  This experiment trains ONE model PER FILE with a per-file
latent dimension and asks: does giving each dataset its E1a-recommended k
improve reconstruction and RUL WITHOUT breaking rollout boundedness?

Controlled comparison (same 2000-epoch budget, only k differs):
  k=2            the original shared bottleneck (baseline)
  k=recommended  FD001->2, FD002->4, FD003->2, FD004->3   (from E1a knees)

Metrics per (dataset, k):
  recon_mean_r2        held-out reconstruction (informative sensors)
  rul_rmse / rul_r2    official-test RUL via robust health->RUL map
  manifold_nrmse_max   max scored manifold-rollout NRMSE
  man_freerun_norm     400-step free-run norm (boundedness check)
  bounded              free-run growth < 5

Usage:  python exp_per_file_dim.py [fd ...]      (default: 1 2 3 4)

Outputs
  results/tables/research/per_file_dim/per_file_dim_summary.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import manifold as mc
import manifold.train as mtrain
from exp_ablation import recon_mean_r2, rul_metrics
from exp_ablation_rollout import latent_diagnostics, manifold_rollout_nrmse_max
from exp_rollout_stability import CUTOFF_FRAC, fit_sensor_var, free_run

SWEEP_EPOCHS = 2000
RECOMMENDED_K = {1: 2, 2: 4, 3: 2, 4: 3}   # from E1a intrinsic-dim knees


def run_one(fd: int, k: int) -> dict:
    mc.configure(fd, retrain=True, k=k, tag=f"perfile_k{k}")
    man = mc.get_manifold()
    df = mc.load_split("train")
    tr, te = mc.split_by_unit(df)
    tr_den = mc.denoise(tr)
    te_den = mc.denoise(te)

    var = fit_sensor_var(tr_den)
    longest = max(te_den["unit_id"].unique(),
                  key=lambda u: (te_den["unit_id"] == u).sum())
    g_long = te_den[te_den["unit_id"] == longest].sort_values("cycle").reset_index(drop=True)
    c0 = int(CUTOFF_FRAC * len(g_long))
    _, man_norm = free_run(var, man, g_long, c0, steps=400)
    growth = float(man_norm[-1] / (man_norm[0] + 1e-12))

    lat = latent_diagnostics(man, te_den)
    row = dict(dataset=f"FD00{fd}", k=k,
               recon_mean_r2=recon_mean_r2(man, te_den),
               manifold_nrmse_max=manifold_rollout_nrmse_max(man, te_den),
               man_freerun_norm=float(man_norm[-1]),
               bounded=bool(growth < 5.0),
               kappa=lat["kappa"], mono_viol_frac=lat["mono_viol_frac"])
    row.update(rul_metrics(man))
    print(f"  FD00{fd} k={k}: recon={row['recon_mean_r2']:.3f}  "
          f"RUL RMSE={row['rul_rmse']:.2f} R2={row['rul_r2']:+.3f}  "
          f"nrmse_max={row['manifold_nrmse_max']:.2f}  bounded={row['bounded']}")
    return row


def main(fds):
    mtrain.EPOCHS = SWEEP_EPOCHS
    out_dir = os.path.join(ROOT, "results", "tables", "research", "per_file_dim")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for fd in fds:
        print("=" * 78)
        print(f"PER-FILE LATENT DIM  FD00{fd}  (epochs={SWEEP_EPOCHS})")
        print("=" * 78)
        ks = sorted({2, RECOMMENDED_K[fd]})
        for k in ks:
            rows.append(run_one(fd, k))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "per_file_dim_summary.csv"), index=False)
    print("\n" + "=" * 78)
    print("PER-FILE LATENT DIM SUMMARY")
    print("=" * 78)
    cols = ["dataset", "k", "recon_mean_r2", "rul_rmse", "rul_r2",
            "manifold_nrmse_max", "man_freerun_norm", "bounded"]
    print(df[cols].to_string(index=False))
    print(f"\nsaved -> {os.path.join(out_dir, 'per_file_dim_summary.csv')}")
    return df


if __name__ == "__main__":
    fds = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 4]
    main(fds)
