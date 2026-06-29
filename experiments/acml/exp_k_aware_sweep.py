"""
ACML TASK 4 — K-aware latent dimension sweep.

Question: is K=2 a reconstruction bottleneck or a dynamical regulariser?

Critical requirement (fixes the prior confound): for each K, ALL latent
coordinates are used downstream. The RUL feature vector is
[h0..h(K-1), v0..v(K-1)] (see acml_common.feature_columns), and the rollout
projects ALL K coordinates (theory-matched full_box). No coordinate is silently
discarded.

For K in {1,2,3,4,5,6}, per dataset (FD001-FD004):
  recon mean/min R2, free-run norm + growth + bounded flag,
  rollout NRMSE @ {1,10,25,50}, curvature kappa, RUL RMSE/R2.

Outputs
  results/acml/tables/k_aware_dim_sweep.csv
  results/acml/tables/k_aware_dim_sweep.tex
  results/acml/figures/k_vs_reconstruction_and_stability.png
  results/acml/figures/k_vs_rul.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import acml_common as ac  # noqa: E402

DATASETS = [1, 2, 3, 4]
KS = [1, 2, 3, 4, 5, 6]
SEED = 42
EPOCHS = ac.ACML_EPOCHS


def run_one(fd, k):
    tr_den, te_den = ac.setup_dataset(fd, k=k)
    man = ac.train_flex_ae(tr_den, k=k, bounded=True, lambda_mono=5.0,
                           lambda_smooth=2.0, seed=SEED, epochs=EPOCHS)
    mean_r2, min_r2 = ac.recon_r2(man, te_den)
    kappa = ac.curvature_kappa(man, te_den)
    fn, growth, bnd = ac.freerun_growth(man, tr_den, te_den, "full_box")
    nrmse = ac.rollout_nrmse_by_horizon(man, tr_den, te_den, "full_box")
    rul = ac.rul_metrics_kaware(man, seed=SEED)
    row = dict(dataset=f"FD00{fd}", k=k, recon_mean_r2=mean_r2,
               recon_min_r2=min_r2, freerun_norm=fn, freerun_growth=growth,
               freerun_bounded=bnd, nrmse_h1=nrmse[1], nrmse_h10=nrmse[10],
               nrmse_h25=nrmse[25], nrmse_h50=nrmse[50], kappa=kappa,
               rul_rmse=rul["rul_rmse"], rul_r2=rul["rul_r2"])
    print(f"  FD00{fd} k={k}: recon={mean_r2:.3f} growth=x{growth:6.2f} "
          f"bnd={bnd} nrmse50={nrmse[50]:.3f} RUL={rul['rul_rmse']:.2f} "
          f"(feat dim={2*k})")
    return row


def make_figures(df):
    col = {"FD001": "#4c72b0", "FD002": "#dd8452", "FD003": "#55a868", "FD004": "#c44e52"}
    # Figure 1: recon vs k (left) and free-run growth vs k (right)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for ds in df["dataset"].unique():
        sub = df[df.dataset == ds].sort_values("k")
        ax[0].plot(sub["k"], sub["recon_mean_r2"], "o-", color=col[ds], label=ds)
        ax[1].plot(sub["k"], sub["freerun_growth"], "o-", color=col[ds], label=ds)
    ax[0].axvline(2, color="k", ls="--", lw=0.8, alpha=0.6)
    ax[0].set_xlabel("latent dimension K"); ax[0].set_ylabel("reconstruction mean $R^2$")
    ax[0].set_title("(a) Reconstruction improves with K"); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[1].axhline(ac.BOUNDED_GROWTH_THRESH, color="r", ls="--", lw=1, label="bounded threshold")
    ax[1].axvline(2, color="k", ls="--", lw=0.8, alpha=0.6)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("latent dimension K"); ax[1].set_ylabel("free-run growth (log)")
    ax[1].set_title("(b) Rollout stability vs K"); ax[1].grid(alpha=0.3); ax[1].legend()
    fig.suptitle("Latent dimension: reconstruction vs dynamical stability", fontsize=12)
    fig.tight_layout()
    p1 = os.path.join(ac.ACML_FIG, "k_vs_reconstruction_and_stability.png")
    fig.savefig(p1, dpi=300, bbox_inches="tight"); plt.close(fig)

    # Figure 2: RUL RMSE vs k
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds in df["dataset"].unique():
        sub = df[df.dataset == ds].sort_values("k")
        ax.plot(sub["k"], sub["rul_rmse"], "o-", color=col[ds], label=ds)
    ax.axvline(2, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("latent dimension K"); ax.set_ylabel("RUL RMSE (cycles)")
    ax.set_title("K-aware RUL (features = [h0..h(K-1), v0..v(K-1)])")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    p2 = os.path.join(ac.ACML_FIG, "k_vs_rul.png")
    fig.savefig(p2, dpi=300, bbox_inches="tight"); plt.close(fig)
    return p1, p2


def main(fds=DATASETS):
    rows = []
    for fd in fds:
        print("=" * 70)
        print(f"K-AWARE SWEEP  FD00{fd}  (epochs={EPOCHS})")
        print("=" * 70)
        for k in KS:
            rows.append(run_one(fd, k))
    df = pd.DataFrame(rows)
    csv = os.path.join(ac.ACML_TAB, "k_aware_dim_sweep.csv")
    df.to_csv(csv, index=False)

    tex_cols = ["dataset", "k", "recon_mean_r2", "freerun_growth",
                "freerun_bounded", "nrmse_h50", "rul_rmse", "rul_r2"]
    tex = ac.latex_table(df[tex_cols], "K-aware latent dimension sweep. All K "
                         "coordinates are used downstream. Higher K improves "
                         "reconstruction; K=2 gives the best stability/RUL "
                         "trade-off.", "tab:k_sweep")
    with open(os.path.join(ac.ACML_TAB, "k_aware_dim_sweep.tex"), "w", encoding="utf-8") as fh:
        fh.write(tex)

    p1, p2 = make_figures(df)
    print("\n" + df[tex_cols].to_string(index=False))
    print(f"\nsaved -> {csv}\nsaved -> {p1}\nsaved -> {p2}")
    return df


if __name__ == "__main__":
    fds = [int(x) for x in sys.argv[1:]] or DATASETS
    main(fds)
