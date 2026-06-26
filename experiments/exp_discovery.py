"""
EXPERIMENT 0  --  IDENTIFIABILITY / DISCOVERY OF THE HEALTH MANIFOLD
===================================================================

Question (per dataset): after operating-condition normalisation, does the
degradation collapse onto a low-dimensional (k=2) health manifold, and does the
autoencoder reconstruct the informative sensors from just two latent
coordinates?

Reported per dataset
--------------------
  n_regimes               operating conditions detected
  n_informative           degradation-bearing sensors
  pca_rho1, pca_rho2      cumulative PCA variance with 1 / 2 components
  recon_mean_r2           mean test reconstruction R2 over informative sensors
  recon_min_r2            worst informative sensor

Figures (results/figures/FD00<fd>/)
  D1_health_trajectories.png   h0(t) for several engines (monotone wear)
  D2_manifold.png              PCA scree + 2-D health-state scatter
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
  sys.path.insert(0, SRC)

import manifold as mc


def main(fd: int = 1) -> dict:
    mc.configure(fd)
    print("=" * 78)
    print(f"EXPERIMENT 0  --  DISCOVERY  (FD00{fd})")
    print("=" * 78)
    man = mc.get_manifold()
    info = mc.discovery_info()

    train = mc.load_split("train")
    tr, _ = mc.split_by_unit(train)
    tr_den = mc.denoise(tr)
    test = mc.load_split("test")
    te_den = mc.denoise(test)

    # ---- PCA intrinsic dimensionality (on train denoised dynamic) -------- #
    Xtr = tr_den[mc.DYNAMIC].to_numpy()
    Xtr = (Xtr - Xtr.mean(0)) / (Xtr.std(0) + 1e-12)
    evr = PCA(n_components=min(6, Xtr.shape[1])).fit(Xtr).explained_variance_ratio_
    rho1, rho2 = float(evr[0]), float(evr[:2].sum())

    # ---- reconstruction R2 on TEST informative sensors ------------------- #
    h = man.encode(te_den)
    recon = man.decode(h)                                  # normalised frame
    cols = mc.DYNAMIC
    idx = [cols.index(s) for s in mc.INFORMATIVE]
    true = te_den[mc.INFORMATIVE].to_numpy()
    pred = recon[:, idx]
    r2s = {s: mc.r2_pooled(true[:, j], pred[:, j])
           for j, s in enumerate(mc.INFORMATIVE)}
    mean_r2 = float(np.mean(list(r2s.values())))
    min_r2 = float(np.min(list(r2s.values())))

    print(f"  regimes={info['n_regimes']}  dynamic={info['n_dynamic']}  "
          f"informative={info['n_informative']}")
    print(f"  PCA: rho1={rho1:.3f}  rho2={rho2:.3f}")
    print(f"  test reconstruction R2: mean={mean_r2:.3f}  min={min_r2:.3f}")

    # ---- D1: health trajectories ---------------------------------------- #
    H = mc.per_engine_health(man, tr_den)
    fig, ax = plt.subplots(figsize=(8, 5))
    uids = sorted(H["unit_id"].unique())[:25]
    for uid in uids:
        g = H[H["unit_id"] == uid].sort_values("cycle")
        ax.plot(g["cycle"], g["h0"], alpha=0.6, lw=1)
    ax.set_xlabel("cycle")
    ax.set_ylabel("health coordinate h0")
    ax.set_title(f"FD00{fd}: identified health state h0(t) (25 engines)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(mc.fig_dir(), "D1_health_trajectories.png"), dpi=130)
    plt.close(fig)

    # ---- D2: PCA scree + 2-D health scatter ----------------------------- #
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(np.arange(1, len(evr) + 1), evr, color="#4c72b0", alpha=0.85)
    ax[0].plot(np.arange(1, len(evr) + 1), np.cumsum(evr), "o-", color="#dd8452")
    ax[0].axhline(rho2, ls="--", color="grey", lw=1)
    ax[0].set_xlabel("principal component")
    ax[0].set_ylabel("explained variance ratio")
    ax[0].set_title(f"Intrinsic dimensionality (rho2={rho2:.2f})")
    ax[0].grid(alpha=0.3)

    sc = ax[1].scatter(H["h0"], H["h1"], c=H["d"], cmap="viridis", s=5, alpha=0.5)
    ax[1].set_xlabel("h0 (wear)")
    ax[1].set_ylabel("h1")
    ax[1].set_title("2-D health manifold (colour = life fraction)")
    fig.colorbar(sc, ax=ax[1], label="life fraction d")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(mc.fig_dir(), "D2_manifold.png"), dpi=130)
    plt.close(fig)

    return dict(fd=fd, n_regimes=info["n_regimes"],
                n_informative=info["n_informative"],
                pca_rho1=rho1, pca_rho2=rho2,
                recon_mean_r2=mean_r2, recon_min_r2=min_r2)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
