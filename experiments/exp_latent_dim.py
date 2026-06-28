"""
RESEARCH EXPERIMENT E1a -- LATENT CAPACITY & REGIME-MINING DIAGNOSTIC
====================================================================

Headline question
-----------------
Is the k=2 latent bottleneck *restricting* learning on the multi-regime /
multi-fault datasets (FD002, FD004)?

This is a capacity probe -- it does NOT retrain the full monotone pipeline.
It isolates two cheap, decisive signals per dataset:

1. INTRINSIC DIMENSIONALITY.  PCA on the condition-normalized, denoised
   *dynamic* sensor block.  We report the number of principal components
   needed to reach 90 / 95 / 99 % cumulative variance.  If FD002/FD004 need
   many more components than FD001/FD003, a 2-D bottleneck is structurally
   lossy for them.

2. RECONSTRUCTION R^2 vs BOTTLENECK k.  We train a *reconstruction-only*
   autoencoder (same HealthAE architecture, no monotonicity / smoothness
   penalties) for k = 1..K_MAX and measure held-out reconstruction R^2 on the
   informative sensors.  If R^2(k) is still climbing past k=2 on FD002/FD004
   but has plateaued by k=2 on FD001/FD003, that is direct evidence the
   bottleneck caps the harder datasets.

3. REGIME MINING (no hardcoded count).  We replace the
   `min(6, #unique-rounded-settings)` heuristic with a silhouette-driven
   KMeans model-selection sweep over K_reg in {1..8} and report the mined
   regime count vs the old heuristic.

Outputs
-------
results/tables/research/latent_dim/intrinsic_dim.csv
results/tables/research/latent_dim/recon_vs_k.csv
results/tables/research/latent_dim/regime_mining.csv
results/figures/research/latent_dim/recon_vs_k.png
results/figures/research/latent_dim/pca_scree.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import manifold as mc

K_MAX = 6
EPOCHS_PROBE = 1200       # capacity probe, not the production 4000-epoch model
LR = 5e-3
VAR_TARGETS = (0.90, 0.95, 0.99)
REGIME_KS = list(range(2, 9))
SETTINGS_CONSTANT_TOL = 1e-3   # below this settings spread => single regime

TAB_DIR = None
FIG_DIR = None


def _dirs():
    global TAB_DIR, FIG_DIR
    TAB_DIR = os.path.join(ROOT, "results", "tables", "research", "latent_dim")
    FIG_DIR = os.path.join(ROOT, "results", "figures", "research", "latent_dim")
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def _standardize(train_block: np.ndarray, *blocks: np.ndarray):
    mu = train_block.mean(0)
    sd = train_block.std(0) + 1e-12
    return tuple(((b - mu) / sd).astype(np.float32) for b in (train_block, *blocks))


def _train_reconstruction_ae(x_train: np.ndarray, k: int, n_in: int):
    """Reconstruction-only AE (no monotonicity / smoothness) -- pure capacity."""
    torch.manual_seed(mc.SEED)
    np.random.seed(mc.SEED)
    model = mc.HealthAE(n_in, k)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xt = torch.tensor(x_train)
    for _ in range(EPOCHS_PROBE):
        opt.zero_grad()
        recon, _ = model(xt)
        loss = ((recon - xt) ** 2).mean()
        loss.backward()
        opt.step()
    model.eval()
    return model


def _recon_r2(model, x_test: np.ndarray, inf_idx) -> tuple[float, float]:
    with torch.no_grad():
        recon, _ = model(torch.tensor(x_test))
    recon = recon.numpy()
    r2s = [mc.r2_pooled(x_test[:, j], recon[:, j]) for j in inf_idx]
    return float(np.mean(r2s)), float(np.min(r2s))


def mine_regimes(settings: np.ndarray) -> dict:
    """Data-driven regime count via silhouette over KMeans(K in REGIME_KS)."""
    spread = settings.std(0).max()
    if spread < SETTINGS_CONSTANT_TOL:
        return dict(mined_k=1, best_sil=float("nan"), sil_by_k={})
    sil_by_k = {}
    for K in REGIME_KS:
        km = KMeans(n_clusters=K, n_init=10, random_state=mc.SEED).fit(settings)
        labels = km.labels_
        if len(np.unique(labels)) < 2:
            continue
        # subsample for silhouette if very large
        idx = np.arange(len(settings))
        if len(idx) > 8000:
            rng = np.random.default_rng(mc.SEED)
            idx = rng.choice(idx, 8000, replace=False)
        sil_by_k[K] = float(silhouette_score(settings[idx], labels[idx]))
    if not sil_by_k:
        return dict(mined_k=1, best_sil=float("nan"), sil_by_k={})
    best_k = max(sil_by_k, key=sil_by_k.get)
    return dict(mined_k=best_k, best_sil=sil_by_k[best_k], sil_by_k=sil_by_k)


def main():
    _dirs()
    intrinsic_rows = []
    recon_rows = []
    regime_rows = []
    scree = {}

    for fd in (1, 2, 3, 4):
        mc.configure(fd)
        info = mc.discovery_info()
        dyn, inf = mc.DYNAMIC, mc.INFORMATIVE
        inf_idx = [dyn.index(s) for s in inf]

        train = mc.load_split("train")
        tr, te = mc.split_by_unit(train)
        tr_den = mc.denoise(tr)
        te_den = mc.denoise(te)

        Xtr = tr_den[dyn].to_numpy().astype(float)
        Xte = te_den[dyn].to_numpy().astype(float)
        Xtr_s, Xte_s = _standardize(Xtr, Xte)

        # ---- 1. intrinsic dimensionality (PCA) -------------------------- #
        pca = PCA(n_components=min(K_MAX + 2, Xtr_s.shape[1])).fit(Xtr_s)
        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr)
        scree[fd] = cum
        dims = {f"d{int(t*100)}": int(np.searchsorted(cum, t) + 1) for t in VAR_TARGETS}
        intrinsic_rows.append(dict(
            dataset=f"FD00{fd}", n_regimes_heuristic=info["n_regimes"],
            n_dynamic=len(dyn), n_informative=len(inf),
            rho1=float(cum[0]), rho2=float(cum[1]),
            rho3=float(cum[2]) if len(cum) > 2 else float("nan"),
            **dims))

        # ---- 2. reconstruction R^2 vs bottleneck k ---------------------- #
        for k in range(1, K_MAX + 1):
            model = _train_reconstruction_ae(Xtr_s, k, Xtr_s.shape[1])
            mean_r2, min_r2 = _recon_r2(model, Xte_s, inf_idx)
            recon_rows.append(dict(dataset=f"FD00{fd}", k=k,
                                   recon_mean_r2=mean_r2, recon_min_r2=min_r2))
            print(f"FD00{fd}  k={k}  recon mean R2={mean_r2:.3f}  min R2={min_r2:.3f}")

        # ---- 3. regime mining ------------------------------------------- #
        settings = train[mc.SETTINGS].to_numpy().astype(float)
        rm = mine_regimes(settings)
        regime_rows.append(dict(
            dataset=f"FD00{fd}", heuristic_k=info["n_regimes"],
            mined_k=rm["mined_k"], best_silhouette=rm["best_sil"],
            silhouette_by_k=str({k: round(v, 3) for k, v in rm["sil_by_k"].items()})))
        print(f"FD00{fd}  regimes: heuristic={info['n_regimes']}  mined={rm['mined_k']}  "
              f"(sil={rm['best_sil']:.3f})")

    intr = pd.DataFrame(intrinsic_rows)
    recon = pd.DataFrame(recon_rows)
    regimes = pd.DataFrame(regime_rows)
    intr.to_csv(os.path.join(TAB_DIR, "intrinsic_dim.csv"), index=False)
    recon.to_csv(os.path.join(TAB_DIR, "recon_vs_k.csv"), index=False)
    regimes.to_csv(os.path.join(TAB_DIR, "regime_mining.csv"), index=False)

    # ---- figures -------------------------------------------------------- #
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    col = {1: "#4c72b0", 2: "#dd8452", 3: "#55a868", 4: "#c44e52"}
    for fd in (1, 2, 3, 4):
        sub = recon[recon.dataset == f"FD00{fd}"]
        ax[0].plot(sub["k"], sub["recon_mean_r2"], "o-", color=col[fd],
                   label=f"FD00{fd}")
    ax[0].axvline(2, color="k", ls="--", lw=0.8, alpha=0.6)
    ax[0].set_xlabel("latent bottleneck dimension k")
    ax[0].set_ylabel("held-out reconstruction mean $R^2$")
    ax[0].set_title("(a) Reconstruction capacity vs latent dim")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    for fd in (1, 2, 3, 4):
        cum = scree[fd]
        ax[1].plot(np.arange(1, len(cum) + 1), cum, "o-", color=col[fd],
                   label=f"FD00{fd}")
    for t in VAR_TARGETS:
        ax[1].axhline(t, color="gray", ls=":", lw=0.7)
    ax[1].axvline(2, color="k", ls="--", lw=0.8, alpha=0.6)
    ax[1].set_xlabel("number of principal components")
    ax[1].set_ylabel("cumulative explained variance")
    ax[1].set_title("(b) Intrinsic dimensionality (PCA scree)")
    ax[1].grid(alpha=0.3)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "recon_vs_k.png"), dpi=160, bbox_inches="tight")

    print("\n" + "=" * 72)
    print("INTRINSIC DIMENSIONALITY")
    print("=" * 72)
    print(intr.to_string(index=False))
    print("\nREGIME MINING")
    print(regimes[["dataset", "heuristic_k", "mined_k", "best_silhouette"]].to_string(index=False))
    print(f"\nsaved tables -> {TAB_DIR}")
    print(f"saved figure -> {os.path.join(FIG_DIR, 'recon_vs_k.png')}")


if __name__ == "__main__":
    main()
