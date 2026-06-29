"""
ACML TASK 9 — Publication-ready figures and tables.

Builds compact, consistent ACML assets from the CSVs produced by Tasks 1-7:

Figures
  fig_pipeline.png                      method pipeline schematic (drawn)
  fig_k_tradeoff.png                    K: reconstruction vs stability (compact)
  fig_ablation_mechanism.png            ablation/mechanism (curvature + RUL)
  fig_baselines.png                     baseline free-run growth (2x2)
  (re-uses Task figures for manifold, stability, health-vs-RUL)

Tables (LaTeX, booktabs)
  tab_main_cross_dataset.tex            main cross-dataset (from production CSV)
  plus the per-task .tex files already written by Tasks 2-7.

All figures are saved at 300 DPI, one-column friendly where possible.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import acml_common as ac  # noqa: E402

ROOT = ac.ROOT
PROD_TAB = os.path.join(ROOT, "results", "tables")
ASSET_FIG = os.path.join(ROOT, "results", "acml", "figures")
ASSET_TAB = os.path.join(ROOT, "results", "acml", "tables")
os.makedirs(ASSET_FIG, exist_ok=True)
os.makedirs(ASSET_TAB, exist_ok=True)


def fig_pipeline():
    stages = ["Raw multivariate\nsensors + controls",
              "Regime ID\n(KMeans on controls)",
              "Regime\nnormalisation",
              "Rolling-median\ndenoise",
              "Trend-based\nsensor selection",
              "Bounded latent AE\n(monotone, smooth)",
              "Bounded latent\nrollout + RUL"]
    fig, ax = plt.subplots(figsize=(15, 2.6))
    ax.axis("off")
    n = len(stages)
    x = np.linspace(0.07, 0.93, n)
    w = 0.10
    for i, (xi, s) in enumerate(zip(x, stages)):
        color = "#1b7837" if i >= n - 2 else "#4c72b0"
        box = FancyBboxPatch((xi - w / 2, 0.32), w, 0.36,
                             boxstyle="round,pad=0.012", linewidth=1.2,
                             edgecolor="black", facecolor=color, alpha=0.18)
        ax.add_patch(box)
        ax.text(xi, 0.5, s, ha="center", va="center", fontsize=8.2)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((xi + w / 2, 0.5), (x[i + 1] - w / 2, 0.5),
                         arrowstyle="-|>", mutation_scale=12, lw=1.1, color="black"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Proposed pipeline: regime-aware preprocessing → bounded latent "
                 "representation → stable rollout & RUL", fontsize=10)
    p = os.path.join(ASSET_FIG, "fig_pipeline.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {p}")


def fig_k_tradeoff():
    csv = os.path.join(ASSET_TAB, "k_aware_dim_sweep.csv")
    if not os.path.exists(csv):
        print("skip fig_k_tradeoff (missing k_aware_dim_sweep.csv)")
        return
    df = pd.read_csv(csv)
    col = {"FD001": "#4c72b0", "FD002": "#dd8452", "FD003": "#55a868", "FD004": "#c44e52"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for ds in df["dataset"].unique():
        sub = df[df.dataset == ds].sort_values("k")
        ax[0].plot(sub["k"], sub["recon_mean_r2"], "o-", color=col.get(ds), label=ds)
        ax[1].plot(sub["k"], sub["rul_rmse"], "o-", color=col.get(ds), label=ds)
    ax[0].axvline(2, color="k", ls="--", lw=0.8, alpha=0.6)
    ax[0].set_xlabel("latent dimension K"); ax[0].set_ylabel("reconstruction $R^2$")
    ax[0].set_title("(a) Reconstruction improves with K"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].axvline(2, color="k", ls="--", lw=0.8, alpha=0.6)
    ax[1].set_xlabel("latent dimension K"); ax[1].set_ylabel("RUL RMSE (cycles)")
    ax[1].set_title("(b) K-aware RUL (all coords used)"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.suptitle("Latent dimension as a regulariser: capacity up, prognosis not improved", fontsize=11)
    fig.tight_layout()
    p = os.path.join(ASSET_FIG, "fig_k_tradeoff.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {p}")


def fig_ablation_mechanism():
    csv = os.path.join(ASSET_TAB, "boundedness_mechanism.csv")
    if not os.path.exists(csv):
        print("skip fig_ablation_mechanism (missing boundedness_mechanism.csv)")
        return
    df = pd.read_csv(csv)
    configs = ["unbounded_no_pen", "bounded_no_pen", "bounded_smooth",
               "bounded_mono", "full"]
    configs = [c for c in configs if c in set(df["config"])]
    x = np.arange(len(configs)); width = 0.38
    colors = {"FD001": "#4c72b0", "FD002": "#dd8452"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, ds in enumerate(df["dataset"].unique()):
        sub = df[df.dataset == ds].set_index("config").loc[configs]
        ax[0].bar(x + (i - 0.5) * width, sub["kappa"], width, color=colors.get(ds), label=ds)
        ax[1].bar(x + (i - 0.5) * width, sub["rul_rmse"], width, color=colors.get(ds), label=ds)
    ax[0].set_yscale("log")
    ax[0].set_xticks(x); ax[0].set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
    ax[0].set_ylabel("curvature kappa (log)"); ax[0].set_title("(a) Penalties reduce curvature")
    ax[0].legend(fontsize=8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
    ax[1].set_ylabel("RUL RMSE"); ax[1].set_title("(b) Downstream RUL")
    ax[1].legend(fontsize=8)
    fig.suptitle("Mechanism: smoothness/monotonicity drive forecastability, not boundedness", fontsize=11)
    fig.tight_layout()
    p = os.path.join(ASSET_FIG, "fig_ablation_mechanism.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {p}")


def fig_baselines():
    csv = os.path.join(ASSET_TAB, "baselines_extended.csv")
    if not os.path.exists(csv):
        print("skip fig_baselines (missing baselines_extended.csv)")
        return
    df = pd.read_csv(csv)
    fds = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, ds in zip(axes.ravel(), fds):
        sub = df[df.dataset == ds]
        bars = ax.bar(sub["model"], sub["freerun_growth"],
                      color=["#1b7837" if m == "manifold" else "#b2182b" for m in sub["model"]])
        ax.axhline(5.0, color="r", ls="--", lw=1)
        ax.set_yscale("log"); ax.set_title(ds); ax.set_ylabel("free-run growth (log)")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Closed-loop free-run growth by model (incl. TCN)", fontsize=11)
    fig.tight_layout()
    p = os.path.join(ASSET_FIG, "fig_baselines.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {p}")


def tab_main_cross_dataset():
    csv = os.path.join(PROD_TAB, "cross_dataset_results.csv")
    if not os.path.exists(csv):
        print("skip tab_main_cross_dataset (missing production cross_dataset_results.csv)")
        return
    df = pd.read_csv(csv)
    keep = df[["dataset", "n_regimes", "pca_rho2", "recon_mean_r2", "rho_var",
               "var_freerun_growth", "man_freerun_norm", "rul_rmse", "rul_r2"]].copy()
    keep.columns = ["Dataset", "Regimes", "rho2", "ReconR2", "rhoVAR",
                    "VARgrowth", "ManifNorm", "RUL_RMSE", "RUL_R2"]
    tex = ac.latex_table(keep, "Main cross-dataset results of the final bounded "
                         "latent model. VAR diverges (rho>1, large growth) while "
                         "the manifold rollout stays bounded.",
                         "tab:main_cross_dataset")
    with open(os.path.join(ASSET_TAB, "tab_main_cross_dataset.tex"), "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"saved -> {os.path.join(ASSET_TAB, 'tab_main_cross_dataset.tex')}")


def main():
    fig_pipeline()
    fig_k_tradeoff()
    fig_ablation_mechanism()
    fig_baselines()
    tab_main_cross_dataset()
    print("\nACML assets built.")


if __name__ == "__main__":
    main()
