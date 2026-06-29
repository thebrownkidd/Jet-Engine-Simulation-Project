"""
ACML TASK 3 — Multi-seed robustness.

Are the key conclusions robust to initialisation? Runs seeds {0,1,2,3,4} for
three clearly-named variants on a representative single-regime (FD001) and
multi-regime (FD002) dataset:

  full           bounded AE + mono + smooth     (full_box rollout)
  unbounded_ae   unbounded latent AE, no pen     (projection="none")
  no_smooth      bounded AE, smooth=0            (full_box rollout)

Metrics per (dataset, variant, seed): recon mean R2, free-run norm + growth,
bounded flag, curvature kappa, forecast skill (cv,k=20), RUL RMSE/R2.

Outputs
  results/acml/tables/seed_robustness.csv
  results/acml/tables/seed_robustness_summary.csv   (mean +/- std)
  results/acml/tables/seed_robustness.tex
  results/acml/figures/seed_robustness_summary.png
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

DATASETS = [1, 2]
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = ac.ACML_EPOCHS

VARIANTS = [
    # name,         bounded, mono, smooth, projection
    ("full",         True,  5.0, 2.0, "full_box"),
    ("unbounded_ae", False, 0.0, 0.0, "none"),
    ("no_smooth",    True,  5.0, 0.0, "full_box"),
]


def run_one(fd, name, bounded, mono, smooth, projection, seed):
    tr_den, te_den = ac.setup_dataset(fd, k=2)
    man = ac.train_flex_ae(tr_den, k=2, bounded=bounded, lambda_mono=mono,
                           lambda_smooth=smooth, seed=seed, epochs=EPOCHS)
    mean_r2, _ = ac.recon_r2(man, te_den)
    kappa = ac.curvature_kappa(man, te_den)
    skill = ac.forecast_skill_cv(man, te_den, horizon=20)
    fn, growth, bnd = ac.freerun_growth(man, tr_den, te_den, projection)
    rul = ac.rul_metrics_kaware(man, seed=42)
    return dict(dataset=f"FD00{fd}", variant=name, seed=seed,
                recon_mean_r2=mean_r2, freerun_norm=fn, freerun_growth=growth,
                freerun_bounded=bnd, kappa=kappa, cv_skill_k20=skill,
                rul_rmse=rul["rul_rmse"], rul_r2=rul["rul_r2"])


def summarize(df):
    g = df.groupby(["dataset", "variant"])
    agg = g.agg(
        recon_mean=("recon_mean_r2", "mean"), recon_std=("recon_mean_r2", "std"),
        growth_mean=("freerun_growth", "mean"), growth_std=("freerun_growth", "std"),
        bounded_frac=("freerun_bounded", "mean"),
        kappa_mean=("kappa", "mean"), kappa_std=("kappa", "std"),
        skill_mean=("cv_skill_k20", "mean"), skill_std=("cv_skill_k20", "std"),
        rul_rmse_mean=("rul_rmse", "mean"), rul_rmse_std=("rul_rmse", "std"),
        rul_r2_mean=("rul_r2", "mean"), rul_r2_std=("rul_r2", "std"),
    ).reset_index()
    return agg


def make_figure(df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    metrics = [("freerun_growth", "free-run growth", True),
               ("cv_skill_k20", "forecast skill (cv,k=20)", False),
               ("rul_rmse", "RUL RMSE", False)]
    variants = df["variant"].unique().tolist()
    datasets = df["dataset"].unique().tolist()
    x = np.arange(len(variants))
    width = 0.38
    colors = {"FD001": "#4c72b0", "FD002": "#dd8452"}
    for ax, (col, title, logy) in zip(axes, metrics):
        for i, ds in enumerate(datasets):
            means, stds = [], []
            for v in variants:
                sub = df[(df.dataset == ds) & (df.variant == v)][col]
                means.append(sub.mean()); stds.append(sub.std())
            ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=3,
                   color=colors.get(ds), label=ds)
        if logy:
            ax.set_yscale("log")
            ax.axhline(ac.BOUNDED_GROWTH_THRESH, color="r", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(variants, rotation=20, fontsize=9)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Seed robustness (mean +/- std over seeds 0-4)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(fds=DATASETS):
    rows = []
    for fd in fds:
        for spec in VARIANTS:
            for seed in SEEDS:
                print(f"  FD00{fd} {spec[0]:<13} seed={seed} ...", flush=True)
                rows.append(run_one(fd, *spec, seed))
                r = rows[-1]
                print(f"    recon={r['recon_mean_r2']:.3f} growth=x{r['freerun_growth']:.2f} "
                      f"bnd={r['freerun_bounded']} skill={r['cv_skill_k20']:+.3f} "
                      f"RUL={r['rul_rmse']:.2f}")
    df = pd.DataFrame(rows)
    csv = os.path.join(ac.ACML_TAB, "seed_robustness.csv")
    df.to_csv(csv, index=False)
    agg = summarize(df)
    agg_csv = os.path.join(ac.ACML_TAB, "seed_robustness_summary.csv")
    agg.to_csv(agg_csv, index=False)

    tex_df = agg[["dataset", "variant", "recon_mean", "growth_mean",
                  "bounded_frac", "skill_mean", "rul_rmse_mean"]].copy()
    tex = ac.latex_table(tex_df, "Seed robustness (means over seeds 0-4). The "
                         "bounded model's stability and skill persist across "
                         "initialisations.", "tab:seed_robustness")
    with open(os.path.join(ac.ACML_TAB, "seed_robustness.tex"), "w", encoding="utf-8") as fh:
        fh.write(tex)

    fig_path = os.path.join(ac.ACML_FIG, "seed_robustness_summary.png")
    make_figure(df, fig_path)
    print("\n" + agg.to_string(index=False))
    print(f"\nsaved -> {csv}\nsaved -> {agg_csv}\nsaved -> {fig_path}")
    return df


if __name__ == "__main__":
    fds = [int(x) for x in sys.argv[1:]] or DATASETS
    main(fds)
