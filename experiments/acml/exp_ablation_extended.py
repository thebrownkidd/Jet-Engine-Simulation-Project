"""
ACML TASK 2 — Extended ablation study.

Goal: separate which component causes BOUNDEDNESS from which improves
FORECASTABILITY / downstream RUL.

Variants (all clearly named; nothing silently replaced)
  1. full                bounded AE, mono=5, smooth=2          (proposed)
  2. no_regime_norm      bounded AE, global standardisation
  3. no_mono             bounded AE, mono=0, smooth=2
  4. no_smooth           bounded AE, mono=5, smooth=0
  5. no_mono_smooth      bounded AE, mono=0, smooth=0
  6. bounded_no_pen      bounded AE, no penalties (== no_mono_smooth, kept as an
                         explicit reference for the boundedness argument)
  7. unbounded_ae        UNBOUNDED latent AE (sigmoid removed), no penalties,
                         rollout projection="none"  (mandatory contrast)

Bounded variants use the theory-matched full_box rollout projection. The
unbounded variant uses projection="none" so its (lack of) boundedness is honest.

Metrics: recon mean/min R2, mono-violation rate, curvature kappa, free-run
growth + bounded flag, rollout NRMSE @ {1,10,25,50}, forecast skill (cv,k=20),
RUL RMSE/MAE/R2 (k-aware features).

Representative datasets: FD001 (single-regime) + FD002 (multi-regime).

Outputs
  results/acml/tables/ablation_extended.csv
  results/acml/tables/ablation_extended.tex
  results/acml/figures/ablation_extended_summary.png
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
SEED = 42
EPOCHS = ac.ACML_EPOCHS

VARIANTS = [
    # name,            bounded, mono, smooth, normalize, projection
    ("full",            True,  5.0, 2.0, True,  "full_box"),
    ("no_regime_norm",  True,  5.0, 2.0, False, "full_box"),
    ("no_mono",         True,  0.0, 2.0, True,  "full_box"),
    ("no_smooth",       True,  5.0, 0.0, True,  "full_box"),
    ("no_mono_smooth",  True,  0.0, 0.0, True,  "full_box"),
    ("bounded_no_pen",  True,  0.0, 0.0, True,  "full_box"),
    ("unbounded_ae",    False, 0.0, 0.0, True,  "none"),
]


def run_variant(fd, name, bounded, mono, smooth, normalize, projection):
    tr_den, te_den = ac.setup_dataset(fd, k=2, normalize=normalize)
    man = ac.train_flex_ae(tr_den, k=2, bounded=bounded, lambda_mono=mono,
                           lambda_smooth=smooth, seed=SEED, epochs=EPOCHS)
    mean_r2, min_r2 = ac.recon_r2(man, te_den)
    kappa = ac.curvature_kappa(man, te_den)
    mono_viol = ac.mono_violation_fraction(man, te_den)
    skill = ac.forecast_skill_cv(man, te_den, horizon=20)
    fn, growth, bnd = ac.freerun_growth(man, tr_den, te_den, projection)
    nrmse = ac.rollout_nrmse_by_horizon(man, tr_den, te_den, projection)
    rul = ac.rul_metrics_kaware(man, seed=SEED)
    row = dict(dataset=f"FD00{fd}", variant=name, bounded_latent=bounded,
               recon_mean_r2=mean_r2, recon_min_r2=min_r2,
               mono_viol_frac=mono_viol, kappa=kappa,
               freerun_growth=growth, freerun_bounded=bnd,
               nrmse_h1=nrmse[1], nrmse_h10=nrmse[10],
               nrmse_h25=nrmse[25], nrmse_h50=nrmse[50],
               cv_skill_k20=skill,
               rul_rmse=rul["rul_rmse"], rul_mae=rul["rul_mae"],
               rul_r2=rul["rul_r2"], base_rmse=rul["base_rmse"])
    print(f"  FD00{fd} {name:<15} recon={mean_r2:.3f} kappa={kappa:.2e} "
          f"growth=x{growth:6.2f} bnd={bnd} skill={skill:+.3f} "
          f"RUL={rul['rul_rmse']:.2f}")
    return row


def make_figure(df, out_path):
    variants = df["variant"].unique().tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(variants))
    width = 0.38
    colors = {"FD001": "#4c72b0", "FD002": "#dd8452"}
    # (a) free-run growth (log) with bounded threshold
    ax = axes[0]
    for i, ds in enumerate(df["dataset"].unique()):
        sub = df[df.dataset == ds].set_index("variant").loc[variants]
        ax.bar(x + (i - 0.5) * width, sub["freerun_growth"], width,
               label=ds, color=colors.get(ds, None))
    ax.axhline(ac.BOUNDED_GROWTH_THRESH, color="r", ls="--", lw=1,
               label="bounded threshold")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(variants, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("free-run growth (log)")
    ax.set_title("(a) Boundedness by variant")
    ax.legend(fontsize=8)
    # (b) curvature kappa
    ax = axes[1]
    for i, ds in enumerate(df["dataset"].unique()):
        sub = df[df.dataset == ds].set_index("variant").loc[variants]
        ax.bar(x + (i - 0.5) * width, sub["kappa"], width, color=colors.get(ds, None), label=ds)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(variants, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("latent curvature kappa (log)")
    ax.set_title("(b) Forecastability proxy")
    ax.legend(fontsize=8)
    # (c) RUL RMSE
    ax = axes[2]
    for i, ds in enumerate(df["dataset"].unique()):
        sub = df[df.dataset == ds].set_index("variant").loc[variants]
        ax.bar(x + (i - 0.5) * width, sub["rul_rmse"], width, color=colors.get(ds, None), label=ds)
    ax.set_xticks(x); ax.set_xticklabels(variants, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("RUL RMSE (cycles)")
    ax.set_title("(c) Downstream RUL")
    ax.legend(fontsize=8)
    fig.suptitle("Extended ablation: bounded geometry controls boundedness; "
                 "penalties control forecastability/RUL", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(fds=DATASETS):
    rows = []
    for fd in fds:
        print("=" * 74)
        print(f"EXTENDED ABLATION  FD00{fd}  (epochs={EPOCHS})")
        print("=" * 74)
        for spec in VARIANTS:
            rows.append(run_variant(fd, *spec))
    df = pd.DataFrame(rows)
    csv = os.path.join(ac.ACML_TAB, "ablation_extended.csv")
    df.to_csv(csv, index=False)

    tex_cols = ["dataset", "variant", "recon_mean_r2", "kappa",
                "freerun_growth", "freerun_bounded", "cv_skill_k20", "rul_rmse"]
    tex = ac.latex_table(df[tex_cols], "Extended ablation study. Bounded latent "
                         "geometry controls free-run boundedness; monotonicity "
                         "and smoothness reduce curvature and improve RUL.",
                         "tab:ablation_extended")
    with open(os.path.join(ac.ACML_TAB, "ablation_extended.tex"), "w", encoding="utf-8") as fh:
        fh.write(tex)

    fig_path = os.path.join(ac.ACML_FIG, "ablation_extended_summary.png")
    make_figure(df, fig_path)

    print("\n" + "=" * 74)
    print(df[tex_cols].to_string(index=False))
    print(f"\nsaved -> {csv}")
    print(f"saved -> {fig_path}")
    return df


if __name__ == "__main__":
    fds = [int(x) for x in sys.argv[1:]] or DATASETS
    main(fds)
