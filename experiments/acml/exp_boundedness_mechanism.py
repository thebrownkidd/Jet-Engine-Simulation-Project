"""
ACML TASK 5 — Boundedness mechanism experiment.

Goal: empirically separate BOUNDEDNESS (an architectural property of the bounded
latent geometry) from FORECASTABILITY (improved by smoothness / monotonicity).

Five clearly-named configurations on FD001 (single-regime) + FD002 (multi-regime):

  1. unbounded_no_pen   unbounded latent AE, no penalties        proj="none"
  2. bounded_no_pen     bounded latent AE,   no penalties        proj="full_box"
  3. bounded_smooth     bounded latent AE,   smooth only         proj="full_box"
  4. bounded_mono       bounded latent AE,   monotonicity only   proj="full_box"
  5. full               bounded latent AE,   mono + smooth        proj="full_box"

Metrics: bounded_latent flag, mono/smooth penalty flags, free-run bounded flag,
free-run growth, curvature kappa, forecast skill, recon R2, RUL RMSE.

Expected: rows 2-5 (bounded latent) are free-run bounded regardless of penalties;
smoothness/monotonicity lower curvature kappa and raise forecast skill / RUL.

Outputs
  results/acml/tables/boundedness_mechanism.csv
  results/acml/tables/boundedness_mechanism.tex
  results/acml/figures/boundedness_mechanism.png
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

CONFIGS = [
    # name,               bounded, mono, smooth, projection
    ("unbounded_no_pen",   False, 0.0, 0.0, "none"),
    ("bounded_no_pen",     True,  0.0, 0.0, "full_box"),
    ("bounded_smooth",     True,  0.0, 2.0, "full_box"),
    ("bounded_mono",       True,  5.0, 0.0, "full_box"),
    ("full",               True,  5.0, 2.0, "full_box"),
]


def run_one(fd, name, bounded, mono, smooth, projection):
    tr_den, te_den = ac.setup_dataset(fd, k=2)
    man = ac.train_flex_ae(tr_den, k=2, bounded=bounded, lambda_mono=mono,
                           lambda_smooth=smooth, seed=SEED, epochs=EPOCHS)
    mean_r2, _ = ac.recon_r2(man, te_den)
    kappa = ac.curvature_kappa(man, te_den)
    skill = ac.forecast_skill_cv(man, te_den, horizon=20)
    fn, growth, bnd = ac.freerun_growth(man, tr_den, te_den, projection)
    lfn, lgrowth, lbnd = ac.latent_freerun_growth(man, te_den, projection)
    rul = ac.rul_metrics_kaware(man, seed=SEED)
    row = dict(dataset=f"FD00{fd}", config=name,
               bounded_latent=bounded, mono_penalty=(mono > 0),
               smooth_penalty=(smooth > 0), freerun_bounded=bnd,
               freerun_growth=growth, latent_growth=lgrowth,
               latent_bounded=lbnd, kappa=kappa, cv_skill_k20=skill,
               recon_mean_r2=mean_r2, rul_rmse=rul["rul_rmse"])
    print(f"  FD00{fd} {name:<17} bnd_latent={bounded!s:<5} "
          f"decoded_bnd={bnd!s:<5} latent_growth=x{lgrowth:7.2f} "
          f"latent_bnd={lbnd!s:<5} kappa={kappa:.2e} skill={skill:+.3f} "
          f"RUL={rul['rul_rmse']:.2f}")
    return row


def make_figure(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    configs = [c[0] for c in CONFIGS]
    x = np.arange(len(configs))
    width = 0.38
    colors = {"FD001": "#4c72b0", "FD002": "#dd8452"}
    # (a) free-run growth — boundedness
    ax = axes[0]
    for i, ds in enumerate(df["dataset"].unique()):
        sub = df[df.dataset == ds].set_index("config").loc[configs]
        ax.bar(x + (i - 0.5) * width, sub["latent_growth"], width,
               color=colors.get(ds), label=ds)
    ax.axhline(ac.BOUNDED_GROWTH_THRESH, color="r", ls="--", lw=1, label="bounded threshold")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(configs, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("latent-norm free-run growth (log)")
    ax.set_title("(a) Latent boundedness follows bounded geometry")
    ax.legend(fontsize=8)
    # (b) curvature kappa — forecastability
    ax = axes[1]
    for i, ds in enumerate(df["dataset"].unique()):
        sub = df[df.dataset == ds].set_index("config").loc[configs]
        ax.bar(x + (i - 0.5) * width, sub["kappa"], width,
               color=colors.get(ds), label=ds)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(configs, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("latent curvature kappa (log)")
    ax.set_title("(b) Penalties reduce curvature (forecastability)")
    ax.legend(fontsize=8)
    fig.suptitle("Mechanism: bounded geometry => boundedness; penalties => forecastability",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(fds=DATASETS):
    rows = []
    for fd in fds:
        print("=" * 74)
        print(f"BOUNDEDNESS MECHANISM  FD00{fd}  (epochs={EPOCHS})")
        print("=" * 74)
        for spec in CONFIGS:
            rows.append(run_one(fd, *spec))
    df = pd.DataFrame(rows)
    csv = os.path.join(ac.ACML_TAB, "boundedness_mechanism.csv")
    df.to_csv(csv, index=False)

    tex_cols = ["dataset", "config", "bounded_latent", "latent_bounded",
                "latent_growth", "freerun_bounded", "kappa", "cv_skill_k20", "rul_rmse"]
    tex = ac.latex_table(df[tex_cols], "Boundedness mechanism. Free-run "
                         "boundedness tracks the bounded latent geometry, not "
                         "the penalties; smoothness/monotonicity reduce "
                         "curvature and improve forecast skill / RUL.",
                         "tab:boundedness_mechanism")
    with open(os.path.join(ac.ACML_TAB, "boundedness_mechanism.tex"), "w", encoding="utf-8") as fh:
        fh.write(tex)

    fig_path = os.path.join(ac.ACML_FIG, "boundedness_mechanism.png")
    make_figure(df, fig_path)
    print("\n" + df[tex_cols].to_string(index=False))
    print(f"\nsaved -> {csv}\nsaved -> {fig_path}")
    return df


if __name__ == "__main__":
    fds = [int(x) for x in sys.argv[1:]] or DATASETS
    main(fds)
