"""
RESEARCH EXPERIMENT E3-ROLLOUT -- ABLATION OF MONOTONICITY / SMOOTHNESS
                                   MEASURED ON ROLLOUT STABILITY
======================================================================

The monotonicity and smoothness penalties were introduced for ROLLOUT
stability, not RUL.  The plain ablation (exp_ablation.py) showed they barely
move RUL RMSE -- which is expected, because a supervised RUL head can read a
noisy latent.  Their real job is to make the latent a *smooth, monotone,
forecastable* trajectory so that velocity-extrapolation rollout stays bounded
and accurate.  This experiment measures exactly that.

For each dataset and each variant {full, no_mono, no_smooth, no_mono_smooth}
we retrain the manifold and measure rollout-stability diagnostics:

  manifold_nrmse_max   max scored manifold-rollout NRMSE over horizons (lower=better)
  man_freerun_norm     decoded state norm after 400 free-run steps (bounded check)
  man_freerun_growth   ratio norm_400 / norm_0  (~1 => bounded)
  kappa                median |2nd diff h0|  -- latent curvature (Theorem 3 envelope)
  mono_viol_frac       fraction of dh0<0 on the denoised trajectories
  latent_step_std      std of dh0 -- latent jitter (smoothness proxy)
  cv_skill_k20         constant-velocity forecast skill of h0 vs persistence @ k=20
  rho_var              sensor-VAR spectral radius (context reference, variant-independent)

Usage:  python exp_ablation_rollout.py [fd ...]      (default: 1 2 3 4)

Outputs
  results/tables/research/ablation/rollout_ablation_summary.csv
  results/figures/research/ablation/rollout_ablation_FD00<fd>.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import manifold as mc
import manifold.train as mtrain
from exp_rollout_stability import (CUTOFF_FRAC, VEL_WINDOW, fit_sensor_var,
                                   free_run, rollout_manifold, rollout_var)

ABLATION_EPOCHS = 2000
MAX_H = 150
SKILL_K = 20

VARIANTS = {
    "full":           dict(),
    "no_mono":        dict(lambda_mono=0.0),
    "no_smooth":      dict(lambda_smooth=0.0),
    "no_mono_smooth": dict(lambda_mono=0.0, lambda_smooth=0.0),
}


# --------------------------------------------------------------------------- #
# latent-trajectory diagnostics
# --------------------------------------------------------------------------- #
def latent_diagnostics(man, te_den) -> dict:
    H = mc.per_engine_health(man, te_den)
    kappas, viols, total, steps = [], 0, 0, []
    for _, g in H.sort_values("cycle").groupby("unit_id"):
        h0 = g["h0"].to_numpy()
        if len(h0) < 3:
            continue
        d1 = np.diff(h0)
        d2 = np.diff(h0, 2)
        kappas.append(np.median(np.abs(d2)))
        viols += int((d1 < 0).sum())
        total += len(d1)
        steps.append(d1)
    step = np.concatenate(steps) if steps else np.array([0.0])
    return dict(kappa=float(np.median(kappas)) if kappas else float("nan"),
                mono_viol_frac=float(viols / max(total, 1)),
                latent_step_std=float(step.std()))


def manifold_rollout_nrmse_max(man, te_den) -> float:
    inf_idx = [mc.DYNAMIC.index(s) for s in mc.INFORMATIVE]
    sigma = te_den[mc.INFORMATIVE].std().to_numpy() + 1e-9
    per_h = {h: {"T": [], "M": []} for h in range(1, MAX_H + 1)}
    for _, g in te_den.groupby("unit_id"):
        g = g.sort_values("cycle").reset_index(drop=True)
        n = len(g)
        c0 = int(CUTOFF_FRAC * n)
        if c0 < VEL_WINDOW + 2 or c0 >= n - 5:
            continue
        steps = min(n - 1 - c0, MAX_H)
        truth = g[mc.DYNAMIC].to_numpy()
        h_hist = man.encode(g.iloc[:c0 + 1])
        man_roll = rollout_manifold(man, h_hist, steps)
        for s in range(steps):
            per_h[s + 1]["T"].append(truth[c0 + s + 1, inf_idx])
            per_h[s + 1]["M"].append(man_roll[s, inf_idx])
    nrmse = []
    for h in range(1, MAX_H + 1):
        if len(per_h[h]["T"]) < 5:
            continue
        T = np.array(per_h[h]["T"]) / sigma
        M = np.array(per_h[h]["M"]) / sigma
        nrmse.append(float(np.sqrt(np.mean((T - M) ** 2))))
    return float(np.max(nrmse)) if nrmse else float("nan")


def cv_skill(man, te_den, k: int = SKILL_K) -> float:
    """Constant-velocity forecast skill of h0 vs persistence at horizon k,
    pooled over engines, launched from 50/65/80% life."""
    H = mc.per_engine_health(man, te_den)
    se_cv, se_pe = [], []
    for _, g in H.sort_values("cycle").groupby("unit_id"):
        y = g["h0"].to_numpy()
        n = len(y)
        for frac in (0.5, 0.65, 0.8):
            c0 = int(frac * n)
            if c0 < VEL_WINDOW + 1 or c0 + k >= n:
                continue
            t = np.arange(max(0, c0 - VEL_WINDOW), c0)
            seg = y[max(0, c0 - VEL_WINDOW):c0]
            v = np.polyfit(t, seg, 1)[0] if len(seg) >= 2 else 0.0
            pred_cv = y[c0 - 1] + v * k
            pred_pe = y[c0 - 1]
            truth = y[c0 + k - 1]
            se_cv.append((pred_cv - truth) ** 2)
            se_pe.append((pred_pe - truth) ** 2)
    if not se_cv:
        return float("nan")
    return float(1.0 - np.mean(se_cv) / (np.mean(se_pe) + 1e-12))


def run_variant(fd: int, name: str, overrides: dict) -> dict:
    mc.configure(fd, retrain=True, k=2, tag=f"abl_{name}", **overrides)
    man = mc.get_manifold()

    df = mc.load_split("train")
    tr, te = mc.split_by_unit(df)
    tr_den = mc.denoise(tr)
    te_den = mc.denoise(te)

    var = fit_sensor_var(tr_den)
    lat = latent_diagnostics(man, te_den)
    nrmse_max = manifold_rollout_nrmse_max(man, te_den)
    skill = cv_skill(man, te_den)

    # free-run on the longest test engine
    longest = max(te_den["unit_id"].unique(),
                  key=lambda u: (te_den["unit_id"] == u).sum())
    g_long = te_den[te_den["unit_id"] == longest].sort_values("cycle").reset_index(drop=True)
    c0_long = int(CUTOFF_FRAC * len(g_long))
    var_norm, man_norm = free_run(var, man, g_long, c0_long, steps=400)

    row = dict(dataset=f"FD00{fd}", variant=name,
               manifold_nrmse_max=nrmse_max,
               man_freerun_norm=float(man_norm[-1]),
               man_freerun_growth=float(man_norm[-1] / (man_norm[0] + 1e-12)),
               kappa=lat["kappa"], mono_viol_frac=lat["mono_viol_frac"],
               latent_step_std=lat["latent_step_std"],
               cv_skill_k20=skill, rho_var=float(var["rho"]))
    print(f"  FD00{fd} {name:<15} nrmse_max={nrmse_max:5.2f}  "
          f"freerun={row['man_freerun_norm']:6.2f}  kappa={lat['kappa']:.2e}  "
          f"mono_viol={lat['mono_viol_frac']:.3f}  skill@20={skill:+.3f}")
    return row


def _plot(df_fd, fd, out_dir):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    v = df_fd["variant"].tolist()
    x = np.arange(len(v))
    ax[0].bar(x, df_fd["manifold_nrmse_max"], color="#4c72b0")
    ax[0].set_title("(a) max manifold-rollout NRMSE")
    ax[0].set_ylabel("NRMSE (std units)")
    ax[1].bar(x, df_fd["kappa"], color="#dd8452")
    ax[1].set_title(r"(b) latent curvature $\kappa$ (smoothness)")
    ax[1].set_ylabel(r"median $|\Delta^2 h_0|$")
    ax[2].bar(x, df_fd["cv_skill_k20"], color="#55a868")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_title("(c) const-velocity skill @ k=20")
    ax[2].set_ylabel("skill vs persistence")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(v, rotation=20, ha="right", fontsize=9)
        a.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Rollout-stability ablation -- FD00{fd}", fontsize=13)
    fig.tight_layout()
    p = os.path.join(out_dir, f"rollout_ablation_FD00{fd}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(fds):
    mtrain.EPOCHS = ABLATION_EPOCHS
    tab_dir = os.path.join(ROOT, "results", "tables", "research", "ablation")
    fig_dir = os.path.join(ROOT, "results", "figures", "research", "ablation")
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    rows = []
    for fd in fds:
        print("=" * 78)
        print(f"ROLLOUT-STABILITY ABLATION  FD00{fd}  (epochs={ABLATION_EPOCHS})")
        print("=" * 78)
        sub = []
        for name, ov in VARIANTS.items():
            r = run_variant(fd, name, ov)
            rows.append(r)
            sub.append(r)
        _plot(pd.DataFrame(sub), fd, fig_dir)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(tab_dir, "rollout_ablation_summary.csv"), index=False)
    print("\n" + "=" * 78)
    print("ROLLOUT-STABILITY ABLATION SUMMARY")
    print("=" * 78)
    cols = ["dataset", "variant", "manifold_nrmse_max", "man_freerun_norm",
            "kappa", "mono_viol_frac", "latent_step_std", "cv_skill_k20"]
    print(df[cols].to_string(index=False))
    print(f"\nsaved -> {os.path.join(tab_dir, 'rollout_ablation_summary.csv')}")
    return df


if __name__ == "__main__":
    fds = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 4]
    main(fds)
