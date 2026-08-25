"""
ACML — Full-budget hyperparameter sensitivity analysis.

Addresses reviewer point (2): "lack of hyperparameter sensitivity analysis due
to reduced epoch budget." This script runs the *production* epoch budget and
sweeps each core hyperparameter one-at-a-time around the validated defaults,
reporting the full metric surface with multiple seeds (mean +/- std).

Core hyperparameters (production defaults in parentheses)
---------------------------------------------------------
  lambda_mono   (5.0)  monotonicity penalty on the primary latent coordinate
  lambda_smooth (2.0)  smoothness penalty on the primary latent coordinate
  gamma == alpha(0.05) residual step size of the bounded latent dynamics:
                       h_{t+1} = Pi_[0,1]^k( h_t + alpha * tanh(g_psi([h_t,c_t])) )
  regime_count  (auto) number of KMeans operating-condition clusters used for
                       per-regime sensor normalisation (multi-regime datasets)

Representative datasets: FD001 (single-regime) + FD002 (multi-regime).

Metrics
-------
  recon_mean_r2, recon_min_r2  reconstruction fidelity (informative sensors)
  kappa                        latent curvature (forecastability proxy; lower=better)
  mono_viol_frac               fraction of non-monotone steps in health coord
  freerun_growth, bounded      long-horizon latent rollout stability
  cv_skill_k20                 constant-velocity forecast skill vs persistence
  rul_rmse, rul_r2             downstream RUL utility (k-aware features)
  For the gamma sweep additionally:
  dyn_skill_h10/h25/h50        learned-dynamics rollout skill vs persistence

Outputs (results/acml/sensitivity/)
-----------------------------------
  sensitivity_lambda_mono.csv    (+ .tex)
  sensitivity_lambda_smooth.csv  (+ .tex)
  sensitivity_gamma.csv          (+ .tex)
  sensitivity_regime_count.csv   (+ .tex)
  sensitivity_summary.png

Usage
-----
  # full budget (production epochs, 3 seeds) -- what the reviewer asked for
  python experiments/acml/exp_hparam_sensitivity.py

  # quick smoke test
  python experiments/acml/exp_hparam_sensitivity.py --epochs-ae 150 \
      --epochs-dyn 80 --seeds 0 --datasets 1
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import acml_common as ac          # full C-MAPSS manifold pipeline + metrics
import latent_dynamics as LD      # learned residual dynamics (for gamma sweep)
import manifold as mc             # core context (configure / load / denoise)
from manifold.state import require_cfg, set_cfg
from manifold.config import ALL_SENSORS, SETTINGS, SEED
from manifold.data import _read, load_split, split_by_unit
from manifold.denoise import denoise as mc_denoise

OUT_DIR = os.path.join(ac.ROOT, "results", "acml", "sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)

# Production defaults (the validated operating point every sweep pivots around)
DEF_K = 2
DEF_MONO = 5.0
DEF_SMOOTH = 2.0
DEF_ALPHA = 0.05

# Sweep grids
GRID_MONO = [0.0, 1.0, 5.0, 10.0]
GRID_SMOOTH = [0.0, 0.5, 2.0, 5.0]
GRID_GAMMA = [0.02, 0.05, 0.10, 0.25]
GRID_REGIMES = [1, 2, 3, 6]

DYN_HORIZONS = [10, 25, 50]


# --------------------------------------------------------------------------- #
# Regime-count forcing (no core modification: override the active context)
# --------------------------------------------------------------------------- #
def setup_forced_regimes(fd: int, n_reg: int, k: int = DEF_K):
    """Configure FD00<fd> then force exactly `n_reg` normalisation regimes.

    Re-fits KMeans on the operating settings with the requested cluster count,
    recomputes per-regime means / pooled residual std, and reloads denoised
    splits. The dynamic/informative sensor set from the initial configure() is
    kept fixed so the sweep isolates the effect of regime count alone.
    """
    mc.configure(fd, k=k, normalize=True, regime_rule="heuristic",
                 tag=f"sens_reg{n_reg}_k{k}")
    cfg = require_cfg()

    tr = _read("train", fd)
    settings = tr[SETTINGS].to_numpy()
    km = KMeans(n_clusters=n_reg, n_init=10, random_state=SEED).fit(settings)
    reg = km.predict(settings)

    x = tr[ALL_SENSORS].to_numpy().astype(float)
    reg_mean = np.zeros((n_reg, len(ALL_SENSORS)))
    for r in range(n_reg):
        m = reg == r
        reg_mean[r] = x[m].mean(0) if m.any() else 0.0
    resid = x - reg_mean[reg]
    resid_std = np.maximum(resid.std(0), 1e-9)

    cfg.n_regimes = n_reg
    cfg.km = km
    cfg.reg_mean = reg_mean
    cfg.resid_std = resid_std
    set_cfg(cfg)

    df = load_split("train")            # regime column assigned from cfg.km
    tr_split, te_split = split_by_unit(df)
    return mc_denoise(tr_split), mc_denoise(te_split)


# --------------------------------------------------------------------------- #
# Learned-dynamics rollout skill (for the gamma / alpha sweep)
# --------------------------------------------------------------------------- #
def _latent_rollout_skill(res, trajs, horizons):
    """Skill vs persistence of the learned dynamics on the primary coord h0."""
    anchors = (0.5, 0.65, 0.8)
    hmax = max(horizons)
    mse_m = {h: 0.0 for h in horizons}
    mse_p = {h: 0.0 for h in horizons}
    n = {h: 0 for h in horizons}
    for tr in trajs:
        T = len(tr.h)
        for f in anchors:
            c0 = int(f * T)
            if c0 < 2 or c0 + hmax >= T:
                continue
            c_future = tr.c[c0:c0 + hmax] if tr.c.shape[1] else np.zeros((hmax, 0), np.float32)
            roll = LD.dyn_rollout(res, tr.h[c0], c_future, hmax)  # (hmax, k)
            for h in horizons:
                pred = roll[h - 1, 0]
                pers = tr.h[c0, 0]
                tgt = tr.h[c0 + h, 0]
                mse_m[h] += (pred - tgt) ** 2
                mse_p[h] += (pers - tgt) ** 2
                n[h] += 1
    out = {}
    for h in horizons:
        if n[h] == 0 or mse_p[h] == 0:
            out[h] = float("nan")
        else:
            out[h] = float(1.0 - mse_m[h] / mse_p[h])
    return out


# --------------------------------------------------------------------------- #
# Single-config evaluation
# --------------------------------------------------------------------------- #
def eval_ae_config(tr_den, te_den, *, k, mono, smooth, seed, epochs_ae):
    """Train a bounded AE at the given penalties and return the metric surface."""
    man = ac.train_flex_ae(tr_den, k=k, bounded=True, lambda_mono=mono,
                           lambda_smooth=smooth, seed=seed, epochs=epochs_ae)
    mean_r2, min_r2 = ac.recon_r2(man, te_den)
    kappa = ac.curvature_kappa(man, te_den)
    mono_v = ac.mono_violation_fraction(man, te_den)
    _, growth, bnd = ac.freerun_growth(man, tr_den, te_den, "full_box")
    skill = ac.forecast_skill_cv(man, te_den, horizon=20)
    rul = ac.rul_metrics_kaware(man, seed=seed)
    return man, dict(recon_mean_r2=mean_r2, recon_min_r2=min_r2, kappa=kappa,
                     mono_viol_frac=mono_v, freerun_growth=growth,
                     bounded=bool(bnd), cv_skill_k20=skill,
                     rul_rmse=rul["rul_rmse"], rul_r2=rul["rul_r2"])


# --------------------------------------------------------------------------- #
# Aggregation over seeds -> mean/std table
# --------------------------------------------------------------------------- #
def aggregate(rows, group_keys):
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns
                   if c not in group_keys + ["seed", "bounded"]]
    agg = {c: ["mean", "std"] for c in metric_cols}
    if "bounded" in df.columns:
        agg["bounded"] = "mean"
    g = df.groupby(group_keys).agg(agg)
    g.columns = ["_".join(c).rstrip("_") for c in g.columns]
    return g.reset_index()


def write_tex(df, path, caption, label, fmt="%.3f"):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(ac.latex_table(df.round(4), caption, label, float_fmt=fmt))


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #
def sweep_lambda(datasets, seeds, epochs_ae, which):
    """which = 'mono' or 'smooth'. Sweeps that penalty, holds the other at default."""
    grid = GRID_MONO if which == "mono" else GRID_SMOOTH
    rows = []
    for fd in datasets:
        tr_den, te_den = ac.setup_dataset(fd, k=DEF_K, normalize=True)
        for val in grid:
            mono = val if which == "mono" else DEF_MONO
            smooth = val if which == "smooth" else DEF_SMOOTH
            for seed in seeds:
                _, m = eval_ae_config(tr_den, te_den, k=DEF_K, mono=mono,
                                      smooth=smooth, seed=seed, epochs_ae=epochs_ae)
                row = dict(dataset=f"FD00{fd}", lambda_mono=mono,
                           lambda_smooth=smooth, seed=seed, **m)
                rows.append(row)
                print(f"  [{which}] FD00{fd} mono={mono} smooth={smooth} "
                      f"seed={seed}: recon={m['recon_mean_r2']:.3f} "
                      f"kappa={m['kappa']:.2e} skill@20={m['cv_skill_k20']:+.3f} "
                      f"RUL={m['rul_rmse']:.2f}")
    key = "lambda_mono" if which == "mono" else "lambda_smooth"
    return rows, ["dataset", key]


def sweep_gamma(datasets, seeds, epochs_ae, epochs_dyn):
    """Sweep the dynamics step size alpha (gamma). AE held at production defaults."""
    rows = []
    for fd in datasets:
        tr_den, te_den = ac.setup_dataset(fd, k=DEF_K, normalize=True)
        for seed in seeds:
            man, _ = eval_ae_config(tr_den, te_den, k=DEF_K, mono=DEF_MONO,
                                    smooth=DEF_SMOOTH, seed=seed, epochs_ae=epochs_ae)
            feats = man.dynamic
            enc = lambda g: man.encode(g)
            tr_trajs = LD.build_trajectories(enc, tr_den, feats, [], man.mu, man.sd)
            te_trajs = LD.build_trajectories(enc, te_den, feats, [], man.mu, man.sd)
            for gamma in GRID_GAMMA:
                res = LD.train_dynamics(tr_trajs, DEF_K, 0, use_context=False,
                                        projection="soft", multistep=True,
                                        hidden=64, alpha=gamma, horizon=16,
                                        epochs=epochs_dyn, seed=seed)
                sk = _latent_rollout_skill(res, te_trajs, DYN_HORIZONS)
                row = dict(dataset=f"FD00{fd}", gamma=gamma, seed=seed,
                           dyn_skill_h10=sk[10], dyn_skill_h25=sk[25],
                           dyn_skill_h50=sk[50], final_loss=res.final_loss)
                rows.append(row)
                print(f"  [gamma] FD00{fd} alpha={gamma} seed={seed}: "
                      f"skill@10={sk[10]:+.3f} @25={sk[25]:+.3f} @50={sk[50]:+.3f}")
    return rows, ["dataset", "gamma"]


def sweep_regimes(datasets, seeds, epochs_ae):
    """Sweep forced regime count. Only meaningful for multi-regime datasets."""
    rows = []
    for fd in datasets:
        for n_reg in GRID_REGIMES:
            tr_den, te_den = setup_forced_regimes(fd, n_reg, k=DEF_K)
            for seed in seeds:
                _, m = eval_ae_config(tr_den, te_den, k=DEF_K, mono=DEF_MONO,
                                      smooth=DEF_SMOOTH, seed=seed, epochs_ae=epochs_ae)
                row = dict(dataset=f"FD00{fd}", regime_count=n_reg, seed=seed, **m)
                rows.append(row)
                print(f"  [regimes] FD00{fd} n_reg={n_reg} seed={seed}: "
                      f"recon={m['recon_mean_r2']:.3f} kappa={m['kappa']:.2e} "
                      f"skill@20={m['cv_skill_k20']:+.3f} RUL={m['rul_rmse']:.2f}")
    return rows, ["dataset", "regime_count"]


# --------------------------------------------------------------------------- #
# Summary figure
# --------------------------------------------------------------------------- #
def make_summary_figure(tables: dict, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    panels = [
        ("lambda_mono", "lambda_mono", "recon_mean_r2_mean", "Recon $R^2$", "(a) $\\lambda_{mono}$"),
        ("lambda_smooth", "lambda_smooth", "recon_mean_r2_mean", "Recon $R^2$", "(b) $\\lambda_{smooth}$"),
        ("gamma", "gamma", "dyn_skill_h50_mean", "Skill@50", "(c) $\\gamma$ (step size)"),
        ("regime_count", "regime_count", "cv_skill_k20_mean", "Skill@20", "(d) regime count"),
    ]
    for ax, (tkey, xcol, ycol, ylabel, title) in zip(axes, panels):
        df = tables.get(tkey)
        if df is None or xcol not in df or ycol not in df:
            ax.set_visible(False)
            continue
        for ds in df["dataset"].unique():
            sub = df[df.dataset == ds].sort_values(xcol)
            std_col = ycol.replace("_mean", "_std")
            yerr = sub[std_col] if std_col in sub else None
            ax.errorbar(sub[xcol], sub[ycol], yerr=yerr, marker="o",
                        capsize=3, label=ds)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Hyperparameter sensitivity (full budget, mean $\\pm$ std over seeds)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  summary figure -> {out_path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=int, nargs="*", default=[1, 2],
                    help="C-MAPSS FD sub-datasets (default: 1 2)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs-ae", type=int, default=4000,
                    help="AE epochs (production budget = 4000)")
    ap.add_argument("--epochs-dyn", type=int, default=400,
                    help="dynamics epochs for the gamma sweep")
    ap.add_argument("--skip", type=str, nargs="*", default=[],
                    help="sweeps to skip: mono smooth gamma regimes")
    args = ap.parse_args()

    print("Full-budget hyperparameter sensitivity analysis")
    print(f"Datasets: {['FD00%d' % d for d in args.datasets]}  seeds={args.seeds}")
    print(f"AE epochs={args.epochs_ae}  dyn epochs={args.epochs_dyn}")
    print("=" * 70)

    tables = {}

    if "mono" not in args.skip:
        print("\n[1/4] lambda_mono sweep ...")
        rows, keys = sweep_lambda(args.datasets, args.seeds, args.epochs_ae, "mono")
        agg = aggregate(rows, keys)
        agg.to_csv(os.path.join(OUT_DIR, "sensitivity_lambda_mono.csv"), index=False)
        write_tex(agg, os.path.join(OUT_DIR, "sensitivity_lambda_mono.tex"),
                  "Sensitivity to the monotonicity penalty $\\lambda_{mono}$ "
                  "(full budget, mean over seeds).", "tab:sens_lambda_mono")
        tables["lambda_mono"] = agg

    if "smooth" not in args.skip:
        print("\n[2/4] lambda_smooth sweep ...")
        rows, keys = sweep_lambda(args.datasets, args.seeds, args.epochs_ae, "smooth")
        agg = aggregate(rows, keys)
        agg.to_csv(os.path.join(OUT_DIR, "sensitivity_lambda_smooth.csv"), index=False)
        write_tex(agg, os.path.join(OUT_DIR, "sensitivity_lambda_smooth.tex"),
                  "Sensitivity to the smoothness penalty $\\lambda_{smooth}$ "
                  "(full budget, mean over seeds).", "tab:sens_lambda_smooth")
        tables["lambda_smooth"] = agg

    if "gamma" not in args.skip:
        print("\n[3/4] gamma (dynamics step size) sweep ...")
        rows, keys = sweep_gamma(args.datasets, args.seeds, args.epochs_ae, args.epochs_dyn)
        agg = aggregate(rows, keys)
        agg.to_csv(os.path.join(OUT_DIR, "sensitivity_gamma.csv"), index=False)
        write_tex(agg, os.path.join(OUT_DIR, "sensitivity_gamma.tex"),
                  "Sensitivity to the dynamics step size $\\gamma$ ($\\alpha$) "
                  "(full budget, mean over seeds).", "tab:sens_gamma")
        tables["gamma"] = agg

    if "regimes" not in args.skip:
        print("\n[4/4] regime-count sweep ...")
        rows, keys = sweep_regimes(args.datasets, args.seeds, args.epochs_ae)
        agg = aggregate(rows, keys)
        agg.to_csv(os.path.join(OUT_DIR, "sensitivity_regime_count.csv"), index=False)
        write_tex(agg, os.path.join(OUT_DIR, "sensitivity_regime_count.tex"),
                  "Sensitivity to the number of normalisation regimes "
                  "(full budget, mean over seeds).", "tab:sens_regimes")
        tables["regime_count"] = agg

    make_summary_figure(tables, os.path.join(OUT_DIR, "sensitivity_summary.png"))

    print("\n" + "=" * 70)
    print(f"DONE — outputs in {OUT_DIR}")
    for name in ["lambda_mono", "lambda_smooth", "gamma", "regime_count"]:
        p = os.path.join(OUT_DIR, f"sensitivity_{name}.csv")
        if os.path.exists(p):
            print(f"  sensitivity_{name}.csv (+ .tex)")
    print("  sensitivity_summary.png")


if __name__ == "__main__":
    main()
