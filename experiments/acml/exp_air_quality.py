"""
Air-quality (Beijing) — CORE-contribution test: stable bounded rollout.

Objectives differ from the degradation datasets: this is a forecasting problem,
so we DO NOT evaluate RUL or monotone health. We test only the method's core
contributions that are meaningful here:

  C1. Reconstruction: a compact bounded latent captures the 11-variable state.
  C2. Stable rollout: bounded-latent free-run stays bounded while a sensor-space
      VAR can diverge (rho>1).
  C3. Forecastability: multi-step latent forecasts beat a persistence baseline.
  C4. K as regularizer: reconstruction rises with K while stability holds.

Because the series are (quasi-)stationary and cyclic rather than monotone, the
autoencoder is trained WITHOUT the monotonicity penalty (lambda_mono=0); the
bounded (sigmoid) latent and a light smoothness penalty are retained since those
are what deliver the stability guarantee.

Reuses the dataset-agnostic helpers from exp_ims_bearing.

Outputs
-------
  results/acml/tables/air_quality_summary.csv (+ .tex)
  results/acml/tables/air_quality_k_sweep.csv (+ .tex)
  results/acml/figures/air_quality_summary.png
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

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))

import exp_ims_bearing as X  # dataset-agnostic building blocks

TAB = os.path.join(ROOT, "results", "acml", "tables")
FIG = os.path.join(ROOT, "results", "acml", "figures")
DEFAULT_FEAT = os.path.join(ROOT, "data", "processed", "air_quality_features.csv")

FREE_STEPS = X.FREE_STEPS
CUTOFF_FRAC = X.CUTOFF_FRAC
BOUNDED_GROWTH_THRESH = X.BOUNDED_GROWTH_THRESH


def forecast_skill(ae, te: pd.DataFrame, cols: list[str], horizons: list[int]) -> dict:
    """Multi-step decoded-forecast skill vs persistence, pooled over test units.

    For each unit and several anchor points, roll the latent forward `h` steps
    (constant-velocity + bounded projection), decode, and compare RMSE against a
    persistence forecast (last observed value held constant). Skill =
    1 - MSE_model / MSE_persistence, per horizon (informative-variable pooled).
    """
    sig = te[cols].std().to_numpy() + 1e-9
    acc = {h: {"m": [], "p": []} for h in horizons}
    for _, g in te.groupby("unit_id"):
        g = g.sort_values("cycle").reset_index(drop=True)
        n = len(g)
        truth = g[cols].to_numpy()
        anchors = [int(f * n) for f in (0.5, 0.65, 0.8)]
        for c0 in anchors:
            if c0 < X.ROLLOUT_VEL_WINDOW + 2 or c0 >= n - max(horizons) - 1:
                continue
            h_hist = X.encode(ae, g.iloc[:c0 + 1])
            _, roll = X.rollout(ae, h_hist, max(horizons) + 1, "full_box")
            for h in horizons:
                pred_m = roll[h - 1]
                pred_p = truth[c0]
                tgt = truth[c0 + h]
                acc[h]["m"].append(((pred_m - tgt) / sig) ** 2)
                acc[h]["p"].append(((pred_p - tgt) / sig) ** 2)
    out = {}
    for h in horizons:
        if len(acc[h]["m"]) >= 3:
            mse_m = np.mean(np.concatenate([a.ravel() for a in acc[h]["m"]]))
            mse_p = np.mean(np.concatenate([a.ravel() for a in acc[h]["p"]]))
            out[h] = float(1.0 - mse_m / (mse_p + 1e-12))
        else:
            out[h] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--features", type=str, default=DEFAULT_FEAT)
    ap.add_argument("--name", type=str, default="air_quality")
    ap.add_argument("--test-frac", type=float, default=0.34)
    ap.add_argument("--horizons", type=int, nargs="*", default=[1, 8, 24])
    args = ap.parse_args()
    name = args.name

    print(f"{name} — core-contribution test (stable rollout, no RUL)")
    print("=" * 66)
    df, feats = X.load_features(args.features)
    print(f"loaded {len(df)} rows, {df['unit_id'].nunique()} stations, "
          f"{len(feats)} variables")

    cols = X.select_trend_features(X.denoise(df, feats), feats)
    # for forecasting we keep all variables (trend filter may drop cyclic ones)
    if len(cols) < len(feats):
        cols = feats
    print(f"using {len(cols)} variables: {cols}")

    dfd = X.denoise(df, cols, win=5)
    tr, te = X.split_units(dfd, test_frac=args.test_frac)
    print(f"train stations: {tr['unit_id'].nunique()}  "
          f"test stations: {te['unit_id'].nunique()}")

    # --- main k=2 comparison; NOTE lambda_mono=0 (no monotone assumption) --- #
    print("\n[1/3] training k=2 bounded + unbounded autoencoders (no mono) ...")
    ae_b = X.train_ae(tr, cols, k=2, bounded=True, lambda_mono=0.0,
                      lambda_smooth=0.5, epochs=args.epochs)
    ae_u = X.train_ae(tr, cols, k=2, bounded=False, lambda_mono=0.0,
                      lambda_smooth=0.5, epochs=args.epochs)
    reg, vmu, vsd, rho = X.fit_var(tr, cols)

    rows = []
    for mname, ae, proj in [("manifold_bounded", ae_b, "full_box"),
                            ("ae_unbounded", ae_u, "none")]:
        rmean, rmin = X.recon_r2(ae, te)
        _, growth, bnd = X.decoded_freerun(ae, tr, te, proj)
        rows.append(dict(model=mname, rho=np.nan, recon_mean_r2=rmean,
                         recon_min_r2=rmin, freerun_growth=growth, bounded=bnd))
    _, vgrowth, vbnd = X.var_freerun(reg, vmu, vsd, te, cols)
    rows.append(dict(model="var_sensor", rho=rho, recon_mean_r2=np.nan,
                     recon_min_r2=np.nan, freerun_growth=vgrowth, bounded=vbnd))
    summary = pd.DataFrame(rows)

    print("[2/3] forecast skill vs persistence ...")
    sk = forecast_skill(ae_b, te, cols, args.horizons)
    for h in args.horizons:
        summary.loc[summary.model == "manifold_bounded", f"skill_h{h}"] = sk[h]

    summary.to_csv(os.path.join(TAB, f"{name}_summary.csv"), index=False)
    with open(os.path.join(TAB, f"{name}_summary.tex"), "w") as fh:
        fh.write(X.latex_table(summary.round(4),
                 f"{name}: core-contribution test — bounded latent vs unbounded "
                 "AE vs sensor VAR (stability + forecast skill; no RUL).",
                 f"tab:{name}"))

    print("[3/3] K-aware sweep (recon + stability + 24-step skill) ...")
    krows = []
    hmax = max(args.horizons)
    for k in range(1, args.kmax + 1):
        ae_k = X.train_ae(tr, cols, k=k, bounded=True, lambda_mono=0.0,
                          lambda_smooth=0.5, epochs=args.epochs)
        rmean, rmin = X.recon_r2(ae_k, te)
        _, growth, bnd = X.decoded_freerun(ae_k, tr, te, "full_box")
        skk = forecast_skill(ae_k, te, cols, [hmax])[hmax]
        krows.append(dict(k=k, recon_mean_r2=rmean, recon_min_r2=rmin,
                          freerun_growth=growth, freerun_bounded=bnd,
                          skill_hmax=skk))
        print(f"    k={k}: recon={rmean:.3f} growth={growth:.2f} "
              f"bounded={bnd} skill_h{hmax}={skk:.3f}")
    ksweep = pd.DataFrame(krows)
    ksweep.to_csv(os.path.join(TAB, f"{name}_k_sweep.csv"), index=False)
    with open(os.path.join(TAB, f"{name}_k_sweep.tex"), "w") as fh:
        fh.write(X.latex_table(ksweep.round(4),
                 f"{name} latent-dimension sweep: reconstruction and forecast "
                 "skill vs stability.", f"tab:{name}_ksweep"))

    make_figure(ae_b, tr, te, cols, reg, vmu, vsd, ksweep, sk, args.horizons, name)

    print("\n" + "=" * 66)
    print(f"CORE-CONTRIBUTION REPRODUCTION ON {name.upper()}")
    mb = summary[summary.model == "manifold_bounded"].iloc[0]
    vv = summary[summary.model == "var_sensor"].iloc[0]
    print(f"  C1 recon R2 (bounded k=2) : {mb.recon_mean_r2:.3f}")
    print(f"  C2 bounded growth={mb.freerun_growth:.2f} (bounded={mb.bounded}) "
          f"vs VAR rho={vv.rho:.3f} growth={vv.freerun_growth:.1f} "
          f"(bounded={vv.bounded})")
    print(f"  C3 forecast skill vs persistence: "
          f"{ {h: round(sk[h], 3) for h in args.horizons} }")
    print(f"  C4 recon(k=1..K)          : "
          f"{ksweep['recon_mean_r2'].round(3).tolist()}")
    print("=" * 66)
    print(f"wrote tables -> {TAB}")
    print(f"wrote figure -> {os.path.join(FIG, f'{name}_summary.png')}")


def make_figure(ae, tr, te, cols, reg, vmu, vsd, ksweep, sk, horizons, name):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    # (a) example decoded 1-step reconstruction of PM2.5 for one station
    g = X.longest_unit(te)
    var = "PM2.5" if "PM2.5" in cols else cols[0]
    j = cols.index(var)
    recon = X.decode(ae, X.encode(ae, g))
    ax[0, 0].plot(g["cycle"][:500], g[var].to_numpy()[:500], label="true", lw=1)
    ax[0, 0].plot(g["cycle"][:500], recon[:500, j], label="reconstructed", lw=1)
    ax[0, 0].legend(); ax[0, 0].set_title(f"(a) {var} reconstruction (test)")
    ax[0, 0].set_xlabel("hour index")

    # (b) bounded latent vs VAR free-run norm
    c0 = int(CUTOFF_FRAC * len(g))
    h_hist = X.encode(ae, g.iloc[:c0 + 1])
    _, roll = X.rollout(ae, h_hist, FREE_STEPS, "full_box")
    mu = tr[cols].mean().to_numpy(); sd = tr[cols].std().to_numpy() + 1e-12
    man_norm = [np.linalg.norm((roll[s] - mu) / sd) for s in range(FREE_STEPS)]
    z = (g[cols].to_numpy()[c0] - vmu) / vsd
    var_norm = []
    for _ in range(FREE_STEPS):
        z = reg.predict(z.reshape(1, -1))[0]; var_norm.append(np.linalg.norm(z))
    ax[0, 1].plot(man_norm, label="manifold (bounded)")
    ax[0, 1].plot(var_norm, label="sensor VAR")
    ax[0, 1].set_yscale("log"); ax[0, 1].legend()
    ax[0, 1].set_title("(b) Free-run state norm (log)")
    ax[0, 1].set_xlabel("rollout step")

    # (c) recon vs K + growth
    ax[1, 0].plot(ksweep["k"], ksweep["recon_mean_r2"], "o-", label="recon R2")
    ax[1, 0].set_ylabel("recon R2"); ax[1, 0].set_xlabel("K")
    axb = ax[1, 0].twinx()
    axb.plot(ksweep["k"], ksweep["freerun_growth"], "s--", color="tab:red")
    axb.axhline(BOUNDED_GROWTH_THRESH, color="gray", ls=":")
    axb.set_ylabel("free-run growth")
    ax[1, 0].set_title("(c) Reconstruction vs stability across K")

    # (d) forecast skill vs horizon
    ax[1, 1].plot(list(horizons), [sk[h] for h in horizons], "o-")
    ax[1, 1].axhline(0.0, color="gray", ls=":")
    ax[1, 1].set_title("(d) Forecast skill vs persistence")
    ax[1, 1].set_xlabel("horizon (steps)"); ax[1, 1].set_ylabel("skill")

    fig.suptitle(f"{name}: core-contribution (stable rollout) test")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{name}_summary.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
