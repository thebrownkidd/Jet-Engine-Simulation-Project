"""
EXPERIMENT C  --  Remaining Useful Life (RUL) from the forecastable health state
================================================================================

Skeptical question
-------------------
We proved (Exp A) that rollout in the 2-D health manifold is *bounded*, and
(Exp B) that the health coordinate h0 is *forecastable* (constant-velocity
extrapolation beats persistence by a wide margin).  The natural pay-off:

    "If the health state is identified and forecastable, can we predict RUL?"

A reviewer should demand two things:

  1. NO LEAKAGE.  The encoder may only see a causal (trailing-median) denoise
     of the sensors up to the truncation cycle of each *official* test engine.
  2. A REAL BASELINE.  Beating a constant 'predict the mean RUL' guess on both
     RMSE and the asymmetric NASA score (which punishes late predictions).

Two estimators are compared, and we are honest about why the naive one fails.

  (i)  threshold_crossing  -- the literal "forecast h0 forward until it hits a
       failure level" estimator.  We SHOW it is fragile: the autoencoder never
       had any incentive to scale h0 to a fixed range, so its dynamic range
       (~0.04) is comparable to its per-step noise.  Extrapolating a noisy
       slope to a fixed threshold blows up.

  (ii) health_to_rul  -- the robust estimator a PHM engineer would actually
       use: a supervised map RUL = f(h0, h1, v0, v1) where v0 is exactly the
       *forecast velocity* of the health coordinate.  It still relies entirely
       on the identified + forecastable health dynamics, but lets the data
       calibrate the (compressed, nonlinear) latent scale.  Targets use the
       standard FD001 piecewise-linear RUL cap.

Outputs
-------
    results/tables/FD00<fd>/rul_predictions.csv      per-engine: unit_id, rul_true, pred_*, ...
    results/tables/FD00<fd>/rul_metrics.json         RMSE / MAE / R2 / NASA for every estimator
    results/figures/C1_rul_scatter.png               predicted vs true (+ error histogram)
    results/figures/C2_examples.png                  example engines: causal h0 + RUL readout
    results/figures/C3_health_vs_rul.png             learned health->RUL relationship
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import manifold as mc

# ----------------------------- configuration ------------------------------ #
VEL_WINDOW = 25      # trailing cycles used to estimate the health velocity
RUL_CAP = 125        # standard FD001 piecewise-linear RUL ceiling
MIN_VEL = 1e-4       # below this the threshold estimator is considered "flat"
SEED = 42


# --------------------------- health-velocity ------------------------------ #
def _trailing_slope(y: np.ndarray, w: int) -> np.ndarray:
    """Causal least-squares slope of y over a trailing window of length <= w.

    slope_t = argmin_b sum_{j} (y_{t-..} - (a + b*tau))^2 ,  no future info.
    Vectorised with a sliding window; short prefixes use whatever is available.
    """
    n = len(y)
    out = np.zeros(n)
    for t in range(n):
        lo = max(0, t - w + 1)
        seg = y[lo:t + 1]
        m = len(seg)
        if m < 2:
            out[t] = 0.0
            continue
        tau = np.arange(m, dtype=float)
        tau -= tau.mean()
        denom = (tau * tau).sum()
        out[t] = float((tau * (seg - seg.mean())).sum() / denom) if denom > 0 else 0.0
    return out


def health_features(man: mc.Manifold, df_den: pd.DataFrame) -> pd.DataFrame:
    """Per-cycle causal health state + forecast velocity for every engine."""
    H = mc.per_engine_health(man, df_den)
    rows = []
    for uid, g in H.sort_values("cycle").groupby("unit_id"):
        g = g.copy()
        g["v0"] = _trailing_slope(g["h0"].to_numpy(), VEL_WINDOW)
        g["v1"] = _trailing_slope(g["h1"].to_numpy(), VEL_WINDOW)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


# ------------------------------- metrics ---------------------------------- #
def nasa_score(err: np.ndarray) -> float:
    """NASA C-MAPSS asymmetric score.  err = pred - true (late => err>0)."""
    err = np.asarray(err, float)
    s = np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)
    return float(s.sum())


def score(true: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    r2 = mc.r2_pooled(true, pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "NASA": nasa_score(err)}


# ============================= main pipeline ============================== #
def main(fd: int = 1) -> dict:
    mc.configure(fd)
    print("=" * 78)
    print(f"EXPERIMENT C  --  RUL FROM THE FORECASTABLE HEALTH STATE  (FD00{fd})")
    print("=" * 78)
    os.makedirs(mc.ART_DIR, exist_ok=True)
    man = mc.get_manifold()

    # ---- TRAIN: every cycle of every official train engine (causal) ------ #
    train = mc.load_split("train")
    train_den = mc.denoise(train, causal=True)
    Ftr = health_features(man, train_den)
    maxc = Ftr.groupby("unit_id")["cycle"].transform("max")
    Ftr["rul"] = np.minimum(maxc - Ftr["cycle"], RUL_CAP)

    feat_cols = ["h0", "h1", "v0", "v1"]
    Xtr = Ftr[feat_cols].to_numpy()
    ytr = Ftr["rul"].to_numpy()

    reg = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=400,
        l2_regularization=1.0, random_state=SEED)
    reg.fit(Xtr, ytr)

    # ---- failure threshold for the naive estimator (median end-of-life) -- #
    thr = float(Ftr.sort_values("cycle").groupby("unit_id")["h0"].last().median())

    # ---- TEST: official truncated engines + ground-truth RUL ------------- #
    test = mc.load_split("test")
    test_den = mc.denoise(test, causal=True)
    Fte = health_features(man, test_den)
    rul_true = mc.load_rul()["rul"].to_numpy()

    units = sorted(Fte["unit_id"].unique().tolist())
    last = (Fte.sort_values("cycle").groupby("unit_id").tail(1)
            .set_index("unit_id").loc[units])

    # (ii) robust supervised health -> RUL ------------------------------- #
    pred_reg = reg.predict(last[feat_cols].to_numpy())
    pred_reg = np.clip(pred_reg, 0, RUL_CAP)

    # (i) naive forecast-to-threshold ------------------------------------ #
    h0_last = last["h0"].to_numpy()
    v0_last = last["v0"].to_numpy()
    pred_thr = np.where(
        h0_last >= thr, 0.0,
        np.where(v0_last <= MIN_VEL, RUL_CAP,
                 np.clip((thr - h0_last) / np.maximum(v0_last, MIN_VEL), 0, RUL_CAP)))

    # baseline: constant mean train RUL (capped, same convention) -------- #
    pred_base = np.full_like(rul_true, float(np.mean(ytr)), dtype=float)

    m_reg = score(rul_true, pred_reg)
    m_thr = score(rul_true, pred_thr)
    m_base = score(rul_true, pred_base)

    print(f"  failure threshold (median end-of-life h0) = {thr:.4f}")
    print(f"  engines scored: {len(units)}\n")
    print("  estimator            RMSE     MAE      R2       NASA")
    print("  " + "-" * 56)
    for name, m in [("health->RUL (robust)", m_reg),
                    ("threshold-crossing  ", m_thr),
                    ("mean-RUL baseline   ", m_base)]:
        print(f"  {name}  {m['RMSE']:6.2f}  {m['MAE']:6.2f}  "
              f"{m['R2']:+6.3f}  {m['NASA']:11.1f}")

    verdict = ("BEATS" if m_reg["RMSE"] < m_base["RMSE"] else "does NOT beat")
    print(f"\n  -> robust health->RUL {verdict} the mean baseline on RMSE")

    # ------------------------------ persist ------------------------------ #
    pd.DataFrame({
        "unit_id": units,
        "rul_true": rul_true,
        "pred_health_rul": pred_reg,
        "pred_threshold": pred_thr,
        "h0_last": h0_last,
        "v0_last": v0_last,
    }).to_csv(os.path.join(mc.ART_DIR, "rul_predictions.csv"), index=False)

    with open(os.path.join(mc.ART_DIR, "rul_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"threshold": thr, "rul_cap": RUL_CAP, "vel_window": VEL_WINDOW,
                   "health_to_rul": m_reg, "threshold_crossing": m_thr,
                   "mean_baseline": m_base}, fh, indent=2)

    # ------------------------------ figures ------------------------------ #
    fdir = mc.fig_dir()

    # C1 -- predicted vs true scatter + error histogram
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    lim = max(rul_true.max(), pred_reg.max()) * 1.05
    ax[0].plot([0, lim], [0, lim], "k--", lw=1, label="ideal")
    ax[0].scatter(rul_true, pred_reg, s=28, c="#1f77b4", alpha=0.75,
                  edgecolor="w", linewidth=0.4, label="health->RUL")
    ax[0].set_xlabel("true RUL (cycles)")
    ax[0].set_ylabel("predicted RUL (cycles)")
    ax[0].set_title(f"Predicted vs true RUL\nRMSE={m_reg['RMSE']:.1f}, "
                    f"R2={m_reg['R2']:+.2f} (baseline RMSE={m_base['RMSE']:.1f})")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=0.3)

    err = pred_reg - rul_true
    ax[1].axvline(0, color="k", lw=1)
    ax[1].hist(err, bins=20, color="#ff7f0e", alpha=0.85, edgecolor="w")
    ax[1].set_xlabel("prediction error  (pred - true)  [late = positive]")
    ax[1].set_ylabel("engines")
    ax[1].set_title(f"Error distribution\nMAE={m_reg['MAE']:.1f}, "
                    f"bias={err.mean():+.1f}")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "C1_rul_scatter.png"), dpi=130)
    plt.close(fig)
    print(f"  saved {os.path.join(fdir, 'C1_rul_scatter.png')}")

    # C2 -- example engines: causal h0 trajectory + RUL readout
    err_abs = np.abs(pred_reg - rul_true)
    order = np.argsort(err_abs)
    picks = list(order[:3]) + list(order[len(order) // 2: len(order) // 2 + 1]) \
        + list(order[-2:])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
    for ax, idx in zip(axes.ravel(), picks):
        uid = units[idx]
        g = Fte[Fte["unit_id"] == uid].sort_values("cycle")
        ax.plot(g["cycle"], g["h0"], color="#1f77b4", lw=1.8, label="causal h0")
        ax.axhline(thr, color="r", ls="--", lw=1, label="failure thr")
        ax.set_title(f"engine {uid}: true={rul_true[idx]:.0f}, "
                     f"pred={pred_reg[idx]:.0f} cyc")
        ax.set_xlabel("cycle")
        ax.set_ylabel("h0 (health)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Example test engines  --  causal health vs RUL readout",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "C2_examples.png"), dpi=130)
    plt.close(fig)
    print(f"  saved {os.path.join(fdir, 'C2_examples.png')}")

    # C3 -- learned health -> RUL surface (h0 & v0 vs RUL on TRAIN)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    sc0 = ax[0].scatter(Ftr["h0"], Ftr["rul"], s=4, c=Ftr["v0"],
                        cmap="viridis", alpha=0.4)
    ax[0].set_xlabel("health level h0")
    ax[0].set_ylabel("RUL (capped)")
    ax[0].set_title("RUL vs health level (colour = velocity v0)")
    fig.colorbar(sc0, ax=ax[0], label="v0")
    ax[0].grid(alpha=0.3)

    # marginal: binned mean RUL vs h0
    bins = np.linspace(Ftr["h0"].quantile(0.01), Ftr["h0"].quantile(0.99), 25)
    bi = np.digitize(Ftr["h0"], bins)
    mean_rul = [Ftr["rul"][bi == b].mean() for b in range(1, len(bins))]
    ax[1].plot(0.5 * (bins[:-1] + bins[1:]), mean_rul, "o-", color="#d62728")
    ax[1].set_xlabel("health level h0")
    ax[1].set_ylabel("mean RUL in bin")
    ax[1].set_title("Monotone health->RUL relationship (train)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "C3_health_vs_rul.png"), dpi=130)
    plt.close(fig)
    print(f"  saved {os.path.join(fdir, 'C3_health_vs_rul.png')}")

    return dict(fd=fd, threshold=thr,
                rul_rmse=m_reg["RMSE"], rul_mae=m_reg["MAE"],
                rul_r2=m_reg["R2"], rul_nasa=m_reg["NASA"],
                base_rmse=m_base["RMSE"], base_nasa=m_base["NASA"],
                thr_rmse=m_thr["RMSE"])


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
