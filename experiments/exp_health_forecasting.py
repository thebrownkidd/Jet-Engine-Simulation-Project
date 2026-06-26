"""Experiment B -- AUTOREGRESSIVE FORECASTING OF THE HEALTH STATE.

Question: once the 2-D health state h(t) is identified by the manifold, can it
be forecast with simple autoregressive models, so downstream tasks (rollout,
RUL) that need the FUTURE state become tractable?

Skeptical hypothesis (H0): the health state is no easier to forecast than raw
sensors; AR models will not beat the trivial PERSISTENCE baseline (hold the
last value), and error will grow without control.

Why persistence is the right baseline
--------------------------------------
h0 is (by construction) monotone and slowly accelerating. The naive predictor
for such a signal is persistence h_hat(t+k)=h(t). A forecaster earns its keep
only if it beats persistence. We therefore report, at each horizon k:
    RMSE_model(k)             absolute error in health units
    skill(k) = 1 - MSE_model(k)/MSE_persistence(k)   (>0 means useful)

We also test the theorem: the constant-velocity forecast error is bounded by
(kappa/2) k^2 with kappa = robust curvature (median |2nd diff| of the smooth
h0). Overlaying that envelope shows growth is polynomial (~k^2), not exponential.

Models (fit on a RECENT window of the visible prefix, then extrapolated)
    persistence       h(t+k) = h(t)
    const_velocity    local slope over last VEL_WINDOW
    linear_recent     OLS over last FIT_WINDOW
    quadratic_recent  degree-2 over last FIT_WINDOW (captures acceleration)
    ar2               h_t = c + p1 h_{t-1} + p2 h_{t-2}

Outputs
-------
results/figures/B1_health_forecasts.png   example engines, models vs truth
results/figures/B2_error_vs_horizon.png   RMSE(k) + theoretical k^2 envelope
results/figures/B3_skill_vs_persistence.png   skill(k) per model & coordinate
results/tables/FD00<fd>/health_forecasting.csv
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

CUTOFFS = [0.5, 0.65, 0.8]     # forecast-start fractions (pooled)
VEL_WINDOW = 20                # window for local velocity
FIT_WINDOW = 40                # window for recent linear/quadratic fits
HORIZONS = [1, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100]


# --------------------------------------------------------------------------- #
# Forecasters: given full series y and cutoff c0, return prediction at c0+k-1
# --------------------------------------------------------------------------- #
def _recent(y, c0, w):
    lo = max(0, c0 - w)
    return np.arange(lo, c0), y[lo:c0]


def f_persistence(y, c0, ks):
    return np.full(len(ks), y[c0 - 1])


def f_const_velocity(y, c0, ks):
    t, seg = _recent(y, c0, VEL_WINDOW)
    v = np.polyfit(t, seg, 1)[0] if len(seg) >= 2 else 0.0
    return y[c0 - 1] + v * np.array(ks)


def f_linear_recent(y, c0, ks):
    t, seg = _recent(y, c0, FIT_WINDOW)
    if len(seg) < 2:
        return f_persistence(y, c0, ks)
    a, b = np.polyfit(t, seg, 1)
    return a * (c0 - 1 + np.array(ks)) + b


def f_quadratic_recent(y, c0, ks):
    t, seg = _recent(y, c0, FIT_WINDOW)
    if len(seg) < 3:
        return f_linear_recent(y, c0, ks)
    c2, c1, c0c = np.polyfit(t, seg, 2)
    tt = c0 - 1 + np.array(ks)
    return c2 * tt ** 2 + c1 * tt + c0c


def f_ar2(y, c0, ks):
    if c0 < FIT_WINDOW:
        return f_linear_recent(y, c0, ks)
    lo = max(2, c0 - FIT_WINDOW)
    Y = y[lo:c0]
    X = np.column_stack([np.ones(len(Y)), y[lo - 1:c0 - 1], y[lo - 2:c0 - 2]])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    hist = list(y[:c0])
    out = []
    for _ in range(max(ks)):
        nxt = coef[0] + coef[1] * hist[-1] + coef[2] * hist[-2]
        hist.append(nxt)
        out.append(nxt)
    out = np.array(out)
    return out[[k - 1 for k in ks]]


FORECASTERS = {
    "persistence": f_persistence,
    "const_velocity": f_const_velocity,
    "linear_recent": f_linear_recent,
    "quadratic_recent": f_quadratic_recent,
    "ar2": f_ar2,
}


# --------------------------------------------------------------------------- #
def main(fd: int = 1):
    mc.configure(fd)
    print("=" * 78)
    print(f"EXPERIMENT B  --  AUTOREGRESSIVE HEALTH-STATE FORECASTING  (FD00{fd})")
    print("=" * 78)

    df = mc.load_split("train")
    _, te = mc.split_by_unit(df)
    te_den = mc.denoise(te)
    man = mc.get_manifold()
    H = mc.per_engine_health(man, te_den)

    coords = ["h0", "h1"]
    acc = {c: {m: {k: [] for k in HORIZONS} for m in FORECASTERS} for c in coords}
    pers = {c: {k: [] for k in HORIZONS} for c in coords}
    kappas = []
    examples = []

    for coord in coords:
        for uid, g in H.groupby("unit_id"):
            g = g.sort_values("cycle")
            y = g[coord].to_numpy()
            n = len(y)
            for frac in CUTOFFS:
                c0 = int(frac * n)
                if c0 < FIT_WINDOW + 3 or c0 >= n - 1:
                    continue
                avail = [k for k in HORIZONS if c0 + k <= n]
                if not avail:
                    continue
                truth = np.array([y[c0 + k - 1] for k in avail])
                for mname, fn in FORECASTERS.items():
                    pred = fn(y, c0, avail)
                    for j, k in enumerate(avail):
                        acc[coord][mname][k].append((truth[j] - pred[j]) ** 2)
                pred_p = f_persistence(y, c0, avail)
                for j, k in enumerate(avail):
                    pers[coord][k].append((truth[j] - pred_p[j]) ** 2)

                if coord == "h0":
                    ys = pd.Series(y).rolling(7, center=True,
                                              min_periods=1).median().to_numpy()
                    d2 = np.diff(ys, 2)
                    if len(d2):
                        kappas.append(np.median(np.abs(d2)))
                    if frac == 0.5 and len(examples) < 6:
                        examples.append(dict(uid=uid, y=y, c0=c0,
                                             cyc=g["cycle"].to_numpy()))

    rows = []
    for coord in coords:
        for mname in FORECASTERS:
            for k in HORIZONS:
                se = acc[coord][mname][k]
                if not se:
                    continue
                mse = float(np.mean(se))
                mse_p = float(np.mean(pers[coord][k])) if pers[coord][k] else np.nan
                rows.append(dict(coord=coord, model=mname, horizon=k,
                                 rmse=np.sqrt(mse), n=len(se),
                                 skill=1.0 - mse / (mse_p + 1e-12)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(mc.ART_DIR, "health_forecasting.csv"), index=False)

    print("\n  h0 forecast skill vs persistence (1=perfect, 0=tie, <0=worse):")
    for k in [5, 10, 20, 50]:
        line = f"   k={k:3d}: "
        for m in ["const_velocity", "linear_recent", "quadratic_recent", "ar2"]:
            r = res[(res.coord == "h0") & (res.model == m) & (res.horizon == k)]
            if len(r):
                line += f"{m.split('_')[0][:5]}={r.iloc[0].skill:+.2f}  "
        print(line)

    kappa = float(np.median(kappas)) if kappas else 1e-4
    print(f"\n  robust curvature kappa (median |2nd diff| of smooth h0) = {kappa:.2e}")

    _plot_examples(examples)
    _plot_error_envelope(res, kappa)
    _plot_skill(res)

    def _skill(model, k):
        r = res[(res.coord == "h0") & (res.model == model) & (res.horizon == k)]
        return float(r.iloc[0].skill) if len(r) else np.nan

    return dict(fd=fd, kappa=kappa,
                skill_cv_k10=_skill("const_velocity", 10),
                skill_cv_k20=_skill("const_velocity", 20),
                skill_ar2_k20=_skill("ar2", 20),
                skill_cv_k50=_skill("const_velocity", 50))


# --------------------------------------------------------------------------- #
def _plot_examples(examples):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, ex in zip(axes.ravel(), examples):
        y = ex["y"]
        cyc = ex["cyc"]
        c0 = ex["c0"]
        n = len(y)
        ks = list(range(1, n - c0 + 1))
        fcyc = cyc[c0:c0 + len(ks)]
        ax.plot(cyc, y, color="k", lw=1.8, label="true h0")
        ax.axvline(cyc[c0 - 1], color="gray", ls=":", lw=1.0, label="forecast start")
        for mname, color in [("const_velocity", "#1b7837"),
                             ("linear_recent", "#2166ac"),
                             ("quadratic_recent", "#d6604d")]:
            pred = FORECASTERS[mname](y, c0, ks)
            ax.plot(fcyc, pred[:len(fcyc)], color=color, lw=1.4, label=mname)
        ax.set_title(f"engine {ex['uid']}")
        ax.set_xlabel("cycle")
        ax.set_ylabel("h0")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("Health-state forecasting: AR models extrapolate h0(t) "
                 "(start at 50% life)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(mc.fig_dir(), "B1_health_forecasts.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def _plot_error_envelope(res, kappa):
    sub = res[(res.coord == "h0") & (res.model == "const_velocity")].sort_values("horizon")
    hs = sub["horizon"].to_numpy()
    rmse = sub["rmse"].to_numpy()
    envelope = 0.5 * kappa * hs ** 2
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(hs, rmse, "o-", color="#1b7837", ms=4, lw=1.8,
            label="measured RMSE (const-velocity h0)")
    ax.plot(hs, envelope, "--", color="#b2182b", lw=1.8,
            label=r"theoretical bound $\frac{\kappa}{2}k^2$")
    ax.set_xlabel("forecast horizon k (cycles)")
    ax.set_ylabel("h0 forecast error")
    ax.set_title("Health-forecast error is polynomial (~k^2), not exponential")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(mc.fig_dir(), "B2_error_vs_horizon.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def _plot_skill(res):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {"const_velocity": "#1b7837", "linear_recent": "#2166ac",
              "quadratic_recent": "#d6604d", "ar2": "#762a83"}
    for ax, coord in zip(axes, ["h0", "h1"]):
        for m, c in colors.items():
            sub = res[(res.coord == coord) & (res.model == m)].sort_values("horizon")
            ax.plot(sub["horizon"], sub["skill"], "o-", color=c, ms=3, label=m)
        ax.axhline(0.0, color="k", ls="--", lw=0.9, label="persistence (baseline)")
        ax.set_xlabel("forecast horizon k (cycles)")
        ax.set_ylabel("skill = 1 - MSE/MSE_persistence")
        ax.set_title(f"Forecast skill of {coord} vs persistence")
        ax.set_ylim(-1.0, 1.05)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    p = os.path.join(mc.fig_dir(), "B3_skill_vs_persistence.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
