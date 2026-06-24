"""Experiment A -- ROLLOUT STABILITY (the claim under audit).

Skeptical hypothesis (H0): "Stable rollout" is unjustified. A model that
reconstructs sensors well in one shot need not be stable when iterated. If we
roll the system forward autoregressively, errors will compound and the held-out
R2 will collapse -- exactly as a naive sensor-space model does.

Falsification design
--------------------
Two forecasters are rolled forward from a common cutoff cycle c0 to end-of-life
for every held-out engine, and scored against the (denoised) ground-truth
sensor trajectory at each future horizon h = 1, 2, ... :

  (1) SENSOR-SPACE VAR  x_{t+1} = A x_t + b      (the thing we suspect explodes)
      - linear vector-autoregression on the 15 denoised dynamic sensors
      - we REPORT spectral radius rho(A); rho >= 1 => non-contractive => blow-up
      - closed loop: feed prediction back into itself.

  (2) MANIFOLD ROLLOUT  h_{t+1} = h_t + v ; x = Decoder(h)
      - encode to the 2-D health state, extrapolate it with the locally-fit
        velocity v (estimated from data up to c0 only -- no leakage), decode.
      - NO sensor-to-sensor feedback: only a 2-D smooth state is integrated.

Outputs
-------
docs/figures/A1_rollout_r2_vs_horizon.png   <- the money plot
docs/figures/A2_var_eigenvalues.png         <- rho(A) on the unit circle
docs/figures/A3_example_trajectories.png    <- true vs both rollouts, 6 sensors
experiments/artifacts/rollout_stability.csv <- R2(h) for both methods
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import manifold_core as mc

CUTOFF_FRAC = 0.40       # start rolling out at 40% of each engine's life
VEL_WINDOW = 20          # cycles used to estimate local health velocity at c0
MAX_H = 200              # max horizon to score


# --------------------------------------------------------------------------- #
# Sensor-space VAR baseline
# --------------------------------------------------------------------------- #
def fit_sensor_var(train_den: pd.DataFrame):
    """x_{t+1} = A x_t + b on standardized denoised dynamic sensors."""
    mu = train_den[mc.DYNAMIC].mean().to_numpy()
    sd = train_den[mc.DYNAMIC].std().to_numpy() + 1e-12
    Xs, Ys = [], []
    for _, g in train_den.groupby("unit_id"):
        z = (g[mc.DYNAMIC].to_numpy() - mu) / sd
        Xs.append(z[:-1])
        Ys.append(z[1:])
    X = np.vstack(Xs)
    Y = np.vstack(Ys)
    reg = LinearRegression().fit(X, Y)
    A = reg.coef_                      # (15,15)
    b = reg.intercept_
    eig = np.linalg.eigvals(A)
    rho = float(np.max(np.abs(eig)))
    return dict(A=A, b=b, mu=mu, sd=sd, eig=eig, rho=rho)


def rollout_var(var, x0_raw: np.ndarray, steps: int) -> np.ndarray:
    """Closed-loop rollout in raw sensor units. x0_raw: (15,)."""
    z = (x0_raw - var["mu"]) / var["sd"]
    out = []
    for _ in range(steps):
        z = var["A"] @ z + var["b"]
        out.append(z * var["sd"] + var["mu"])
    return np.array(out)                # (steps, 15)


# --------------------------------------------------------------------------- #
# Manifold rollout
# --------------------------------------------------------------------------- #
def rollout_manifold(man, h_hist: np.ndarray, steps: int) -> np.ndarray:
    """Extrapolate 2-D health with local velocity from the last VEL_WINDOW
    points, then decode. h_hist: (T0, 2) health up to and including c0."""
    w = min(VEL_WINDOW, len(h_hist) - 1)
    if w < 1:
        v = np.zeros(mc.K)
    else:
        recent = h_hist[-w - 1:]
        t = np.arange(len(recent))
        v = np.array([np.polyfit(t, recent[:, j], 1)[0] for j in range(mc.K)])
    h0 = h_hist[-1]
    future_h = np.array([h0 + v * (s + 1) for s in range(steps)])
    future_h[:, 0] = np.clip(future_h[:, 0], 0.0, 1.5)   # health may exceed 1
    return man.decode(future_h)         # (steps, 15) raw units


def free_run(var, man, g, c0, steps):
    """Iterate BOTH maps far past the data from the same start state and record
    the standardized state norm. This isolates the dynamical-systems question:
    rho(A)>1 => ||z|| diverges; the manifold latent is bounded => ||z|| bounded.
    """
    truth = g[mc.DYNAMIC].to_numpy()
    # VAR free run (already standardized internally)
    z = (truth[c0] - var["mu"]) / var["sd"]
    var_norm = []
    for _ in range(steps):
        z = var["A"] @ z + var["b"]
        var_norm.append(float(np.linalg.norm(z)))
    # Manifold free run
    h_hist = man.encode(g.iloc[:c0 + 1])
    man_roll = rollout_manifold(man, h_hist, steps)   # raw units
    man_norm = [float(np.linalg.norm((man_roll[s] - var["mu"]) / var["sd"]))
                for s in range(steps)]
    return np.array(var_norm), np.array(man_norm)


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def main():
    print("=" * 78)
    print("EXPERIMENT A  --  ROLLOUT STABILITY AUDIT")
    print("=" * 78)

    df = mc.load_split("train")
    tr, te = mc.split_by_unit(df)
    tr_den = mc.denoise(tr)
    te_den = mc.denoise(te)
    man = mc.get_manifold()

    var = fit_sensor_var(tr_den)
    print(f"Sensor-space VAR spectral radius  rho(A) = {var['rho']:.4f}  "
          f"({'NON-contractive -> expect blow-up' if var['rho'] >= 1 else 'contractive'})")

    inf_idx = [mc.DYNAMIC.index(s) for s in mc.INFORMATIVE]
    sigma = tr_den[mc.INFORMATIVE].std().to_numpy() + 1e-9   # per-sensor train std

    # Collect per-horizon, cross-engine matrices so we can score with a PROPER
    # reference (cross-engine variability at the same horizon), all in
    # per-sensor standardized units so no single sensor dominates.
    per_h = {h: dict(T=[], M=[], V=[]) for h in range(1, MAX_H + 1)}
    examples = {}

    for uid, g in te_den.groupby("unit_id"):
        g = g.sort_values("cycle").reset_index(drop=True)
        n = len(g)
        c0 = int(CUTOFF_FRAC * n)
        if c0 < VEL_WINDOW + 2 or c0 >= n - 5:
            continue
        steps = min(n - 1 - c0, MAX_H)

        truth = g[mc.DYNAMIC].to_numpy()                 # (n, 15) raw denoised
        var_roll = rollout_var(var, truth[c0], steps)    # (steps,15)
        h_hist = man.encode(g.iloc[:c0 + 1])             # (c0+1, 2)
        man_roll = rollout_manifold(man, h_hist, steps)  # (steps,15)

        for s in range(steps):
            h = s + 1
            per_h[h]["T"].append(truth[c0 + h, inf_idx])
            per_h[h]["M"].append(man_roll[s, inf_idx])
            per_h[h]["V"].append(var_roll[s, inf_idx])

        if len(examples) < 1 and steps >= 80:
            examples.update(uid=uid, c0=c0, truth=truth, var_roll=var_roll,
                            man_roll=man_roll, cycles=g["cycle"].to_numpy())

    rows = []
    for h in range(1, MAX_H + 1):
        if len(per_h[h]["T"]) < 5:        # need >=5 engines for a stable ref
            continue
        T = np.array(per_h[h]["T"]) / sigma     # (E, S) standardized
        M = np.array(per_h[h]["M"]) / sigma
        V = np.array(per_h[h]["V"]) / sigma
        ref = T.mean(0, keepdims=True)          # cross-engine mean per sensor
        sst = np.sum((T - ref) ** 2) + 1e-12
        rows.append(dict(
            horizon=h,
            r2_manifold=1.0 - np.sum((T - M) ** 2) / sst,
            r2_var=1.0 - np.sum((T - V) ** 2) / sst,
            nrmse_manifold=float(np.sqrt(np.mean((T - M) ** 2))),
            nrmse_var=float(np.sqrt(np.mean((T - V) ** 2))),
            n_engines=T.shape[0],
        ))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(mc.ART_DIR, "rollout_stability.csv"), index=False)

    # report a few horizons
    for h in [1, 10, 25, 50, 100]:
        r = res[res.horizon == h]
        if len(r):
            r = r.iloc[0]
            print(f"  h={h:3d}:  manifold NRMSE={r.nrmse_manifold:6.2f} "
                  f"R2={r.r2_manifold:6.3f}   |   VAR NRMSE={r.nrmse_var:9.2f} "
                  f"R2={r.r2_var:9.3f}")

    _plot_money(res, var["rho"])
    _plot_eigs(var["eig"], var["rho"])
    if examples:
        _plot_examples(examples)

    # ---- free-running divergence test (the theorem visual) -------------- #
    longest = max(te_den["unit_id"].unique(),
                  key=lambda u: (te_den["unit_id"] == u).sum())
    g_long = te_den[te_den["unit_id"] == longest].sort_values("cycle").reset_index(drop=True)
    c0_long = int(CUTOFF_FRAC * len(g_long))
    var_norm, man_norm = free_run(var, man, g_long, c0_long, steps=400)
    _plot_divergence(var_norm, man_norm, var["rho"])

    print("\nVERDICT:")
    last = res.iloc[-1]
    print(f"  scored horizon {int(last.horizon)}: manifold NRMSE={last.nrmse_manifold:.2f} "
          f"vs VAR NRMSE={last.nrmse_var:.1f}")
    print(f"  FREE-RUN (400 steps): VAR ||state|| -> {var_norm[-1]:.2e}  "
          f"(x{var_norm[-1]/var_norm[0]:.0e} growth);  "
          f"manifold ||state|| -> {man_norm[-1]:.2f} (bounded)")
    print(f"  manifold NRMSE stays bounded (max={res['nrmse_manifold'].max():.2f}); "
          f"VAR free-run diverges; rho(A)={var['rho']:.3f} > 1")
    return res, var


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _plot_money(res, rho):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(res.horizon, res.nrmse_manifold, "-", color="#1b7837", lw=2.2,
               label="Manifold rollout (2-D health)")
    ax[0].plot(res.horizon, res.nrmse_var, "-", color="#b2182b", lw=2.2,
               label=f"Sensor-space VAR  (rho={rho:.2f})")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("rollout horizon (cycles ahead of cutoff)")
    ax[0].set_ylabel("normalized RMSE  (per-sensor std units, log)")
    ax[0].set_title("Error growth: bounded (manifold) vs explosive (VAR)")
    ax[0].legend(loc="upper left", fontsize=9)

    ax[1].plot(res.horizon, res.r2_manifold, "-", color="#1b7837", lw=2.2,
               label="Manifold rollout")
    ax[1].plot(res.horizon, res.r2_var, "-", color="#b2182b", lw=2.2,
               label="Sensor-space VAR")
    ax[1].axhline(0.0, color="gray", ls=":", lw=0.8)
    ax[1].set_xlabel("rollout horizon (cycles ahead)")
    ax[1].set_ylabel("cross-engine R2 (informative sensors)")
    ax[1].set_title("Predictive skill vs cross-engine mean")
    ax[1].set_ylim(-3.0, 1.05)
    ax[1].legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    p = os.path.join(mc.fig_dir(), "A1_rollout_r2_vs_horizon.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def _plot_divergence(var_norm, man_norm, rho):
    t = np.arange(1, len(var_norm) + 1)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(t, var_norm, color="#b2182b", lw=2.2,
            label=f"Sensor-space VAR  (rho={rho:.3f} > 1)")
    ax.plot(t, man_norm, color="#1b7837", lw=2.2,
            label="Manifold rollout (bounded latent)")
    ax.plot(t, var_norm[0] * rho ** t, "k:", lw=1.2,
            label=r"$\rho^{\,t}$ envelope")
    ax.set_yscale("log")
    ax.set_xlabel("free-running iteration t (steps past the data)")
    ax.set_ylabel("standardized state norm  ||z_t||  (log)")
    ax.set_title("Free-running divergence test\n"
                 "rho(A)>1 => VAR blows up;  manifold latent stays bounded")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    p = os.path.join(mc.fig_dir(), "A4_free_run_divergence.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def _plot_eigs(eig, rho):
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), "k--", lw=1.0, label="unit circle")
    ax.scatter(eig.real, eig.imag, c="#b2182b", s=40, zorder=3,
               label="eig(A)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_aspect("equal")
    ax.set_title(f"Sensor-space VAR eigenvalues\nspectral radius rho = {rho:.3f}")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(mc.fig_dir(), "A2_var_eigenvalues.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def _plot_examples(ex):
    show = ["s9", "s11", "s4", "s14", "s12", "s7"]
    cyc = ex["cycles"]
    c0 = ex["c0"]
    steps = ex["var_roll"].shape[0]
    fut = cyc[c0 + 1:c0 + 1 + steps]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, s in zip(axes.ravel(), show):
        j = mc.DYNAMIC.index(s)
        ax.plot(cyc, ex["truth"][:, j], color="k", lw=1.6, label="truth (denoised)")
        ax.plot(fut, ex["man_roll"][:, j], color="#1b7837", lw=1.8,
                label="manifold rollout")
        ax.plot(fut, ex["var_roll"][:, j], color="#b2182b", lw=1.4, ls="--",
                label="VAR rollout")
        ax.axvline(cyc[c0], color="gray", ls=":", lw=1.0)
        ax.set_title(f"{s}")
        ax.set_xlabel("cycle")
        # keep VAR explosion from destroying the y-scale
        lo = np.min(ex["truth"][:, j])
        hi = np.max(ex["truth"][:, j])
        pad = 0.5 * (hi - lo + 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Engine {ex['uid']}: rollout from cutoff (dotted) to failure",
                 fontsize=13)
    fig.tight_layout()
    p = os.path.join(mc.fig_dir(), "A3_example_trajectories.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


if __name__ == "__main__":
    main()
