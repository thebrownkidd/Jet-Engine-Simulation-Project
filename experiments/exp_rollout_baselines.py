"""
RESEARCH EXPERIMENT E4 -- ROLLOUT BASELINES (VAR2 / LSTM / GRU)
==============================================================

The original stability audit pitted the bounded 2-D manifold rollout against a
single sensor-space baseline: a first-order linear VAR.  A reviewer will ask:
is the manifold's stability advantage just an artifact of using a *weak* linear
baseline?  This experiment establishes stronger sensor-space rollout baselines
and rolls every one of them closed-loop, far past the data, to test which
diverge:

  manifold     bounded 2-D health rollout (the proposed method)
  var1         x_{t+1} = A x_t + b                 (original baseline)
  var2         x_{t+1} = A1 x_t + A2 x_{t-1} + b   (second-order linear)
  lstm         1-layer LSTM next-step predictor, closed-loop
  gru          1-layer GRU  next-step predictor, closed-loop

For each model we report, per dataset:
  rho                spectral radius (linear models; companion form for VAR2)
  freerun_norm       standardized state norm after 400 closed-loop steps
  freerun_growth     norm_400 / norm_0   (>>1 => divergence)
  nrmse_h{1,10,25,50} cross-engine NRMSE in informative-sensor std units
  bounded            freerun_growth < 5  (heuristic stability flag)

Usage:  python exp_rollout_baselines.py [fd ...]     (default: 1 2 3 4)

Outputs
  results/tables/research/rollout_baselines/baselines_summary.csv
  results/figures/research/rollout_baselines/freerun_FD00<fd>.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import manifold as mc
from exp_rollout_stability import (CUTOFF_FRAC, VEL_WINDOW, fit_sensor_var,
                                   rollout_manifold, rollout_var)

SEQ_WINDOW = 10
RNN_HIDDEN = 32
RNN_EPOCHS = 300
RNN_LR = 5e-3
FREE_STEPS = 400
SCORE_H = [1, 10, 25, 50]
MAX_H = 200


# --------------------------------------------------------------------------- #
# Linear VAR(2)
# --------------------------------------------------------------------------- #
def fit_var2(train_den: pd.DataFrame):
    mu = train_den[mc.DYNAMIC].mean().to_numpy()
    sd = train_den[mc.DYNAMIC].std().to_numpy() + 1e-12
    X, Y = [], []
    for _, g in train_den.groupby("unit_id"):
        z = (g[mc.DYNAMIC].to_numpy() - mu) / sd
        if len(z) < 3:
            continue
        X.append(np.hstack([z[1:-1], z[:-2]]))   # [x_t, x_{t-1}]
        Y.append(z[2:])                          # x_{t+1}
    X = np.vstack(X)
    Y = np.vstack(Y)
    reg = LinearRegression().fit(X, Y)
    d = len(mc.DYNAMIC)
    A1 = reg.coef_[:, :d]
    A2 = reg.coef_[:, d:]
    b = reg.intercept_
    # companion matrix spectral radius
    comp = np.block([[A1, A2], [np.eye(d), np.zeros((d, d))]])
    rho = float(np.max(np.abs(np.linalg.eigvals(comp))))
    return dict(A1=A1, A2=A2, b=b, mu=mu, sd=sd, rho=rho)


def rollout_var2(m, x0_raw, x1_raw, steps):
    """Closed-loop VAR2 rollout in raw units. needs two seed states."""
    z_prev = (x0_raw - m["mu"]) / m["sd"]
    z_cur = (x1_raw - m["mu"]) / m["sd"]
    out = []
    for _ in range(steps):
        z_next = m["A1"] @ z_cur + m["A2"] @ z_prev + m["b"]
        out.append(z_next * m["sd"] + m["mu"])
        z_prev, z_cur = z_cur, z_next
    return np.array(out)


# --------------------------------------------------------------------------- #
# RNN next-step predictors (LSTM / GRU)
# --------------------------------------------------------------------------- #
class RNNStep(nn.Module):
    def __init__(self, n_in, hidden, kind):
        super().__init__()
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_in)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


def _windows(train_den, mu, sd):
    Xs, Ys = [], []
    for _, g in train_den.groupby("unit_id"):
        z = ((g[mc.DYNAMIC].to_numpy() - mu) / sd).astype(np.float32)
        if len(z) <= SEQ_WINDOW:
            continue
        for t in range(SEQ_WINDOW, len(z)):
            Xs.append(z[t - SEQ_WINDOW:t])
            Ys.append(z[t])
    return np.asarray(Xs, np.float32), np.asarray(Ys, np.float32)


def fit_rnn(train_den, kind):
    torch.manual_seed(mc.SEED)
    np.random.seed(mc.SEED)
    mu = train_den[mc.DYNAMIC].mean().to_numpy()
    sd = train_den[mc.DYNAMIC].std().to_numpy() + 1e-12
    X, Y = _windows(train_den, mu, sd)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    model = RNNStep(len(mc.DYNAMIC), RNN_HIDDEN, kind)
    opt = torch.optim.Adam(model.parameters(), lr=RNN_LR)
    lossf = nn.MSELoss()
    bs = 4096
    n = len(Xt)
    for _ in range(RNN_EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = lossf(pred, Yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    return dict(model=model, mu=mu, sd=sd, kind=kind)


def rollout_rnn(m, seed_raw, steps):
    """Closed-loop RNN rollout. seed_raw: (>=SEQ_WINDOW, d) raw units."""
    z = ((seed_raw - m["mu"]) / m["sd"]).astype(np.float32)
    window = list(z[-SEQ_WINDOW:])
    out = []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(np.asarray(window[-SEQ_WINDOW:])[None, :, :])
            nxt = m["model"](x).numpy()[0]
            window.append(nxt)
            out.append(nxt * m["sd"] + m["mu"])
    return np.array(out)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def standardized_norm(state_raw, mu, sd):
    return float(np.linalg.norm((state_raw - mu) / sd))


def evaluate(fd: int) -> list[dict]:
    mc.configure(fd)
    df = mc.load_split("train")
    tr, te = mc.split_by_unit(df)
    tr_den = mc.denoise(tr)
    te_den = mc.denoise(te)
    man = mc.get_manifold()

    var1 = fit_sensor_var(tr_den)
    var2 = fit_var2(tr_den)
    lstm = fit_rnn(tr_den, "lstm")
    gru = fit_rnn(tr_den, "gru")
    print(f"  FD00{fd}: rho(VAR1)={var1['rho']:.3f}  rho(VAR2)={var2['rho']:.3f}")

    mu, sd = var1["mu"], var1["sd"]
    inf_idx = [mc.DYNAMIC.index(s) for s in mc.INFORMATIVE]
    sigma = tr_den[mc.INFORMATIVE].std().to_numpy() + 1e-9

    models = ["manifold", "var1", "var2", "lstm", "gru"]
    per_h = {mdl: {h: {"T": [], "P": []} for h in range(1, MAX_H + 1)} for mdl in models}
    free_traces = {mdl: [] for mdl in models}

    longest = max(te_den["unit_id"].unique(),
                  key=lambda u: (te_den["unit_id"] == u).sum())

    for uid, g in te_den.groupby("unit_id"):
        g = g.sort_values("cycle").reset_index(drop=True)
        n = len(g)
        c0 = int(CUTOFF_FRAC * n)
        if c0 < max(VEL_WINDOW, SEQ_WINDOW) + 2 or c0 >= n - 5:
            continue
        steps = min(n - 1 - c0, MAX_H)
        truth = g[mc.DYNAMIC].to_numpy()

        rolls = {
            "manifold": rollout_manifold(man, man.encode(g.iloc[:c0 + 1]), steps),
            "var1": rollout_var(var1, truth[c0], steps),
            "var2": rollout_var2(var2, truth[c0 - 1], truth[c0], steps),
            "lstm": rollout_rnn(lstm, truth[:c0 + 1], steps),
            "gru": rollout_rnn(gru, truth[:c0 + 1], steps),
        }
        for mdl, roll in rolls.items():
            for s in range(steps):
                per_h[mdl][s + 1]["T"].append(truth[c0 + s + 1, inf_idx])
                per_h[mdl][s + 1]["P"].append(roll[s, inf_idx])

    # free-run on the longest engine
    g = te_den[te_den["unit_id"] == longest].sort_values("cycle").reset_index(drop=True)
    c0 = int(CUTOFF_FRAC * len(g))
    truth = g[mc.DYNAMIC].to_numpy()
    free = {
        "manifold": rollout_manifold(man, man.encode(g.iloc[:c0 + 1]), FREE_STEPS),
        "var1": rollout_var(var1, truth[c0], FREE_STEPS),
        "var2": rollout_var2(var2, truth[c0 - 1], truth[c0], FREE_STEPS),
        "lstm": rollout_rnn(lstm, truth[:c0 + 1], FREE_STEPS),
        "gru": rollout_rnn(gru, truth[:c0 + 1], FREE_STEPS),
    }
    for mdl, roll in free.items():
        free_traces[mdl] = [standardized_norm(roll[s], mu, sd) for s in range(FREE_STEPS)]

    rho_map = {"manifold": float("nan"), "var1": var1["rho"], "var2": var2["rho"],
               "lstm": float("nan"), "gru": float("nan")}
    rows = []
    for mdl in models:
        trace = np.array(free_traces[mdl])
        growth = float(trace[-1] / (trace[0] + 1e-12))
        row = dict(dataset=f"FD00{fd}", model=mdl, rho=rho_map[mdl],
                   freerun_norm=float(trace[-1]), freerun_growth=growth,
                   bounded=bool(growth < 5.0))
        for h in SCORE_H:
            T = per_h[mdl][h]["T"]
            P = per_h[mdl][h]["P"]
            if len(T) >= 5:
                T = np.array(T) / sigma
                P = np.array(P) / sigma
                row[f"nrmse_h{h}"] = float(np.sqrt(np.mean((T - P) ** 2)))
            else:
                row[f"nrmse_h{h}"] = float("nan")
        rows.append(row)
        print(f"    {mdl:<9} freerun_norm={row['freerun_norm']:10.2f}  "
              f"growth=x{growth:8.1f}  bounded={row['bounded']}  "
              f"nrmse_h50={row.get('nrmse_h50', float('nan')):.2f}")

    _plot_free(fd, free_traces)
    return rows


def _plot_free(fd, traces):
    out_dir = os.path.join(ROOT, "results", "figures", "research", "rollout_baselines")
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    col = {"manifold": "#1b7837", "var1": "#b2182b", "var2": "#d6604d",
           "lstm": "#4c72b0", "gru": "#dd8452"}
    for mdl, tr in traces.items():
        ax.plot(tr, color=col[mdl], lw=2, label=mdl)
    ax.set_yscale("log")
    ax.set_xlabel("free-run step (closed-loop, beyond data)")
    ax.set_ylabel("standardized state norm (log)")
    ax.set_title(f"Free-run divergence of rollout baselines -- FD00{fd}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"freerun_FD00{fd}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(fds):
    tab_dir = os.path.join(ROOT, "results", "tables", "research", "rollout_baselines")
    os.makedirs(tab_dir, exist_ok=True)
    rows = []
    for fd in fds:
        print("=" * 78)
        print(f"ROLLOUT BASELINES  FD00{fd}")
        print("=" * 78)
        rows.extend(evaluate(fd))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(tab_dir, "baselines_summary.csv"), index=False)
    print("\n" + "=" * 78)
    print("ROLLOUT BASELINES SUMMARY")
    print("=" * 78)
    cols = ["dataset", "model", "rho", "freerun_norm", "freerun_growth",
            "bounded", "nrmse_h1", "nrmse_h50"]
    print(df[cols].to_string(index=False))
    print(f"\nsaved -> {os.path.join(tab_dir, 'baselines_summary.csv')}")
    return df


if __name__ == "__main__":
    fds = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 4]
    main(fds)
