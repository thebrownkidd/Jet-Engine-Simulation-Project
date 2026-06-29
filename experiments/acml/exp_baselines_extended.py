"""
ACML TASK 6 — Extended rollout baselines with a modern neural forecaster (TCN).

Adds a Temporal Convolutional Network (TCN) next-step forecaster to the existing
closed-loop rollout comparison (VAR, VAR2, LSTM, GRU, proposed manifold). The TCN
uses dilated causal 1-D convolutions over a lag window, the same train/test
protocol and the same closed-loop free-run evaluation as the other baselines.

We evaluate CLOSED-LOOP rollout (free-run), not just one-step prediction:
  rho (linear models), free-run norm + growth + bounded flag,
  rollout NRMSE @ {1,10,25,50}.

Reuses the production baseline fitters from experiments/exp_rollout_baselines.py
and the manifold rollout from experiments/exp_rollout_stability.py so the
comparison is apples-to-apples.

Outputs
  results/acml/tables/baselines_extended.csv
  results/acml/tables/baselines_extended.tex
  results/acml/figures/baselines_extended_freerun.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXP = os.path.join(ROOT, "experiments")
for p in (HERE, EXP):
    if p not in sys.path:
        sys.path.insert(0, p)

import acml_common as ac  # noqa: E402
import manifold as mc  # noqa: E402
from exp_rollout_stability import (CUTOFF_FRAC, VEL_WINDOW, fit_sensor_var,  # noqa: E402
                                   rollout_manifold, rollout_var)
from exp_rollout_baselines import (SEQ_WINDOW, FREE_STEPS, SCORE_H, MAX_H,  # noqa: E402
                                   fit_var2, rollout_var2, fit_rnn, rollout_rnn,
                                   standardized_norm)

TCN_HIDDEN = 32
TCN_LEVELS = 3
TCN_KERNEL = 3
TCN_EPOCHS = 300
TCN_LR = 5e-3
DATASETS = [1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# TCN next-step predictor
# --------------------------------------------------------------------------- #
class TCNBlock(nn.Module):
    def __init__(self, n_ch, kernel, dilation):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(n_ch, n_ch, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(n_ch, n_ch, kernel, padding=pad, dilation=dilation)
        self.relu = nn.ReLU()

    def _crop(self, x):
        return x[:, :, :-self.pad] if self.pad > 0 else x

    def forward(self, x):
        y = self.relu(self._crop(self.conv1(x)))
        y = self.relu(self._crop(self.conv2(y)))
        return x + y          # residual


class TCNStep(nn.Module):
    """Dilated causal TCN that maps a lag window to the next-step sensor vector."""

    def __init__(self, n_in, hidden, levels, kernel):
        super().__init__()
        self.inp = nn.Conv1d(n_in, hidden, 1)
        self.blocks = nn.ModuleList(
            [TCNBlock(hidden, kernel, 2 ** i) for i in range(levels)])
        self.head = nn.Linear(hidden, n_in)

    def forward(self, x):                 # x: (B, T, n_in)
        z = x.transpose(1, 2)             # (B, n_in, T)
        z = self.inp(z)
        for b in self.blocks:
            z = b(z)
        return self.head(z[:, :, -1])     # last timestep -> (B, n_in)


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


def fit_tcn(train_den, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu = train_den[mc.DYNAMIC].mean().to_numpy()
    sd = train_den[mc.DYNAMIC].std().to_numpy() + 1e-12
    X, Y = _windows(train_den, mu, sd)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    model = TCNStep(len(mc.DYNAMIC), TCN_HIDDEN, TCN_LEVELS, TCN_KERNEL)
    opt = torch.optim.Adam(model.parameters(), lr=TCN_LR)
    lossf = nn.MSELoss()
    bs, n = 4096, len(Xt)
    for _ in range(TCN_EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    return dict(model=model, mu=mu, sd=sd, kind="tcn")


def rollout_tcn(m, seed_raw, steps):
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
# Evaluation (manifold + var1 + var2 + lstm + gru + tcn)
# --------------------------------------------------------------------------- #
def evaluate(fd):
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
    tcn = fit_tcn(tr_den)
    print(f"  FD00{fd}: fitted var1/var2/lstm/gru/tcn")

    mu, sd = var1["mu"], var1["sd"]
    inf_idx = [mc.DYNAMIC.index(s) for s in mc.INFORMATIVE]
    sigma = tr_den[mc.INFORMATIVE].std().to_numpy() + 1e-9

    models = ["manifold", "var1", "var2", "lstm", "gru", "tcn"]
    per_h = {mdl: {h: {"T": [], "P": []} for h in range(1, MAX_H + 1)} for mdl in models}
    free_traces = {mdl: [] for mdl in models}
    longest = max(te_den["unit_id"].unique(), key=lambda u: (te_den["unit_id"] == u).sum())

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
            "tcn": rollout_tcn(tcn, truth[:c0 + 1], steps),
        }
        for mdl, roll in rolls.items():
            for s in range(steps):
                per_h[mdl][s + 1]["T"].append(truth[c0 + s + 1, inf_idx])
                per_h[mdl][s + 1]["P"].append(roll[s, inf_idx])

    g = te_den[te_den["unit_id"] == longest].sort_values("cycle").reset_index(drop=True)
    c0 = int(CUTOFF_FRAC * len(g))
    truth = g[mc.DYNAMIC].to_numpy()
    free = {
        "manifold": rollout_manifold(man, man.encode(g.iloc[:c0 + 1]), FREE_STEPS),
        "var1": rollout_var(var1, truth[c0], FREE_STEPS),
        "var2": rollout_var2(var2, truth[c0 - 1], truth[c0], FREE_STEPS),
        "lstm": rollout_rnn(lstm, truth[:c0 + 1], FREE_STEPS),
        "gru": rollout_rnn(gru, truth[:c0 + 1], FREE_STEPS),
        "tcn": rollout_tcn(tcn, truth[:c0 + 1], FREE_STEPS),
    }
    for mdl, roll in free.items():
        free_traces[mdl] = [standardized_norm(roll[s], mu, sd) for s in range(FREE_STEPS)]

    rho_map = {"manifold": float("nan"), "var1": var1["rho"], "var2": var2["rho"],
               "lstm": float("nan"), "gru": float("nan"), "tcn": float("nan")}
    rows = []
    for mdl in models:
        trace = np.array(free_traces[mdl])
        growth = float(trace[-1] / (trace[0] + 1e-12))
        row = dict(dataset=f"FD00{fd}", model=mdl, rho=rho_map[mdl],
                   freerun_norm=float(trace[-1]), freerun_growth=growth,
                   bounded=bool(growth < 5.0))
        for h in SCORE_H:
            T, P = per_h[mdl][h]["T"], per_h[mdl][h]["P"]
            if len(T) >= 5:
                T = np.array(T) / sigma
                P = np.array(P) / sigma
                row[f"nrmse_h{h}"] = float(np.sqrt(np.mean((T - P) ** 2)))
            else:
                row[f"nrmse_h{h}"] = float("nan")
        rows.append(row)
        print(f"    {mdl:<9} growth=x{growth:9.1f}  bounded={row['bounded']!s:<5} "
              f"nrmse_h50={row.get('nrmse_h50', float('nan')):.3f}")
    _plot_free(fd, free_traces)
    return rows


def _plot_free(fd, traces):
    out_dir = ac.ACML_FIG
    fig, ax = plt.subplots(figsize=(8, 5))
    col = {"manifold": "#1b7837", "var1": "#b2182b", "var2": "#d6604d",
           "lstm": "#4c72b0", "gru": "#dd8452", "tcn": "#9467bd"}
    for mdl, tr in traces.items():
        ax.plot(tr, color=col[mdl], lw=2, label=mdl)
    ax.set_yscale("log")
    ax.set_xlabel("free-run step (closed-loop, beyond data)")
    ax.set_ylabel("standardized state norm (log)")
    ax.set_title(f"Free-run divergence incl. TCN baseline — FD00{fd}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, f"baselines_extended_freerun_FD00{fd}.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_grid(df):
    fds = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, ds in zip(axes.ravel(), fds):
        sub = df[df.dataset == ds]
        ax.bar(sub["model"], sub["freerun_growth"],
               color=["#1b7837" if m == "manifold" else "#b2182b" for m in sub["model"]])
        ax.axhline(5.0, color="r", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_title(ds)
        ax.set_ylabel("free-run growth (log)")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Closed-loop free-run growth by model (incl. TCN). "
                 "Manifold stays bounded; sensor-space models vary.", fontsize=12)
    fig.tight_layout()
    p = os.path.join(ac.ACML_FIG, "baselines_extended_freerun.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def main(fds=DATASETS):
    rows = []
    for fd in fds:
        print("=" * 70)
        print(f"EXTENDED BASELINES (+TCN)  FD00{fd}")
        print("=" * 70)
        rows.extend(evaluate(fd))
    df = pd.DataFrame(rows)
    csv = os.path.join(ac.ACML_TAB, "baselines_extended.csv")
    df.to_csv(csv, index=False)
    tex_cols = ["dataset", "model", "rho", "freerun_growth", "bounded",
                "nrmse_h1", "nrmse_h50"]
    tex = ac.latex_table(df[tex_cols], "Extended rollout baselines including a "
                         "TCN forecaster. Neural models can be competitive at "
                         "short horizons but closed-loop boundedness is "
                         "inconsistent; the bounded manifold stays bounded "
                         "everywhere.", "tab:baselines_extended")
    with open(os.path.join(ac.ACML_TAB, "baselines_extended.tex"), "w", encoding="utf-8") as fh:
        fh.write(tex)
    grid = _plot_grid(df)
    print("\n" + df[tex_cols].to_string(index=False))
    print(f"\nsaved -> {csv}\nsaved -> {grid}")
    return df


if __name__ == "__main__":
    fds = [int(x) for x in sys.argv[1:]] or DATASETS
    main(fds)
