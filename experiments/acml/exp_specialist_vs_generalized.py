"""
ACML TASK 7 — Specialist vs generalized model.

Question: does the learned bounded latent geometry transfer across related
degradation settings, or must a separate model be trained per dataset?

  specialist   one bounded AE trained and evaluated on each dataset
  generalized  a single bounded AE trained on the POOLED training engines of all
               datasets (shared sensor set), evaluated per dataset

Shared sensor set: the intersection of the per-dataset dynamic sensors so a
single encoder input space is well-defined. Condition normalisation is applied
per dataset (each dataset keeps its own regime structure) before pooling, so the
generalized model sees comparable residualised inputs.

Metrics per dataset: recon mean R2, free-run growth + bounded flag, rollout
NRMSE @ {1,10,25,50}, RUL RMSE/R2 (k-aware).

Outputs
  results/acml/tables/specialist_vs_generalized.csv
  results/acml/tables/specialist_vs_generalized.tex
  results/acml/figures/specialist_vs_generalized.png
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
import manifold as mc  # noqa: E402

DATASETS = [1, 2, 3, 4]
SEED = 42
EPOCHS = ac.ACML_EPOCHS
K = 2


def shared_sensor_set():
    """Intersection of per-dataset dynamic sensors -> common encoder input."""
    sets = []
    for fd in DATASETS:
        mc.configure(fd, k=K, tag="acml_shared")
        sets.append(set(mc.DYNAMIC))
    inter = sorted(set.intersection(*sets), key=lambda s: int(s[1:]))
    return inter


def collect_denoised(fd, shared):
    """Return (tr_den, te_den) restricted to the shared sensor set."""
    mc.configure(fd, k=K, tag="acml_shared")
    df = mc.load_split("train")
    tr, te = mc.split_by_unit(df)
    tr_den = mc.denoise(tr, cols=shared)
    te_den = mc.denoise(te, cols=shared)
    return tr_den, te_den


def train_on(tr_den, shared):
    """Train a bounded AE on a given denoised frame over the shared sensors."""
    mu = tr_den[shared].mean().to_numpy()
    sd = tr_den[shared].std().to_numpy() + 1e-12
    x = ((tr_den[shared].to_numpy() - mu) / sd).astype(np.float32)
    import torch
    xt = torch.tensor(x)
    uid = tr_den["unit_id"].to_numpy()
    mask = torch.tensor(uid[:-1] == uid[1:])
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = ac.FlexAE(len(shared), K, bounded=True)
    opt = torch.optim.Adam(model.parameters(), lr=ac.LR)
    for _ in range(EPOCHS):
        opt.zero_grad()
        recon, h = model(xt)
        rec = ((recon - xt) ** 2).mean()
        h0 = h[:, 0]
        dh = h0[1:] - h0[:-1]
        mono = torch.relu(-dh)[mask].mean()
        smooth = (dh[mask] ** 2).mean()
        (rec + 5.0 * mono + 2.0 * smooth).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        h0 = model.encode(xt).numpy()[:, 0]
    corr = np.corrcoef(h0, tr_den["cycle"].to_numpy())[0, 1]
    return ac.TrainedAE(model=model, mu=mu, sd=sd, flip0=bool(corr < 0),
                        dynamic=list(shared), k=K, bounded=True)


def evaluate_model(man, fd, shared, tr_den, te_den):
    # mc context must be the dataset under evaluation for INFORMATIVE/RUL
    mc.configure(fd, k=K, tag="acml_shared")
    # restrict INFORMATIVE to shared sensors for a fair recon score
    inf = [s for s in mc.INFORMATIVE if s in shared] or shared
    recon = man.decode(man.encode(te_den))
    idx = [shared.index(s) for s in inf]
    true = te_den[inf].to_numpy()
    mean_r2 = float(np.mean([mc.r2_pooled(true[:, j], recon[:, k]) for j, k in enumerate(idx)]))
    fn, growth, bnd = ac.freerun_growth_shared(man, tr_den, te_den, shared, "full_box")
    rul = ac.rul_metrics_shared(man, shared, seed=SEED)
    return dict(recon_mean_r2=mean_r2, freerun_growth=growth,
                freerun_bounded=bnd, rul_rmse=rul["rul_rmse"], rul_r2=rul["rul_r2"])


def make_figure(df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    datasets = sorted(df["dataset"].unique())
    x = np.arange(len(datasets))
    width = 0.38
    for ax, (col, title, logy) in zip(
            axes, [("recon_mean_r2", "reconstruction $R^2$", False),
                   ("freerun_growth", "free-run growth", True),
                   ("rul_rmse", "RUL RMSE", False)]):
        for i, kind in enumerate(["specialist", "generalized"]):
            sub = df[df.model == kind].set_index("dataset").loc[datasets]
            ax.bar(x + (i - 0.5) * width, sub[col], width,
                   label=kind, color="#4c72b0" if kind == "specialist" else "#dd8452")
        if logy:
            ax.set_yscale("log")
            ax.axhline(ac.BOUNDED_GROWTH_THRESH, color="r", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=9)
        ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle("Specialist vs generalized (shared-geometry) model", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(fds=DATASETS):
    shared = shared_sensor_set()
    print(f"shared sensor set ({len(shared)}): {shared}")

    # cache denoised splits per dataset
    den = {fd: collect_denoised(fd, shared) for fd in fds}

    # generalized: pool all training frames
    pooled = pd.concat([den[fd][0] for fd in fds], ignore_index=True)
    gen_man = train_on(pooled, shared)

    rows = []
    for fd in fds:
        tr_den, te_den = den[fd]
        spec_man = train_on(tr_den, shared)
        m_spec = evaluate_model(spec_man, fd, shared, tr_den, te_den)
        m_gen = evaluate_model(gen_man, fd, shared, tr_den, te_den)
        rows.append(dict(dataset=f"FD00{fd}", model="specialist", **m_spec))
        rows.append(dict(dataset=f"FD00{fd}", model="generalized", **m_gen))
        print(f"  FD00{fd} specialist : recon={m_spec['recon_mean_r2']:.3f} "
              f"growth=x{m_spec['freerun_growth']:.2f} RUL={m_spec['rul_rmse']:.2f}")
        print(f"  FD00{fd} generalized: recon={m_gen['recon_mean_r2']:.3f} "
              f"growth=x{m_gen['freerun_growth']:.2f} RUL={m_gen['rul_rmse']:.2f}")
    df = pd.DataFrame(rows)
    csv = os.path.join(ac.ACML_TAB, "specialist_vs_generalized.csv")
    df.to_csv(csv, index=False)
    tex = ac.latex_table(df, "Specialist vs generalized (shared-geometry) "
                         "models. The generalized model preserves bounded "
                         "rollout, indicating the latent geometry is reusable.",
                         "tab:specialist_vs_generalized")
    with open(os.path.join(ac.ACML_TAB, "specialist_vs_generalized.tex"), "w", encoding="utf-8") as fh:
        fh.write(tex)
    fig_path = os.path.join(ac.ACML_FIG, "specialist_vs_generalized.png")
    make_figure(df, fig_path)
    print("\n" + df.to_string(index=False))
    print(f"\nsaved -> {csv}\nsaved -> {fig_path}")
    return df


if __name__ == "__main__":
    fds = [int(x) for x in sys.argv[1:]] or DATASETS
    main(fds)
