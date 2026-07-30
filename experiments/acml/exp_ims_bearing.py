"""
IMS bearing external-dataset validation of the bounded-latent manifold method.

Goal
----
Test whether the *same algorithmic claims* established on C-MAPSS reproduce on an
independent run-to-failure dataset (NASA/IMS bearings), namely:

  C1. A compact (k=2) latent manifold reconstructs the multivariate feature
      stream well (recon R2 high).
  C2. Bounded latent constant-velocity rollout stays bounded, while a
      sensor-space VAR rollout diverges (spectral radius rho>1).
  C3. The learned primary health coordinate is (near-)monotone with time.
  C4. Latent dimension K acts as a regularizer: reconstruction rises with K but
      stability / prognosis need not improve.
  C5. Downstream RUL utility beats a mean-RUL baseline.

This script is SELF-CONTAINED: it reuses only the dataset-agnostic FlexAE model
from acml_common and does NOT depend on the C-MAPSS-specific `manifold` context
(regimes, parquet FD loaders, informative-sensor globals).

Inputs
------
  data/processed/ims_bearing_features.csv   (from ims_prep.py)

Outputs
-------
  results/acml/tables/ims_bearing_summary.csv (+ .tex)
  results/acml/tables/ims_k_sweep.csv (+ .tex)
  results/acml/figures/ims_bearing_summary.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))

from acml_common import FlexAE  # dataset-agnostic autoencoder

DEFAULT_FEAT_CSV = os.path.join(ROOT, "data", "processed", "ims_bearing_features.csv")
TAB = os.path.join(ROOT, "results", "acml", "tables")
FIG = os.path.join(ROOT, "results", "acml", "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# Constants mirrored from the C-MAPSS ACML pipeline for comparability.
LR = 5e-3
EPOCHS = 1500
VEL_WINDOW = 25
ROLLOUT_VEL_WINDOW = 20
FREE_STEPS = 400
CUTOFF_FRAC = 0.40
SCORE_H = [1, 10, 25, 50]
BOUNDED_GROWTH_THRESH = 5.0
RUL_CAP = 125
SEED = 42
TREND_THRESH = 0.30  # |corr(feature, cycle)| to be "trend-bearing"


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def load_features(feat_csv: str) -> tuple[pd.DataFrame, list[str]]:
    if not os.path.exists(feat_csv):
        print(f"ERROR: {feat_csv} not found. Run the *_prep.py step first.")
        sys.exit(1)
    df = pd.read_csv(feat_csv)
    feats = [c for c in df.columns if c not in ("unit_id", "cycle")]
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    return df, feats


def denoise(df: pd.DataFrame, cols: list[str], win: int = 15) -> pd.DataFrame:
    """Causal rolling-median denoise per unit (matches pipeline preprocessing)."""
    out = df.copy()
    for c in cols:
        out[c] = out.groupby("unit_id")[c].transform(
            lambda v: v.rolling(win, min_periods=1).median())
    return out


def select_trend_features(df: pd.DataFrame, feats: list[str]) -> list[str]:
    """Keep features with mean |corr vs cycle| above threshold (trend-bearing)."""
    scores = {}
    for c in feats:
        cs = []
        for _, g in df.groupby("unit_id"):
            if g[c].std() > 1e-9 and len(g) > 5:
                cs.append(abs(np.corrcoef(g[c], g["cycle"])[0, 1]))
        scores[c] = float(np.nanmean(cs)) if cs else 0.0
    chosen = [c for c in feats if scores[c] >= TREND_THRESH]
    if len(chosen) < 3:  # fall back to top-5 by trend score
        chosen = sorted(feats, key=lambda c: scores[c], reverse=True)[:5]
    return chosen


def split_units(df: pd.DataFrame, test_frac: float = 0.35):
    units = sorted(df["unit_id"].unique().tolist())
    rng = np.random.RandomState(SEED)
    rng.shuffle(units)
    n_te = max(1, int(round(test_frac * len(units))))
    te_u = set(units[:n_te])
    tr = df[~df["unit_id"].isin(te_u)].reset_index(drop=True)
    te = df[df["unit_id"].isin(te_u)].reset_index(drop=True)
    return tr, te


# --------------------------------------------------------------------------- #
# Model training / encoding (self-contained; monotone+smooth penalties on h0)
# --------------------------------------------------------------------------- #
def train_ae(tr: pd.DataFrame, cols: list[str], *, k: int, bounded: bool,
             lambda_mono: float, lambda_smooth: float,
             epochs: int = EPOCHS, seed: int = SEED):
    mu = tr[cols].mean().to_numpy()
    sd = tr[cols].std().to_numpy() + 1e-12
    x = ((tr[cols].to_numpy() - mu) / sd).astype(np.float32)
    xt = torch.tensor(x)
    uid = tr["unit_id"].to_numpy()
    mask = torch.tensor(uid[:-1] == uid[1:])

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FlexAE(len(cols), k, bounded=bounded)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        opt.zero_grad()
        recon, h = model(xt)
        rec = ((recon - xt) ** 2).mean()
        h0 = h[:, 0]
        dh = h0[1:] - h0[:-1]
        mono = torch.relu(-dh)[mask].mean()
        smooth = (dh[mask] ** 2).mean()
        loss = rec + lambda_mono * mono + lambda_smooth * smooth
        loss.backward()
        opt.step()
    model.eval()

    with torch.no_grad():
        h0 = model.encode(xt).numpy()[:, 0]
    flip0 = bool(np.corrcoef(h0, tr["cycle"].to_numpy())[0, 1] < 0)
    return dict(model=model, mu=mu, sd=sd, flip0=flip0, cols=cols, k=k,
                bounded=bounded)


def encode(ae, df: pd.DataFrame) -> np.ndarray:
    x = ((df[ae["cols"]].to_numpy() - ae["mu"]) / ae["sd"]).astype(np.float32)
    with torch.no_grad():
        h = ae["model"].encode(torch.tensor(x)).numpy()
    if ae["flip0"]:
        h = h.copy()
        h[:, 0] = (1.0 - h[:, 0]) if ae["bounded"] else (-h[:, 0])
    return h


def decode(ae, h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, np.float32).reshape(-1, ae["k"])
    if ae["flip0"]:
        h = h.copy()
        h[:, 0] = (1.0 - h[:, 0]) if ae["bounded"] else (-h[:, 0])
    with torch.no_grad():
        x = ae["model"].dec(torch.tensor(h)).numpy()
    return x * ae["sd"] + ae["mu"]


def rollout(ae, h_hist: np.ndarray, steps: int, projection: str) -> np.ndarray:
    k = ae["k"]
    w = min(ROLLOUT_VEL_WINDOW, len(h_hist) - 1)
    if w < 1:
        v = np.zeros(k)
    else:
        recent = h_hist[-w - 1:]
        t = np.arange(len(recent))
        v = np.array([np.polyfit(t, recent[:, j], 1)[0] for j in range(k)])
    h0 = h_hist[-1]
    future = np.array([h0 + v * (s + 1) for s in range(steps)])
    if projection == "full_box":
        future = np.clip(future, 0.0, 1.0)
    elif projection == "h0_clip":
        future[:, 0] = np.clip(future[:, 0], 0.0, 1.5)
    return future, decode(ae, future)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def r2_pooled(y, p):
    y, p = np.asarray(y), np.asarray(p)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def recon_r2(ae, te: pd.DataFrame) -> tuple[float, float]:
    h = encode(ae, te)
    rec = decode(ae, h)
    true = te[ae["cols"]].to_numpy()
    r2s = [r2_pooled(true[:, j], rec[:, j]) for j in range(len(ae["cols"]))]
    return float(np.mean(r2s)), float(np.min(r2s))


def curvature_kappa(ae, te: pd.DataFrame) -> float:
    diffs = []
    for _, g in te.groupby("unit_id"):
        h0 = encode(ae, g)[:, 0]
        if len(h0) >= 3:
            diffs.append(np.abs(np.diff(h0, 2)))
    return float(np.median(np.concatenate(diffs))) if diffs else float("nan")


def mono_violation(ae, te: pd.DataFrame) -> float:
    v, t = 0, 0
    for _, g in te.groupby("unit_id"):
        dh = np.diff(encode(ae, g)[:, 0])
        v += int((dh < 0).sum()); t += len(dh)
    return float(v / max(t, 1))


def longest_unit(te: pd.DataFrame) -> pd.DataFrame:
    u = max(te["unit_id"].unique(), key=lambda k: (te["unit_id"] == k).sum())
    return te[te["unit_id"] == u].sort_values("cycle").reset_index(drop=True)


def decoded_freerun(ae, tr: pd.DataFrame, te: pd.DataFrame, projection: str):
    mu = tr[ae["cols"]].mean().to_numpy()
    sd = tr[ae["cols"]].std().to_numpy() + 1e-12
    g = longest_unit(te)
    c0 = int(CUTOFF_FRAC * len(g))
    h_hist = encode(ae, g.iloc[:c0 + 1])
    _, roll = rollout(ae, h_hist, FREE_STEPS, projection)
    norms = np.array([np.linalg.norm((roll[s] - mu) / sd) for s in range(FREE_STEPS)])
    growth = float(norms[-1] / (norms[0] + 1e-12))
    return float(norms[-1]), growth, bool(growth < BOUNDED_GROWTH_THRESH)


def fit_var(tr: pd.DataFrame, cols: list[str]):
    mu = tr[cols].mean().to_numpy()
    sd = tr[cols].std().to_numpy() + 1e-12
    Xs, Ys = [], []
    for _, g in tr.groupby("unit_id"):
        z = (g[cols].to_numpy() - mu) / sd
        if len(z) > 1:
            Xs.append(z[:-1]); Ys.append(z[1:])
    reg = LinearRegression().fit(np.vstack(Xs), np.vstack(Ys))
    A = reg.coef_
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    return reg, mu, sd, rho


def var_freerun(reg, mu, sd, te: pd.DataFrame, cols: list[str]):
    g = longest_unit(te)
    c0 = int(CUTOFF_FRAC * len(g))
    z = (g[cols].to_numpy()[c0] - mu) / sd
    norms = []
    for _ in range(FREE_STEPS):
        z = reg.predict(z.reshape(1, -1))[0]
        norms.append(float(np.linalg.norm(z)))
    norms = np.array(norms)
    growth = float(norms[-1] / (norms[0] + 1e-12))
    return float(norms[-1]), growth, bool(growth < BOUNDED_GROWTH_THRESH)


def kaware_features(ae, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in df.sort_values("cycle").groupby("unit_id"):
        g = g.copy()
        h = encode(ae, g)
        for j in range(ae["k"]):
            g[f"h{j}"] = h[:, j]
        for j in range(ae["k"]):
            y = g[f"h{j}"].to_numpy()
            v = np.zeros(len(y))
            for t in range(len(y)):
                lo = max(0, t - VEL_WINDOW + 1)
                seg = y[lo:t + 1]
                if len(seg) >= 2:
                    tau = np.arange(len(seg)) - (len(seg) - 1) / 2
                    v[t] = float((tau * (seg - seg.mean())).sum() / (tau ** 2).sum())
            g[f"v{j}"] = v
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def rul_utility(ae, tr: pd.DataFrame, te: pd.DataFrame) -> dict:
    """RUL = remaining snapshots to end-of-run (run-to-failure), capped."""
    feat = [f"h{j}" for j in range(ae["k"])] + [f"v{j}" for j in range(ae["k"])]
    Ftr = kaware_features(ae, tr)
    maxc = Ftr.groupby("unit_id")["cycle"].transform("max")
    Ftr["rul"] = np.minimum(maxc - Ftr["cycle"], RUL_CAP)
    reg = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                        max_iter=400, l2_regularization=1.0,
                                        random_state=SEED)
    reg.fit(Ftr[feat].to_numpy(), Ftr["rul"].to_numpy())

    Fte = kaware_features(ae, te)
    maxc_te = Fte.groupby("unit_id")["cycle"].transform("max")
    Fte["rul"] = np.minimum(maxc_te - Fte["cycle"], RUL_CAP)
    pred = np.clip(reg.predict(Fte[feat].to_numpy()), 0, RUL_CAP)
    err = pred - Fte["rul"].to_numpy()
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r2 = r2_pooled(Fte["rul"].to_numpy(), pred)
    base = np.full_like(Fte["rul"].to_numpy(), float(Ftr["rul"].mean()), dtype=float)
    base_rmse = float(np.sqrt(np.mean((base - Fte["rul"].to_numpy()) ** 2)))
    return dict(rul_rmse=rmse, rul_r2=r2, base_rmse=base_rmse)


def latex_table(df: pd.DataFrame, caption: str, label: str, fmt="%.3f") -> str:
    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(str(c).replace("_", r"\_") for c in cols) + r" \\",
             r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("nan" if v != v else (fmt % v))
            else:
                cells.append(str(v).replace("_", r"\_"))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--features", type=str, default=DEFAULT_FEAT_CSV,
                    help="long-format features CSV (unit_id, cycle, <features>)")
    ap.add_argument("--name", type=str, default="ims_bearing",
                    help="dataset tag used for output filenames/titles")
    ap.add_argument("--test-frac", type=float, default=0.35)
    args = ap.parse_args()
    name = args.name

    print(f"{name} — external validation of bounded-latent manifold")
    print("=" * 66)
    df, all_feats = load_features(args.features)
    print(f"loaded {len(df)} rows, {df['unit_id'].nunique()} units, "
          f"{len(all_feats)} raw features")

    cols = select_trend_features(denoise(df, all_feats), all_feats)
    print(f"trend-bearing features ({len(cols)}): {cols}")

    dfd = denoise(df, cols)
    tr, te = split_units(dfd, test_frac=args.test_frac)
    print(f"train units: {tr['unit_id'].nunique()}  test units: {te['unit_id'].nunique()}")

    # --- Main comparison at k=2: bounded manifold vs unbounded AE vs VAR --- #
    print("\n[1/3] training k=2 bounded + unbounded autoencoders ...")
    ae_b = train_ae(tr, cols, k=2, bounded=True, lambda_mono=1.0,
                    lambda_smooth=0.5, epochs=args.epochs)
    ae_u = train_ae(tr, cols, k=2, bounded=False, lambda_mono=1.0,
                    lambda_smooth=0.5, epochs=args.epochs)

    reg, vmu, vsd, rho = fit_var(tr, cols)

    rows = []
    for mname, ae, proj in [("manifold_bounded", ae_b, "full_box"),
                            ("ae_unbounded", ae_u, "none")]:
        rmean, rmin = recon_r2(ae, te)
        _, growth, bnd = decoded_freerun(ae, tr, te, proj)
        rows.append(dict(model=mname, rho=np.nan, recon_mean_r2=rmean,
                         recon_min_r2=rmin, freerun_growth=growth, bounded=bnd,
                         kappa=curvature_kappa(ae, te),
                         mono_viol=mono_violation(ae, te)))
    _, vgrowth, vbnd = var_freerun(reg, vmu, vsd, te, cols)
    rows.append(dict(model="var_sensor", rho=rho, recon_mean_r2=np.nan,
                     recon_min_r2=np.nan, freerun_growth=vgrowth, bounded=vbnd,
                     kappa=np.nan, mono_viol=np.nan))
    summary = pd.DataFrame(rows)

    # --- RUL utility (bounded manifold) --- #
    print("[2/3] RUL utility (bounded manifold) ...")
    rul = rul_utility(ae_b, tr, te)
    summary.loc[summary["model"] == "manifold_bounded", "rul_rmse"] = rul["rul_rmse"]
    summary.loc[summary["model"] == "manifold_bounded", "rul_r2"] = rul["rul_r2"]
    summary.loc[summary["model"] == "manifold_bounded", "base_rmse"] = rul["base_rmse"]

    summary.to_csv(os.path.join(TAB, f"{name}_summary.csv"), index=False)
    with open(os.path.join(TAB, f"{name}_summary.tex"), "w") as fh:
        fh.write(latex_table(summary.round(4),
                 f"{name} external validation: bounded latent manifold vs "
                 "unbounded AE vs sensor-space VAR.", f"tab:{name}"))

    # --- K-sweep --- #
    print("[3/3] K-aware dimension sweep ...")
    ks = list(range(1, args.kmax + 1))
    krows = []
    for k in ks:
        ae_k = train_ae(tr, cols, k=k, bounded=True, lambda_mono=1.0,
                        lambda_smooth=0.5, epochs=args.epochs)
        rmean, rmin = recon_r2(ae_k, te)
        _, growth, bnd = decoded_freerun(ae_k, tr, te, "full_box")
        rk = rul_utility(ae_k, tr, te)
        krows.append(dict(k=k, recon_mean_r2=rmean, recon_min_r2=rmin,
                          freerun_growth=growth, freerun_bounded=bnd,
                          kappa=curvature_kappa(ae_k, te),
                          rul_rmse=rk["rul_rmse"], rul_r2=rk["rul_r2"]))
        print(f"    k={k}: recon={rmean:.3f} growth={growth:.2f} "
              f"bounded={bnd} rul_rmse={rk['rul_rmse']:.2f}")
    ksweep = pd.DataFrame(krows)
    ksweep.to_csv(os.path.join(TAB, f"{name}_k_sweep.csv"), index=False)
    with open(os.path.join(TAB, f"{name}_k_sweep.tex"), "w") as fh:
        fh.write(latex_table(ksweep.round(4),
                 f"{name} latent-dimension sweep: reconstruction rises with "
                 "K while stability/prognosis need not.", f"tab:{name}_ksweep"))

    # --- Figure --- #
    make_figure(ae_b, tr, te, cols, reg, vmu, vsd, ksweep, name)

    # --- Console verdict --- #
    print("\n" + "=" * 66)
    print(f"CLAIM REPRODUCTION ON {name.upper()}")
    mb = summary[summary.model == "manifold_bounded"].iloc[0]
    vv = summary[summary.model == "var_sensor"].iloc[0]
    print(f"  C1 recon R2 (bounded k=2) : {mb.recon_mean_r2:.3f}")
    print(f"  C2 bounded growth={mb.freerun_growth:.2f} (bounded={mb.bounded}) "
          f"vs VAR rho={vv.rho:.3f} growth={vv.freerun_growth:.1f} "
          f"(bounded={vv.bounded})")
    print(f"  C3 mono-violation frac    : {mb.mono_viol:.3f}")
    print(f"  C4 recon(k=1..K)          : "
          f"{ksweep['recon_mean_r2'].round(3).tolist()}")
    print(f"  C5 RUL RMSE {mb.rul_rmse:.2f} vs mean-baseline {mb.base_rmse:.2f}")
    print("=" * 66)
    print(f"wrote tables -> {TAB}")
    print(f"wrote figure -> {os.path.join(FIG, f'{name}_summary.png')}")


def make_figure(ae, tr, te, cols, reg, vmu, vsd, ksweep, name="ims_bearing"):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    # (a) health trajectories
    for _, g in te.groupby("unit_id"):
        g = g.sort_values("cycle")
        ax[0, 0].plot(g["cycle"], encode(ae, g)[:, 0], alpha=0.8)
    ax[0, 0].set_title("(a) Learned health h0 vs time (test units)")
    ax[0, 0].set_xlabel("snapshot"); ax[0, 0].set_ylabel("h0")

    # (b) bounded latent vs VAR free-run norm
    g = longest_unit(te)
    c0 = int(CUTOFF_FRAC * len(g))
    h_hist = encode(ae, g.iloc[:c0 + 1])
    _, roll = rollout(ae, h_hist, FREE_STEPS, "full_box")
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

    # (c) recon vs K
    ax[1, 0].plot(ksweep["k"], ksweep["recon_mean_r2"], "o-", label="recon R2")
    ax[1, 0].set_ylabel("recon R2"); ax[1, 0].set_xlabel("K")
    axb = ax[1, 0].twinx()
    axb.plot(ksweep["k"], ksweep["freerun_growth"], "s--", color="tab:red",
             label="growth")
    axb.axhline(BOUNDED_GROWTH_THRESH, color="gray", ls=":")
    axb.set_ylabel("free-run growth")
    ax[1, 0].set_title("(c) Reconstruction vs stability across K")

    # (d) RUL vs K
    ax[1, 1].plot(ksweep["k"], ksweep["rul_rmse"], "o-")
    ax[1, 1].set_title("(d) RUL RMSE vs K")
    ax[1, 1].set_xlabel("K"); ax[1, 1].set_ylabel("RUL RMSE")

    fig.suptitle(f"{name}: bounded-latent manifold reproduction")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{name}_summary.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
