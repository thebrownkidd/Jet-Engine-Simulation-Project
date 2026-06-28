"""
RESEARCH EXPERIMENT E1b -- UNIFIED ALL-FILES MODEL (ONE MODEL, HARDCODED REGIMES)
================================================================================

Question: can a SINGLE health-manifold model serve all four C-MAPSS files at
once, if we (a) give it enough latent capacity (k=4, from E1a) and (b) use the
hardcoded operating-regime counts (1/6/1/6 for FD001-FD004)?

Design
------
1. For each dataset, fit its OWN operating-condition normaliser with the
   hardcoded heuristic regime count (regime_rule='heuristic', the default);
   this puts every dataset's degradation residual on a common, regime-free
   scale.
2. Restrict to the sensor channels that are degradation-bearing in ALL four
   datasets (intersection of the per-dataset dynamic sets) so the shared
   encoder input is meaningful everywhere.
3. Pool the per-dataset condition-normalised, denoised TRAIN trajectories and
   train ONE HealthAE (k=4) with the same monotonicity + smoothness penalties,
   using a per-(dataset,engine) monotonic mask.
4. Train ONE health->RUL head (gradient-boosted trees) on the pooled causal
   health features with per-file RUL caps.
5. Evaluate PER FILE on the official test sets: reconstruction, RUL, and
   rollout boundedness.  Compare to the per-file specialised k=2 models.

Usage:  python exp_unified_model.py

Outputs
  results/tables/research/unified/unified_per_file.csv
  results/figures/research/unified/unified_health_FD00<fd>.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import manifold as mc
from manifold.model import HealthAE, Manifold
from exp_rul_prediction import RUL_CAP, SEED, VEL_WINDOW, health_features, score

UNIFIED_K = 4
EPOCHS = 3000
LR = 5e-3
LAMBDA_MONO = 5.0
LAMBDA_SMOOTH = 2.0
FDS = (1, 2, 3, 4)


def _prepare_dataset(fd: int):
    """Configure with hardcoded regimes; return denoised train/test (ALL sensors)."""
    mc.configure(fd)
    info = mc.discovery_info()
    train = mc.load_split("train")
    test = mc.load_split("test")
    tr_den = mc.denoise(train, cols=mc.ALL_SENSORS)
    te_den = mc.denoise(test, cols=mc.ALL_SENSORS, causal=False)
    te_causal = mc.denoise(test, cols=mc.ALL_SENSORS, causal=True)
    tr_causal = mc.denoise(train, cols=mc.ALL_SENSORS, causal=True)
    return dict(fd=fd, n_regimes=info["n_regimes"], dynamic=set(mc.DYNAMIC),
                informative=set(mc.INFORMATIVE), tr_den=tr_den, te_den=te_den,
                te_causal=te_causal, tr_causal=tr_causal)


def _same_engine_mask(df, gid_col="gid"):
    gid = df[gid_col].to_numpy()
    return gid[:-1] == gid[1:]


def _train_unified(pooled_tr, common_dyn):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    mu = pooled_tr[common_dyn].mean().to_numpy()
    sd = pooled_tr[common_dyn].std().to_numpy() + 1e-12
    x = ((pooled_tr[common_dyn].to_numpy() - mu) / sd).astype(np.float32)
    xt = torch.tensor(x)
    mask = torch.tensor(_same_engine_mask(pooled_tr))

    model = HealthAE(len(common_dyn), UNIFIED_K)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(EPOCHS):
        opt.zero_grad()
        recon, h = model(xt)
        rec = ((recon - xt) ** 2).mean()
        h0 = h[:, 0]
        dh = h0[1:] - h0[:-1]
        mono = torch.relu(-dh)[mask].mean()
        smooth = (dh[mask] ** 2).mean()
        loss = rec + LAMBDA_MONO * mono + LAMBDA_SMOOTH * smooth
        loss.backward()
        opt.step()
        if (ep + 1) % 1000 == 0:
            print(f"    epoch {ep+1}/{EPOCHS}  loss={loss.item():.4f}  rec={rec.item():.4f}")
    model.eval()

    with torch.no_grad():
        h0 = model.encode(xt).numpy()[:, 0]
    corr = np.corrcoef(h0, pooled_tr["cycle"].to_numpy())[0, 1]
    flip0 = bool(corr < 0)
    return Manifold(model=model, mu=mu, sd=sd, flip0=flip0,
                    dynamic=list(common_dyn), k=UNIFIED_K), mu, sd


def main():
    tab_dir = os.path.join(ROOT, "results", "tables", "research", "unified")
    fig_dir = os.path.join(ROOT, "results", "figures", "research", "unified")
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 78)
    print("E1b  UNIFIED ALL-FILES MODEL  (k=4, hardcoded regimes)")
    print("=" * 78)

    data = {fd: _prepare_dataset(fd) for fd in FDS}
    common_dyn = sorted(set.intersection(*[data[fd]["dynamic"] for fd in FDS]),
                        key=lambda s: int(s[1:]))
    print(f"common degradation channels across all files: {common_dyn}")
    for fd in FDS:
        print(f"  FD00{fd}: hardcoded regimes = {data[fd]['n_regimes']}")

    # ---- pool train trajectories with a global engine id ---------------- #
    pooled = []
    for fd in FDS:
        d = data[fd]["tr_den"].copy()
        d["gid"] = fd * 100000 + d["unit_id"]
        pooled.append(d.sort_values(["gid", "cycle"]))
    pooled_tr = pd.concat(pooled, ignore_index=True).sort_values(["gid", "cycle"]).reset_index(drop=True)
    print(f"pooled training rows: {len(pooled_tr):,}")

    man, mu, sd = _train_unified(pooled_tr, common_dyn)

    # ---- one unified RUL head on pooled causal health features ---------- #
    feat_cols = ["h0", "h1", "v0", "v1"]
    Ftr_all = []
    for fd in FDS:
        # reconfigure normaliser context for this dataset's causal denoise frame
        mc.configure(fd)
        Ftr = health_features(man, data[fd]["tr_causal"])
        maxc = Ftr.groupby("unit_id")["cycle"].transform("max")
        Ftr["rul"] = np.minimum(maxc - Ftr["cycle"], RUL_CAP)
        Ftr["fd"] = fd
        Ftr_all.append(Ftr)
    Ftr_all = pd.concat(Ftr_all, ignore_index=True)
    reg = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                        max_iter=400, l2_regularization=1.0,
                                        random_state=SEED)
    reg.fit(Ftr_all[feat_cols].to_numpy(), Ftr_all["rul"].to_numpy())
    base_rul = float(Ftr_all["rul"].mean())

    # ---- per-file evaluation -------------------------------------------- #
    rows = []
    for fd in FDS:
        d = data[fd]
        inf_common = [s for s in common_dyn if s in d["informative"]]
        # reconstruction on this file's held-out test (informative ∩ common)
        h = man.encode(d["te_den"])
        recon = man.decode(h)
        idx = [common_dyn.index(s) for s in inf_common]
        true = d["te_den"][inf_common].to_numpy()
        recon_r2 = float(np.mean([mc.r2_pooled(true[:, j], recon[:, k])
                                  for j, k in enumerate(idx)]))

        # RUL on official test
        mc.configure(fd)
        Fte = health_features(man, d["te_causal"])
        rul_true = mc.load_rul()["rul"].to_numpy()
        units = sorted(Fte["unit_id"].unique().tolist())
        last = (Fte.sort_values("cycle").groupby("unit_id").tail(1)
                .set_index("unit_id").loc[units])
        pred = np.clip(reg.predict(last[feat_cols].to_numpy()), 0, RUL_CAP)
        m = score(rul_true, pred)
        base = score(rul_true, np.full_like(rul_true, base_rul, dtype=float))

        # rollout boundedness (free-run latent norm via decode)
        H = man.encode(d["te_den"])
        Hdf = d["te_den"][["unit_id", "cycle"]].copy()
        Hdf["h0"] = H[:, 0]
        longest = Hdf["unit_id"].value_counts().idxmax()
        gl = Hdf[Hdf["unit_id"] == longest].sort_values("cycle")
        h0_series = gl["h0"].to_numpy()
        # constant-velocity free-run of full latent, decode, measure norm growth
        hist = man.encode(d["te_den"][d["te_den"]["unit_id"] == longest]
                          .sort_values("cycle"))
        c0 = int(0.4 * len(hist))
        w = min(VEL_WINDOW, len(hist[:c0 + 1]) - 1)
        recent = hist[:c0 + 1][-w - 1:]
        tt = np.arange(len(recent))
        v = np.array([np.polyfit(tt, recent[:, j], 1)[0] for j in range(UNIFIED_K)])
        future = np.array([hist[c0] + v * (s + 1) for s in range(400)])
        future[:, 0] = np.clip(future[:, 0], 0.0, 1.5)
        dec = man.decode(future)
        norm0 = np.linalg.norm((dec[0] - mu) / sd)
        norm400 = np.linalg.norm((dec[-1] - mu) / sd)
        growth = float(norm400 / (norm0 + 1e-12))

        row = dict(dataset=f"FD00{fd}", n_regimes=d["n_regimes"],
                   recon_mean_r2=recon_r2, rul_rmse=m["RMSE"], rul_r2=m["R2"],
                   rul_nasa=m["NASA"], base_rmse=base["RMSE"],
                   freerun_growth=growth, bounded=bool(growth < 5.0))
        rows.append(row)
        print(f"  FD00{fd}: recon={recon_r2:.3f}  RUL RMSE={m['RMSE']:.2f} "
              f"R2={m['R2']:+.3f}  (base {base['RMSE']:.2f})  "
              f"freerun_growth=x{growth:.2f}  bounded={row['bounded']}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(tab_dir, "unified_per_file.csv"), index=False)
    print("\n" + "=" * 78)
    print("UNIFIED MODEL -- PER-FILE RESULTS")
    print("=" * 78)
    print(df.to_string(index=False))
    print(f"\nsaved -> {os.path.join(tab_dir, 'unified_per_file.csv')}")
    return df


if __name__ == "__main__":
    main()
