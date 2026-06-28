"""
RESEARCH EXPERIMENT E2 + E3 -- COMPONENT ABLATIONS
==================================================

Questions (needed before publication):
  E2  What happens if we REMOVE regime normalisation?
  E3  What happens if we REMOVE monotonicity / smoothness / both?

For each dataset and each variant we retrain the manifold (same architecture,
k=2) through the configurable core and measure:
  * recon_mean_r2   held-out reconstruction quality (informative sensors)
  * mono_viol_frac  fraction of test health steps with dh0 < 0 (non-monotone)
  * rul_rmse / rul_r2 / rul_nasa  official-test RUL via the robust health->RUL map
  * base_rmse       mean-RUL baseline (reference)

Variants
  full            normalize=on,  mono=5, smooth=2   (validated pipeline)
  no_regime_norm  normalize=off (global standardisation)
  no_mono         mono=0
  no_smooth       smooth=0
  no_mono_smooth  mono=0, smooth=0

Usage:  python exp_ablation.py [fd ...]        (default: 1 2 3 4)

Outputs
  results/tables/research/ablation/ablation_summary.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import manifold as mc
import manifold.train as mtrain
from exp_rul_prediction import (MIN_VEL, RUL_CAP, SEED, VEL_WINDOW,
                                health_features, score)

ABLATION_EPOCHS = 2000     # bounded sweep budget; relative comparisons hold

VARIANTS = {
    "full":           dict(),
    "no_regime_norm": dict(normalize=False),
    "no_mono":        dict(lambda_mono=0.0),
    "no_smooth":      dict(lambda_smooth=0.0),
    "no_mono_smooth": dict(lambda_mono=0.0, lambda_smooth=0.0),
}


def recon_mean_r2(man, te_den) -> float:
    h = man.encode(te_den)
    recon = man.decode(h)
    idx = [mc.DYNAMIC.index(s) for s in mc.INFORMATIVE]
    true = te_den[mc.INFORMATIVE].to_numpy()
    return float(np.mean([mc.r2_pooled(true[:, j], recon[:, k])
                          for j, k in enumerate(idx)]))


def mono_violation_fraction(man, te_den) -> float:
    H = mc.per_engine_health(man, te_den)
    viols, total = 0, 0
    for _, g in H.sort_values("cycle").groupby("unit_id"):
        dh = np.diff(g["h0"].to_numpy())
        viols += int((dh < 0).sum())
        total += len(dh)
    return float(viols / max(total, 1))


def rul_metrics(man) -> dict:
    train = mc.load_split("train")
    train_den = mc.denoise(train, causal=True)
    Ftr = health_features(man, train_den)
    maxc = Ftr.groupby("unit_id")["cycle"].transform("max")
    Ftr["rul"] = np.minimum(maxc - Ftr["cycle"], RUL_CAP)
    feat = ["h0", "h1", "v0", "v1"]
    reg = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                        max_iter=400, l2_regularization=1.0,
                                        random_state=SEED)
    reg.fit(Ftr[feat].to_numpy(), Ftr["rul"].to_numpy())

    test = mc.load_split("test")
    test_den = mc.denoise(test, causal=True)
    Fte = health_features(man, test_den)
    rul_true = mc.load_rul()["rul"].to_numpy()
    units = sorted(Fte["unit_id"].unique().tolist())
    last = (Fte.sort_values("cycle").groupby("unit_id").tail(1)
            .set_index("unit_id").loc[units])
    pred = np.clip(reg.predict(last[feat].to_numpy()), 0, RUL_CAP)
    base = np.full_like(rul_true, float(Ftr["rul"].mean()), dtype=float)
    m = score(rul_true, pred)
    mb = score(rul_true, base)
    return dict(rul_rmse=m["RMSE"], rul_r2=m["R2"], rul_nasa=m["NASA"],
                base_rmse=mb["RMSE"])


def run_variant(fd: int, name: str, overrides: dict) -> dict:
    mc.configure(fd, retrain=True, k=2, tag=f"abl_{name}", **overrides)
    man = mc.get_manifold()
    test = mc.load_split("test")
    te_den = mc.denoise(test)
    row = dict(dataset=f"FD00{fd}", variant=name,
               n_regimes=mc.discovery_info()["n_regimes"],
               recon_mean_r2=recon_mean_r2(man, te_den),
               mono_viol_frac=mono_violation_fraction(man, te_den))
    row.update(rul_metrics(man))
    print(f"  FD00{fd:>1} {name:<15} recon={row['recon_mean_r2']:.3f}  "
          f"mono_viol={row['mono_viol_frac']:.3f}  "
          f"RUL RMSE={row['rul_rmse']:.2f}  R2={row['rul_r2']:+.3f}")
    return row


def main(fds):
    mtrain.EPOCHS = ABLATION_EPOCHS
    out_dir = os.path.join(ROOT, "results", "tables", "research", "ablation")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for fd in fds:
        print("=" * 78)
        print(f"ABLATIONS  FD00{fd}  (epochs={ABLATION_EPOCHS})")
        print("=" * 78)
        for name, ov in VARIANTS.items():
            rows.append(run_variant(fd, name, ov))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "ablation_summary.csv"), index=False)
    print("\n" + "=" * 78)
    print("ABLATION SUMMARY")
    print("=" * 78)
    print(df.to_string(index=False))
    print(f"\nsaved -> {os.path.join(out_dir, 'ablation_summary.csv')}")
    return df


if __name__ == "__main__":
    fds = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 4]
    main(fds)
