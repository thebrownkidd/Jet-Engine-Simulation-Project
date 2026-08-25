"""
NASA Li-ion battery aging dataset -> long-format degradation feature table.

The NASA/PCoE battery dataset stores, per battery `.mat` file, a sequence of
charge / discharge / impedance operations. Battery health degrades as usable
Capacity (Ahr) fades over successive DISCHARGE cycles until end-of-life (EOL,
typically 70% of initial capacity). This is a *gradual, near-monotone*
degradation process (contrast with the abrupt IMS bearing failures), so it is a
useful second external test of the bounded-latent manifold method.

We convert each battery's discharge cycles into a long-format table:

    unit_id, cycle, <feature_1..feature_p>

where
  * unit_id = battery id (e.g. B0005)
  * cycle   = discharge-cycle index (chronological)
  * features = per-discharge-cycle summaries of the voltage / current /
    temperature curves plus the measured Capacity.

Output
------
  data/processed/battery_features.csv

Usage
-----
  python experiments/acml/battery_prep.py
  python experiments/acml/battery_prep.py --min-cycles 40
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MAT_DIR = os.path.join(ROOT, "data", "raw_battery", "Battery_DataSet",
                       "Battery_DataSet")
META_CSV = os.path.join(ROOT, "data", "raw_battery", "metadata.csv")
OUT_CSV = os.path.join(ROOT, "data", "processed", "battery_features.csv")

MIN_STD = 1e-4


def _f(x) -> np.ndarray:
    """Flatten a MATLAB-loaded field to a 1-D float array."""
    return np.asarray(x, dtype=float).ravel()


def _discharge_features(d) -> dict | None:
    """Per-discharge-cycle features from the measured curves."""
    try:
        V = _f(d["Voltage_measured"])
        I = _f(d["Current_measured"])
        T = _f(d["Temperature_measured"])
        t = _f(d["Time"])
        cap = _f(d["Capacity"])
    except (KeyError, ValueError, IndexError):
        return None
    if V.size < 5 or cap.size < 1 or not np.isfinite(cap[0]):
        return None
    dur = float(t[-1] - t[0]) if t.size >= 2 else float("nan")
    # time spent below a mid voltage knee (proxy for usable plateau length)
    v_lo = 3.0
    below = t[V < v_lo]
    knee_time = float(below[0] - t[0]) if below.size else dur
    return dict(
        capacity=float(cap[0]),
        discharge_time=dur,
        v_mean=float(np.mean(V)), v_min=float(np.min(V)),
        v_start=float(V[0]), v_end=float(V[-1]),
        i_mean=float(np.mean(np.abs(I))),
        t_mean=float(np.mean(T)), t_max=float(np.max(T)),
        knee_time=knee_time,
    )


def process_battery(mat_path: str) -> pd.DataFrame:
    bid = os.path.splitext(os.path.basename(mat_path))[0]
    m = sio.loadmat(mat_path)
    if bid not in m:
        return pd.DataFrame()
    cycles = m[bid][0, 0]["cycle"][0]
    rows, k = [], 0
    for c in cycles:
        if str(c["type"][0]) != "discharge":
            continue
        feats = _discharge_features(c["data"][0, 0])
        if feats is None:
            continue
        feats["unit_id"] = bid
        feats["cycle"] = k
        rows.append(feats)
        k += 1
    return pd.DataFrame(rows)


def _load_conditions() -> dict:
    """Map battery_id -> operating-condition group from metadata.csv."""
    if not os.path.exists(META_CSV):
        return {}
    meta = pd.read_csv(META_CSV)
    if "battery_id" not in meta or "group" not in meta:
        return {}
    return dict(zip(meta["battery_id"].astype(str), meta["group"].astype(str)))


def condition_normalize(out: pd.DataFrame, feat_cols: list, cond_map: dict):
    """Per-operating-condition normalization (mirrors the C-MAPSS regime step).

    Batteries in the same group share ambient temperature and discharge protocol,
    so the group is the operating condition. Subtract per-condition mean and divide
    by pooled within-condition std to remove the condition offset, leaving the
    degradation residual. Near-constant features are dropped.
    """
    cond = out["unit_id"].map(lambda u: cond_map.get(str(u), "G0")).to_numpy()
    conds = sorted(np.unique(cond).tolist())
    n_cond = len(conds)

    good = []
    for f in feat_cols:
        vals = out[f].to_numpy().astype(float)
        pooled_var, total = 0.0, 0
        for c in conds:
            mask = cond == c
            if mask.sum() < 2:
                continue
            v = vals[mask]
            pooled_var += (v - v.mean()).var() * mask.sum()
            total += int(mask.sum())
        pooled_std = float(np.sqrt(pooled_var / max(total - n_cond, 1)))
        if pooled_std < MIN_STD:
            continue
        normed = vals.copy()
        for c in conds:
            mask = cond == c
            if mask.sum() < 1:
                continue
            normed[mask] = (normed[mask] - vals[mask].mean()) / pooled_std
        out[f] = normed
        good.append(f)

    for c in conds:
        out[f"ctx_{c}"] = (cond == c).astype(float)
    ctx_cols = [f"ctx_{c}" for c in conds]
    return out, good, ctx_cols, n_cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cycles", type=int, default=30,
                    help="drop batteries with fewer discharge cycles than this")
    args = ap.parse_args()

    print("NASA battery feature extraction")
    print("=" * 60)
    files = sorted(glob.glob(os.path.join(MAT_DIR, "B*.mat")))
    if not files:
        print(f"No .mat files under {MAT_DIR}")
        sys.exit(1)

    parts = []
    for fp in files:
        df = process_battery(fp)
        if df.empty:
            print(f"  [skip] {os.path.basename(fp)}: no usable discharge cycles")
            continue
        n = df["cycle"].max() + 1
        if n < args.min_cycles:
            print(f"  [skip] {df['unit_id'].iloc[0]}: only {n} cycles")
            continue
        parts.append(df)
        print(f"  {df['unit_id'].iloc[0]}: {n} discharge cycles")

    if not parts:
        print("No batteries met the minimum-cycle threshold; aborting.")
        sys.exit(1)

    out = pd.concat(parts, ignore_index=True)
    feat_cols = [c for c in out.columns if c not in ("unit_id", "cycle")]

    # Regime (operating-condition) identification + per-condition normalization.
    cond_map = _load_conditions()
    out, feat_cols, ctx_cols, n_cond = condition_normalize(out, feat_cols, cond_map)

    out = out[["unit_id", "cycle"] + feat_cols + ctx_cols].sort_values(
        ["unit_id", "cycle"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print("=" * 60)
    print(f"units: {out['unit_id'].nunique()}  rows: {len(out)}  "
          f"features: {len(feat_cols)}  conditions: {n_cond}  ctx: {len(ctx_cols)}")
    print(f"wrote -> {OUT_CSV}")


if __name__ == "__main__":
    main()
