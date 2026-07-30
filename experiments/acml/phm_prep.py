"""
PHM 2010 CNC milling tool-wear dataset -> long-format degradation feature table.

The PHM 2010 challenge data contains 6 high-speed milling cutters (c1..c6), each
run for 315 cuts. Every cut is a raw multi-sensor acquisition (~127k samples at
50 kHz) with 7 channels:

    Force_x, Force_y, Force_z, Vibration_x, Vibration_y, Vibration_z, AE_RMS

Tool wear (flute wear in microns) grows gradually and near-monotonically with
cut index; cutters c1, c4, c6 ship with measured wear labels. This makes PHM a
gradual, monotone degradation process (like turbofan) but driven by
force/vibration dynamics (like bearings) -- a useful third external test.

We convert each cutter's cuts into a long-format table:

    unit_id, cycle, <per-channel time-domain features>

* unit_id = cutter id (c1..c6)
* cycle   = cut index (1..315, chronological)
* features = per-channel RMS / std / peak / kurtosis / skew (time domain only;
  cheap and effective for tool-wear monitoring).

Output
------
  data/processed/phm_features.csv

Usage
-----
  python experiments/acml/phm_prep.py
  python experiments/acml/phm_prep.py --cutters c1 c4 c6   # labelled only
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PHM_DIR = os.path.join(ROOT, "data", "raw_PHM")
OUT_CSV = os.path.join(ROOT, "data", "processed", "phm_features.csv")

CHANNELS = ["Fx", "Fy", "Fz", "Vx", "Vy", "Vz", "AE"]
_CUT_RE = re.compile(r"_(\d+)\.csv$")


def _channel_features(x: np.ndarray, name: str) -> dict:
    x = x.astype(np.float64)
    mu = x.mean()
    m2 = np.mean((x - mu) ** 2) + 1e-12
    return {
        f"{name}_rms": float(np.sqrt(np.mean(x ** 2))),
        f"{name}_std": float(np.sqrt(m2)),
        f"{name}_peak": float(np.max(np.abs(x))),
        f"{name}_kurt": float(np.mean((x - mu) ** 4) / (m2 ** 2)),
        f"{name}_skew": float(np.mean((x - mu) ** 3) / (m2 ** 1.5)),
    }


def _cut_index(path: str) -> int:
    m = _CUT_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def process_cutter(cutter: str, max_rows: int | None) -> pd.DataFrame:
    cut_dir = os.path.join(PHM_DIR, cutter, cutter)
    files = glob.glob(os.path.join(cut_dir, "*.csv"))
    files = sorted([f for f in files if _cut_index(f) > 0], key=_cut_index)
    if not files:
        print(f"  [skip] {cutter}: no cut files")
        return pd.DataFrame()
    print(f"  {cutter}: {len(files)} cuts")
    rows = []
    for fp in files:
        arr = pd.read_csv(fp, header=None, dtype=np.float32).to_numpy()
        if arr.shape[1] < len(CHANNELS):
            continue
        if max_rows is not None and arr.shape[0] > max_rows:
            arr = arr[:max_rows]
        feats = {}
        for j, ch in enumerate(CHANNELS):
            feats.update(_channel_features(arr[:, j], ch))
        feats["unit_id"] = cutter
        feats["cycle"] = _cut_index(fp)
        rows.append(feats)
        if len(rows) % 100 == 0:
            print(f"    processed {len(rows)}/{len(files)} cuts")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutters", nargs="*",
                    default=["c1", "c2", "c3", "c4", "c5", "c6"])
    ap.add_argument("--max-rows", type=int, default=None,
                    help="use only the first N samples per cut (speed)")
    args = ap.parse_args()

    print("PHM 2010 milling feature extraction")
    print("=" * 60)
    parts = []
    for c in args.cutters:
        df = process_cutter(c, args.max_rows)
        if not df.empty:
            parts.append(df)
    if not parts:
        print("No data processed; aborting.")
        sys.exit(1)
    out = pd.concat(parts, ignore_index=True)
    feat_cols = [c for c in out.columns if c not in ("unit_id", "cycle")]
    out = out[["unit_id", "cycle"] + feat_cols].sort_values(
        ["unit_id", "cycle"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print("=" * 60)
    print(f"units: {out['unit_id'].nunique()}  rows: {len(out)}  "
          f"features: {len(feat_cols)}")
    print(f"wrote -> {OUT_CSV}")


if __name__ == "__main__":
    main()
