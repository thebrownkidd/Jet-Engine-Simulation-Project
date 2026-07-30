"""
IMS bearing dataset -> long-format degradation feature table.

The NASA/IMS bearing dataset stores raw vibration snapshots: each file is one
~1 s acquisition (20,480 samples at 20 kHz) taken every ~10 min over a
run-to-failure test. This is fundamentally different from C-MAPSS (cycle-indexed
multi-unit sensor tables), so we first convert the raw snapshots into a
long-format feature table that the manifold pipeline can consume:

    unit_id, cycle, <feature_1..feature_p>

Design choices (mirroring the C-MAPSS analogy)
----------------------------------------------
* Each vibration CHANNEL is treated as one degradation "unit" (a bearing has
  1 or 2 accelerometer channels depending on the test set).
* `cycle` is the chronological snapshot index (filenames sort chronologically,
  format YYYY.MM.DD.HH.MM.SS).
* Per-snapshot time- and frequency-domain features play the role of "sensors":
  they are correlated and degradation is expected to be low-dimensional, exactly
  the assumption the bounded-latent manifold model exploits.

Test-set layout (auto-detected from column count):
  1st_test : 8 channels (bearings 1-4, x & y)  -> 8 units
  2nd_test : 4 channels (bearings 1-4)          -> 4 units
  3rd_test : 4 channels (bearings 1-4)          -> 4 units

Output
------
  data/processed/ims_bearing_features.csv   (long format)

Usage
-----
  python experiments/acml/ims_prep.py                # all sets
  python experiments/acml/ims_prep.py --max-files 300  # quick subsample per set
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import glob

import numpy as np
import pandas as pd

# IMS snapshot filenames are timestamps: YYYY.MM.DD.HH.MM.SS
_TS_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RAW_DIR = os.path.join(ROOT, "data", "raw_bearing")
OUT_CSV = os.path.join(ROOT, "data", "processed", "ims_bearing_features.csv")

FS = 20_000.0  # sampling rate (Hz)
N_BANDS = 4    # spectral band-energy features


def _snapshot_files(set_dir: str) -> list[str]:
    """Return chronologically ordered snapshot files under a test-set folder.

    The IMS archive nests the data one level deep (e.g. 1st_test/1st_test/...),
    so we search recursively and keep only extension-less data files.
    """
    files = []
    for p in glob.glob(os.path.join(set_dir, "**", "*"), recursive=True):
        if os.path.isfile(p) and _TS_RE.match(os.path.basename(p)):
            files.append(p)
    files.sort(key=lambda p: os.path.basename(p))  # timestamp names sort in time
    return files


def _channel_features(x: np.ndarray) -> dict:
    """Time- and frequency-domain features for a single vibration channel."""
    x = x.astype(np.float64)
    n = len(x)
    mean_abs = np.mean(np.abs(x)) + 1e-12
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    std = float(np.std(x))
    mu = float(np.mean(x))
    # central moments
    m2 = np.mean((x - mu) ** 2) + 1e-12
    kurt = float(np.mean((x - mu) ** 4) / (m2 ** 2))
    skew = float(np.mean((x - mu) ** 3) / (m2 ** 1.5))
    crest = float(peak / (rms + 1e-12))
    shape = float(rms / mean_abs)
    impulse = float(peak / mean_abs)
    clearance = float(peak / (np.mean(np.sqrt(np.abs(x))) ** 2 + 1e-12))
    # spectral band energy fractions
    mag = np.abs(np.fft.rfft(x * np.hanning(n)))
    power = mag ** 2
    total = power.sum() + 1e-12
    bands = np.array_split(power, N_BANDS)
    band_frac = [float(b.sum() / total) for b in bands]
    feats = dict(rms=rms, peak=peak, p2p=float(x.max() - x.min()), std=std,
                 kurtosis=kurt, skew=skew, crest=crest, shape=shape,
                 impulse=impulse, clearance=clearance)
    for i, bf in enumerate(band_frac):
        feats[f"band{i}"] = bf
    return feats


def process_set(set_name: str, max_files: int | None) -> pd.DataFrame:
    set_dir = os.path.join(RAW_DIR, set_name)
    if not os.path.isdir(set_dir):
        print(f"  [skip] {set_name}: not found")
        return pd.DataFrame()
    files = _snapshot_files(set_dir)
    if not files:
        print(f"  [skip] {set_name}: no snapshot files")
        return pd.DataFrame()
    if max_files is not None and len(files) > max_files:
        idx = np.linspace(0, len(files) - 1, max_files).round().astype(int)
        files = [files[i] for i in sorted(set(idx))]
    # detect channel count from first file
    first = pd.read_csv(files[0], sep="\t", header=None).to_numpy()
    n_ch = first.shape[1]
    print(f"  {set_name}: {len(files)} snapshots x {n_ch} channels")

    rows = []
    for cyc, fpath in enumerate(files):
        arr = pd.read_csv(fpath, sep="\t", header=None).to_numpy()
        if arr.shape[1] != n_ch:
            continue
        for ch in range(n_ch):
            feats = _channel_features(arr[:, ch])
            feats["unit_id"] = f"{set_name}_ch{ch + 1}"
            feats["cycle"] = cyc
            rows.append(feats)
        if (cyc + 1) % 200 == 0:
            print(f"    processed {cyc + 1}/{len(files)} snapshots")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-files", type=int, default=None,
                    help="subsample this many snapshots per set (evenly spaced)")
    ap.add_argument("--sets", nargs="*",
                    default=["1st_test", "2nd_test", "3rd_test"])
    args = ap.parse_args()

    print("IMS feature extraction")
    print("=" * 60)
    parts = []
    for s in args.sets:
        df = process_set(s, args.max_files)
        if not df.empty:
            parts.append(df)
    if not parts:
        print("No data processed; aborting.")
        sys.exit(1)
    out = pd.concat(parts, ignore_index=True)
    # order columns: id/cycle first
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
