"""
C-MAPSS (NASA turbofan) parquet -> unified long-format features CSV.

Reads the pre-processed parquet files in data/processed/train_FD*.parquet,
applies per-regime sensor normalization, drops constant/near-constant sensors,
and writes a standard features CSV that the unified experiment runner can load
with exactly the same interface as IMS/battery/PHM/air-quality.

Operating settings are emitted as ctx_setting_* columns so the context-
conditioned dynamics model can condition on current operating regime.

Usage
-----
    python experiments/acml/cmapss_prep.py          # all four FDs
    python experiments/acml/cmapss_prep.py --fd 1   # FD001 only
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]

# Sensors known to be constant in all FD sub-datasets (zero variance) —
# they carry no information and will be auto-dropped by std filter anyway,
# but listing them makes the intent explicit.
CONSTANT_SENSORS = {"s1", "s5", "s6", "s10", "s16", "s18", "s19"}

MIN_STD = 1e-4       # drop sensor if pooled std < this
N_REGIMES_MAP = {    # known operating-condition counts from the C-MAPSS docs
    "FD001": 1,
    "FD002": 6,
    "FD003": 1,
    "FD004": 6,
}


def regime_normalize(df: pd.DataFrame, fd_tag: str):
    """
    Normalize each sensor by per-regime mean/std to remove the operating-
    condition staircase effect.

    For FD001/FD003 (1 regime) this reduces to global standardisation.
    For FD002/FD004 (6 regimes) KMeans clusters the 3 operating settings.
    """
    n_reg = N_REGIMES_MAP[fd_tag]
    settings = df[SETTING_COLS].to_numpy()

    if n_reg == 1:
        df = df.copy()
        df["regime"] = 0
    else:
        km = KMeans(n_clusters=n_reg, random_state=42, n_init=10)
        df = df.copy()
        df["regime"] = km.fit_predict(settings)

    # Compute per-regime means and pooled within-regime std
    good_sensors = []
    for s in SENSOR_COLS:
        vals = df[s].to_numpy()
        # pooled within-regime std
        pooled_var = 0.0
        total = 0
        for r in range(n_reg):
            mask = (df["regime"] == r).to_numpy()
            if mask.sum() < 2:
                continue
            v = vals[mask]
            pooled_var += (v - v.mean()).var() * mask.sum()
            total += mask.sum()
        pooled_std = float(np.sqrt(pooled_var / max(total - n_reg, 1)))
        if pooled_std < MIN_STD:
            continue  # constant sensor, drop
        # per-regime mean subtraction
        normed = vals.copy().astype(float)
        for r in range(n_reg):
            mask = (df["regime"] == r).to_numpy()
            if mask.sum() < 1:
                continue
            normed[mask] = (normed[mask] - vals[mask].mean()) / pooled_std
        df[s] = normed
        good_sensors.append(s)

    df.drop(columns=["regime"], inplace=True)
    return df, good_sensors


def standardize_settings(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize operating settings and rename to ctx_setting_* columns."""
    df = df.copy()
    for c in SETTING_COLS:
        s = df[c].std()
        if s < 1e-9:
            # constant (FD001/FD003): still include as constant context
            df[f"ctx_{c}"] = 0.0
        else:
            df[f"ctx_{c}"] = (df[c] - df[c].mean()) / s
    return df


def process_fd(fd: int, out_dir: str):
    tag = f"FD{fd:03d}"
    parquet_path = os.path.join(ROOT, "data", "processed", f"train_{tag}.parquet")
    if not os.path.exists(parquet_path):
        print(f"  SKIP {tag}: {parquet_path} not found")
        return

    df = pd.read_parquet(parquet_path)
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    print(f"  {tag}: {len(df)} rows, {df['unit_id'].nunique()} units")

    # Regime-normalize sensors
    df, good_sensors = regime_normalize(df, tag)
    print(f"       {len(good_sensors)} sensors after regime-normalization "
          f"(dropped {len(SENSOR_COLS) - len(good_sensors)} constant)")

    # Add standardized settings as context columns
    df = standardize_settings(df)
    ctx_cols = [f"ctx_{c}" for c in SETTING_COLS]

    # Assemble output: unit_id, cycle, good_sensors, ctx_cols
    out_cols = ["unit_id", "cycle"] + good_sensors + ctx_cols
    out = df[out_cols].copy()

    out_path = os.path.join(out_dir, f"cmapss_{tag}_features.csv")
    out.to_csv(out_path, index=False)
    print(f"       -> {out_path}  "
          f"({len(good_sensors)} features, {len(ctx_cols)} context)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", type=int, nargs="*", default=[1, 2, 3, 4],
                    help="which FD sub-datasets to process (default: all)")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    print("C-MAPSS turbofan -> unified features CSV")
    print("=" * 60)
    for fd in args.fd:
        process_fd(fd, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
