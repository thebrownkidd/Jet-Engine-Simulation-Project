"""
Beijing Multi-Site Air-Quality dataset -> long-format multivariate table.

Unlike the degradation datasets (turbofan / bearing / battery / milling), this is
a *forecasting* problem: 12 monitoring stations record hourly air-quality and
meteorological variables from 2013-03-01 to 2017-02-28. There is no run-to-
failure and no RUL. We therefore use it to test only the CORE contribution of the
method -- stable bounded-latent rollout and multi-step forecastability -- not
degradation-specific claims (RUL, monotone health).

Layout produced:

    unit_id, cycle, <numeric variables>

* unit_id = station name (12 stations = 12 "series")
* cycle   = hour index within the station (chronological)
* features = PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM
  (the categorical wind-direction `wd` is dropped; missing values are linearly
  interpolated per station).

Output
------
  data/processed/air_quality_features.csv

Usage
-----
  python experiments/acml/air_quality_prep.py                 # stride=3
  python experiments/acml/air_quality_prep.py --stride 1      # full hourly
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
AQ_DIR = os.path.join(ROOT, "data", "raw_bejing_multi_sire_air_quality",
                      "PRSA2017_Data_20130301-20170228",
                      "PRSA_Data_20130301-20170228")
OUT_CSV = os.path.join(ROOT, "data", "processed", "air_quality_features.csv")

NUMERIC = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP",
           "RAIN", "WSPM"]

# Deterministic calendar context columns (known arbitrarily far into the future).
# These are cyclic encodings of hour-of-day, day-of-week and month-of-year, so
# they are exact at any forecast horizon and require no exogenous forecasting.
CTX_COLS = ["ctx_sin_hour", "ctx_cos_hour", "ctx_sin_dow", "ctx_cos_dow",
            "ctx_sin_month", "ctx_cos_month"]


def _cyclic(vals: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    ang = 2.0 * np.pi * (vals.astype(float) / period)
    return np.sin(ang), np.cos(ang)


def process_station(path: str, stride: int, context: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    station = str(df["station"].iloc[0])
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["year", "month", "day", "hour"]).reset_index(drop=True)
    df[NUMERIC] = (df[NUMERIC].interpolate(limit_direction="both")
                   .ffill().bfill())
    if context:
        dt = pd.to_datetime(dict(year=df["year"], month=df["month"],
                                 day=df["day"], hour=df["hour"]))
        df["_dow"] = dt.dt.dayofweek.to_numpy()
    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)
    out = df[NUMERIC].copy()
    if context:
        sh, ch = _cyclic(df["hour"].to_numpy(), 24.0)
        sd, cd = _cyclic(df["_dow"].to_numpy(), 7.0)
        sm, cm = _cyclic(df["month"].to_numpy() - 1, 12.0)
        out["ctx_sin_hour"], out["ctx_cos_hour"] = sh, ch
        out["ctx_sin_dow"], out["ctx_cos_dow"] = sd, cd
        out["ctx_sin_month"], out["ctx_cos_month"] = sm, cm
    out.insert(0, "cycle", np.arange(len(out)))
    out.insert(0, "unit_id", station)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3,
                    help="keep every Nth hour (downsample for tractable training)")
    ap.add_argument("--context", action="store_true",
                    help="append deterministic cyclic calendar context columns")
    ap.add_argument("--out", type=str, default=OUT_CSV,
                    help="output CSV path")
    args = ap.parse_args()

    print("Beijing air-quality feature extraction")
    print("=" * 60)
    files = sorted(glob.glob(os.path.join(AQ_DIR, "PRSA_Data_*.csv")))
    if not files:
        print(f"No station files under {AQ_DIR}")
        sys.exit(1)
    parts = []
    for fp in files:
        d = process_station(fp, args.stride, args.context)
        parts.append(d)
        print(f"  {d['unit_id'].iloc[0]}: {len(d)} samples (stride={args.stride})")
    out = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print("=" * 60)
    ctx_note = f" + {len(CTX_COLS)} context" if args.context else ""
    print(f"units: {out['unit_id'].nunique()}  rows: {len(out)}  "
          f"features: {len(NUMERIC)}{ctx_note}")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
