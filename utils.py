# utils.py — shared data loading helpers

import os
import numpy as np
import pandas as pd

DATA_DIR = 'Data'

SENSOR_COLS = [f's{i}' for i in range(1, 22)]   # s1 … s21
SETTING_COLS = ['setting_1', 'setting_2', 'setting_3']

SENSOR_LABELS = [
    "T24 - Total temperature at LPC outlet (°R)",
    "T30 - Total temperature at HPC outlet (°R)",
    "T50 - Total temperature at LPT outlet (°R)",
    "P15 - Total pressure in bypass duct (psia)",
    "P30 - Total pressure at HPC outlet (psia)",
    "Nf  - Physical fan speed (rpm)",
    "Nc  - Physical core speed (rpm)",
    "epr - Engine pressure ratio (-)",
    "Ps30 - Static pressure at HPC outlet (psia)",
    "phi - Ratio of fuel flow to Ps30 (pps/psi)",
    "NRf - Corrected fan speed (rpm)",
    "NRc - Corrected core speed (rpm)",
    "BPR - Bypass Ratio (-)",
    "farB - Burner fuel-air ratio (-)",
    "htBleed - Bleed enthalpy (-)",
    "Nf_dmd - Demanded fan speed (rpm)",
    "PCNfR_dmd - Demanded corrected fan speed (rpm)",
    "W31 - HPT coolant bleed flow (lbm/s)",
    "W32 - LPT coolant bleed flow (lbm/s)",
    "s20 (unused)",
    "s21 (unused)",
]


def load_parquet(fd: int, split: str = 'train', data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Load a raw C-MAPSS parquet file.

    Parameters
    ----------
    fd : int
        Dataset number 1–4  (FD001 … FD004).
    split : str
        'train' or 'test'.
    data_dir : str
        Directory that contains the parquet files.

    Returns
    -------
    pd.DataFrame  with columns: unit_id, cycle, setting_1-3, s1-s21
    """
    path = os.path.join(data_dir, f'{split}_FD00{fd}.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Parquet file not found: {path}\n"
            "Run convert_to_parquet.py first."
        )
    return pd.read_parquet(path)


def load_rul(fd: int, data_dir: str = DATA_DIR) -> np.ndarray:
    """
    Load the RUL ground-truth vector for the test set of a given dataset.

    Returns
    -------
    np.ndarray of shape (n_test_units,) — RUL for each unit (0-indexed).
    """
    path = os.path.join(data_dir, f'RUL_FD00{fd}.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"RUL parquet not found: {path}\n"
            "Run convert_to_parquet.py first."
        )
    return pd.read_parquet(path)['rul'].to_numpy()


def df_to_timeseries(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> list[np.ndarray]:
    """
    Convert a flat C-MAPSS DataFrame into a list of per-unit time series.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_parquet().
    columns : list[str] | None
        Which columns to include in each array.
        Defaults to all 21 sensor columns (s1–s21).

    Returns
    -------
    list of np.ndarray, one per unit, each of shape (n_cycles, n_features).
    Units are sorted by unit_id and returned in ascending order.
    """
    if columns is None:
        columns = SENSOR_COLS

    units = []
    for _, grp in df.sort_values(['unit_id', 'cycle']).groupby('unit_id', sort=True):
        units.append(grp[columns].to_numpy(dtype=np.float32))

    return units


def _resolve_columns(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    """Return validated target columns for column-wise transforms."""
    if columns is None:
        columns = SENSOR_COLS
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")
    return columns


def standard_scale_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    eps: float = 1e-12,
    return_stats: bool = False,
):
    """
    Standard-scale selected columns: (x - mean) / std.

    Scaling is applied independently per column.
    Constant columns are mapped to 0.0 (safe divide).
    """
    cols = _resolve_columns(df, columns)
    out = df.copy()

    means = out[cols].mean(axis=0)
    stds = out[cols].std(axis=0, ddof=0)
    safe_stds = stds.mask(stds.abs() < eps, 1.0)

    out[cols] = (out[cols] - means) / safe_stds

    if return_stats:
        return out, {"mean": means.to_dict(), "std": stds.to_dict()}
    return out


def vector_norm_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    eps: float = 1e-12,
    return_stats: bool = False,
):
    """
    L2-normalize selected columns: x / ||x||_2.

    Normalization is applied independently per column.
    Zero/constant-zero columns remain 0.0 (safe divide).
    """
    cols = _resolve_columns(df, columns)
    out = df.copy()

    norms = np.sqrt((out[cols] ** 2).sum(axis=0))
    safe_norms = norms.mask(norms.abs() < eps, 1.0)

    out[cols] = out[cols] / safe_norms

    if return_stats:
        return out, {"l2_norm": norms.to_dict()}
    return out


def min_max_scale_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    feature_range: tuple[float, float] = (0.0, 1.0),
    eps: float = 1e-12,
    return_stats: bool = False,
):
    """
    Min-max scale selected columns to [low, high].

    Scaling is applied independently per column.
    Constant columns are mapped to low.
    """
    low, high = feature_range
    if high <= low:
        raise ValueError("feature_range must satisfy high > low")

    cols = _resolve_columns(df, columns)
    out = df.copy()

    col_min = out[cols].min(axis=0)
    col_max = out[cols].max(axis=0)
    denom = col_max - col_min
    safe_denom = denom.mask(denom.abs() < eps, 1.0)

    scaled_01 = (out[cols] - col_min) / safe_denom
    out[cols] = scaled_01 * (high - low) + low

    # Constant columns should map exactly to low.
    const_cols = denom.index[denom.abs() < eps]
    if len(const_cols) > 0:
        out.loc[:, const_cols] = low

    if return_stats:
        return out, {
            "min": col_min.to_dict(),
            "max": col_max.to_dict(),
            "feature_range": feature_range,
        }
    return out


def apply_min_max_scale_columns(
    df: pd.DataFrame,
    stats: dict,
    columns: list[str] | None = None,
    eps: float = 1e-12,
):
    """
    Apply precomputed min-max stats to selected columns.

    This is used to fit on train and transform test/all without leakage.
    """
    cols = _resolve_columns(df, columns)
    out = df.copy()

    low, high = stats.get("feature_range", (0.0, 1.0))
    if high <= low:
        raise ValueError("feature_range in stats must satisfy high > low")

    col_min = pd.Series(stats["min"])  # keyed by column
    col_max = pd.Series(stats["max"])
    col_min = col_min.reindex(cols)
    col_max = col_max.reindex(cols)

    missing_stats = [c for c in cols if pd.isna(col_min[c]) or pd.isna(col_max[c])]
    if missing_stats:
        raise KeyError(f"Missing min/max stats for columns: {missing_stats}")

    denom = col_max - col_min
    safe_denom = denom.mask(denom.abs() < eps, 1.0)

    scaled_01 = (out[cols] - col_min) / safe_denom
    out[cols] = scaled_01 * (high - low) + low

    const_cols = denom.index[denom.abs() < eps]
    if len(const_cols) > 0:
        out.loc[:, const_cols] = low

    return out
