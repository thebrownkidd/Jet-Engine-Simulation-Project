from __future__ import annotations

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, SEED, SETTINGS, TEST_SIZE
from .regimes import assign_regime
from .state import require_cfg


def _parquet(split: str, fd: int) -> str:
    return os.path.join(DATA_DIR, f"{split}_FD00{fd}.parquet")


def _read(split: str, fd: int) -> pd.DataFrame:
    return pd.read_parquet(_parquet(split, fd)).sort_values(["unit_id", "cycle"]).reset_index(drop=True)


def n_conditions(fd: int) -> int:
    tr = _read("train", fd)
    return min(6, tr[SETTINGS].round(0).drop_duplicates().shape[0])


def load_split(name: str) -> pd.DataFrame:
    cfg = require_cfg()
    df = _read(name, cfg.fd)
    df["regime"] = assign_regime(df)
    maxc = df.groupby("unit_id")["cycle"].transform("max")
    df["d"] = df["cycle"] / maxc
    return df


def load_rul() -> pd.DataFrame:
    cfg = require_cfg()
    return pd.read_parquet(os.path.join(DATA_DIR, f"RUL_FD00{cfg.fd}.parquet"))


def split_by_unit(df: pd.DataFrame):
    units = sorted(df["unit_id"].unique().tolist())
    tr_u, te_u = train_test_split(units, test_size=TEST_SIZE, random_state=SEED)
    tr = df[df["unit_id"].isin(tr_u)].sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    te = df[df["unit_id"].isin(te_u)].sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    return tr, te
