from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .config import WINDOW
from .regimes import condition_normalize
from .state import require_cfg


def denoise(df: pd.DataFrame, cols: List[str] | None = None, window: int = WINDOW, causal: bool = False) -> pd.DataFrame:
    cfg = require_cfg()
    cols = cols or cfg.dynamic
    out = condition_normalize(df)
    for s in cols:
        if causal:
            out[s] = out.groupby("unit_id")[s].transform(lambda v: v.rolling(window, min_periods=1).median())
        else:
            out[s] = out.groupby("unit_id")[s].transform(
                lambda v: v.rolling(window, center=True, min_periods=1).median()
            )
    return out


def same_engine_mask(df: pd.DataFrame) -> np.ndarray:
    uid = df["unit_id"].to_numpy()
    return uid[:-1] == uid[1:]
