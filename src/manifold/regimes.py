from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ALL_SENSORS, SETTINGS
from .state import require_cfg


def assign_regime(df: pd.DataFrame) -> np.ndarray:
    cfg = require_cfg()
    return cfg.km.predict(df[SETTINGS].to_numpy())


def condition_normalize(df: pd.DataFrame) -> pd.DataFrame:
    cfg = require_cfg()
    out = df.copy()
    reg = out["regime"].to_numpy() if "regime" in out else assign_regime(out)
    x = out[ALL_SENSORS].to_numpy().astype(float)
    x = x - cfg.reg_mean[reg]
    x = x / cfg.resid_std[None, :]
    out[ALL_SENSORS] = x
    return out
