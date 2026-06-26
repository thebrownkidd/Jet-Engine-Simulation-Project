from __future__ import annotations

import numpy as np
import pandas as pd

from .model import Manifold


def per_engine_health(man: Manifold, df_den: pd.DataFrame) -> pd.DataFrame:
    h = man.encode(df_den)
    out = df_den[["unit_id", "cycle"]].copy()
    if "d" in df_den:
        out["d"] = df_den["d"].to_numpy()
    out["h0"] = h[:, 0]
    out["h1"] = h[:, 1]
    return out


def r2_pooled(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))
