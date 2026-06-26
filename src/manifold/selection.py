from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import ALL_SENSORS, TREND_DYNAMIC, TREND_INFORMATIVE


def sensor_trends(df_den: pd.DataFrame) -> Dict[str, float]:
    trends: Dict[str, float] = {}
    groups = list(df_den.groupby("unit_id"))
    for s in ALL_SENSORS:
        vals = []
        for _, g in groups:
            y = g[s].to_numpy()
            c = g["cycle"].to_numpy().astype(float)
            if y.std() < 1e-9 or len(y) < 5:
                continue
            r = np.corrcoef(y, c)[0, 1]
            if np.isfinite(r):
                vals.append(abs(r))
        trends[s] = float(np.mean(vals)) if vals else 0.0
    return trends


def choose_sensor_sets(trend: Dict[str, float]) -> Tuple[List[str], List[str], np.ndarray]:
    dynamic = [s for s in ALL_SENSORS if trend[s] >= TREND_DYNAMIC]
    informative = [s for s in ALL_SENSORS if trend[s] >= TREND_INFORMATIVE]
    if len(dynamic) < 3:
        dynamic = sorted(ALL_SENSORS, key=lambda s: -trend[s])[:8]
        informative = [s for s in dynamic if trend[s] >= 0.3] or dynamic[:5]
    weights = np.array([max(trend[s], 0.05) for s in dynamic])
    return dynamic, informative, weights
