from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .config import ALL_SENSORS, SEED, SETTINGS
from .state import require_cfg


def assign_regime(df: pd.DataFrame) -> np.ndarray:
    cfg = require_cfg()
    return cfg.km.predict(df[SETTINGS].to_numpy())


def select_n_regimes(settings: np.ndarray, k_range=range(2, 9),
                     sil_threshold: float = 0.7, constant_tol: float = 1e-3) -> int:
    """Data-driven regime count via silhouette over KMeans.

    Returns 1 when the operating settings have no real cluster structure
    (near-constant settings, or best silhouette below `sil_threshold`).
    """
    if settings.std(0).max() < constant_tol:
        return 1
    best_k, best_sil = 1, -1.0
    n = len(settings)
    idx = np.arange(n)
    if n > 8000:
        idx = np.random.default_rng(SEED).choice(idx, 8000, replace=False)
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(settings)
        labels = km.labels_
        if len(np.unique(labels)) < 2:
            continue
        sil = silhouette_score(settings[idx], labels[idx])
        if sil > best_sil:
            best_sil, best_k = sil, k
    return best_k if best_sil >= sil_threshold else 1


def condition_normalize(df: pd.DataFrame) -> pd.DataFrame:
    cfg = require_cfg()
    out = df.copy()
    x = out[ALL_SENSORS].to_numpy().astype(float)
    if cfg.normalize:
        reg = out["regime"].to_numpy() if "regime" in out else assign_regime(out)
        x = (x - cfg.reg_mean[reg]) / cfg.resid_std[None, :]
    else:
        # ablation: global standardisation, no regime structure
        gm = cfg.global_mean if cfg.global_mean is not None else x.mean(0)
        x = (x - gm) / np.maximum(cfg.global_std, 1e-9)[None, :]
    out[ALL_SENSORS] = x
    return out
