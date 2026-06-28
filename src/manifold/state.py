from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans

from .config import K, LAMBDA_MONO, LAMBDA_SMOOTH, TABLE_DIR


@dataclass
class Config:
    fd: int
    n_regimes: int
    km: KMeans
    reg_mean: np.ndarray
    resid_std: np.ndarray
    global_std: np.ndarray
    dynamic: List[str]
    informative: List[str]
    trend: Dict[str, float]
    weights: np.ndarray
    model_dir: str
    table_dir: str
    fig_dir: str
    # --- experiment knobs (defaults reproduce the validated pipeline) ------ #
    k: int = K
    lambda_mono: float = LAMBDA_MONO
    lambda_smooth: float = LAMBDA_SMOOTH
    normalize: bool = True            # False -> global standardisation (no regimes)
    global_mean: Optional[np.ndarray] = None
    regime_rule: str = "heuristic"    # 'heuristic' | 'silhouette'
    tag: str = ""                     # cache namespace for experiment variants


CFG: Config | None = None
FD = 1
DYNAMIC: List[str] = []
INFORMATIVE: List[str] = []
ART_DIR = TABLE_DIR


def require_cfg() -> Config:
    if CFG is None:
        raise RuntimeError("configure(fd) must be called first")
    return CFG


def set_cfg(cfg: Config) -> None:
    global CFG, FD, DYNAMIC, INFORMATIVE, ART_DIR
    CFG = cfg
    FD = cfg.fd
    DYNAMIC = list(cfg.dynamic)
    INFORMATIVE = list(cfg.informative)
    ART_DIR = cfg.table_dir


def refresh_runtime_lists() -> None:
    cfg = require_cfg()
    set_cfg(cfg)
