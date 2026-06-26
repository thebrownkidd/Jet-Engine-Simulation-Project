from __future__ import annotations

import os

import numpy as np
from sklearn.cluster import KMeans

from .config import ALL_SENSORS, FIG_BASE, MODEL_DIR, SEED, SETTINGS, TABLE_DIR
from .data import _read, load_split
from .denoise import denoise
from .selection import choose_sensor_sets, sensor_trends
from .state import Config, require_cfg, set_cfg
from .train import get_manifold


def configure(fd: int, retrain: bool = False) -> None:
    model_dir = os.path.join(MODEL_DIR, f"FD00{fd}")
    table_dir = os.path.join(TABLE_DIR, f"FD00{fd}")
    fig_d = os.path.join(FIG_BASE, f"FD00{fd}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(fig_d, exist_ok=True)

    tr = _read("train", fd)
    nreg = min(6, tr[SETTINGS].round(0).drop_duplicates().shape[0])
    km = KMeans(n_clusters=nreg, n_init=10, random_state=SEED).fit(tr[SETTINGS].to_numpy())
    reg = km.predict(tr[SETTINGS].to_numpy())

    x = tr[ALL_SENSORS].to_numpy().astype(float)
    reg_mean = np.zeros((nreg, len(ALL_SENSORS)))
    for r in range(nreg):
        m = reg == r
        reg_mean[r] = x[m].mean(0) if m.any() else 0.0
    resid = x - reg_mean[reg]
    resid_std = np.maximum(resid.std(0), 1e-9)
    global_std = x.std(0)

    cfg = Config(
        fd=fd,
        n_regimes=nreg,
        km=km,
        reg_mean=reg_mean,
        resid_std=resid_std,
        global_std=global_std,
        dynamic=list(ALL_SENSORS),
        informative=[],
        trend={},
        weights=np.ones(len(ALL_SENSORS)),
        model_dir=model_dir,
        table_dir=table_dir,
        fig_dir=fig_d,
    )
    set_cfg(cfg)

    tr_df = load_split("train")
    tr_den = denoise(tr_df, cols=ALL_SENSORS)
    trend = sensor_trends(tr_den)
    dynamic, informative, weights = choose_sensor_sets(trend)

    cfg.dynamic = dynamic
    cfg.informative = informative
    cfg.trend = trend
    cfg.weights = weights
    set_cfg(cfg)

    if retrain:
        get_manifold(retrain=True)


def fig_dir() -> str:
    cfg = require_cfg()
    os.makedirs(cfg.fig_dir, exist_ok=True)
    return cfg.fig_dir


def discovery_info() -> dict:
    cfg = require_cfg()
    return dict(
        fd=cfg.fd,
        n_regimes=cfg.n_regimes,
        n_dynamic=len(cfg.dynamic),
        n_informative=len(cfg.informative),
        dynamic=list(cfg.dynamic),
        informative=list(cfg.informative),
        trend=cfg.trend,
    )
