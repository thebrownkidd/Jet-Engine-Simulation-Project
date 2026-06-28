from __future__ import annotations

import os

import numpy as np
from sklearn.cluster import KMeans

from .config import ALL_SENSORS, FIG_BASE, K, LAMBDA_MONO, LAMBDA_SMOOTH, MODEL_DIR, SEED, SETTINGS, TABLE_DIR
from .data import _read, load_split
from .denoise import denoise
from .regimes import select_n_regimes
from .selection import choose_sensor_sets, sensor_trends
from .state import Config, require_cfg, set_cfg
from .train import get_manifold


def configure(fd: int, retrain: bool = False, *, k: int | None = None,
              lambda_mono: float | None = None, lambda_smooth: float | None = None,
              normalize: bool = True, regime_rule: str = "heuristic",
              tag: str = "") -> None:
    """Select dataset FD00<fd> and prepare the pipeline.

    Experiment knobs (all default to the validated production pipeline):
        k             latent bottleneck dimension (default 2)
        lambda_mono   monotonicity penalty weight (default 5.0)
        lambda_smooth smoothness penalty weight (default 2.0)
        normalize     True -> per-regime normalisation; False -> global std
        regime_rule   'heuristic' (min(6, #rounded-settings)) | 'silhouette'
        tag           cache namespace so variants don't overwrite each other
    """
    k = K if k is None else k
    lambda_mono = LAMBDA_MONO if lambda_mono is None else lambda_mono
    lambda_smooth = LAMBDA_SMOOTH if lambda_smooth is None else lambda_smooth

    model_dir = os.path.join(MODEL_DIR, f"FD00{fd}")
    table_dir = os.path.join(TABLE_DIR, f"FD00{fd}")
    fig_d = os.path.join(FIG_BASE, f"FD00{fd}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(fig_d, exist_ok=True)

    tr = _read("train", fd)
    settings = tr[SETTINGS].to_numpy()
    if regime_rule == "silhouette":
        nreg = select_n_regimes(settings)
    else:
        nreg = min(6, tr[SETTINGS].round(0).drop_duplicates().shape[0])
    km = KMeans(n_clusters=nreg, n_init=10, random_state=SEED).fit(settings)
    reg = km.predict(settings)

    x = tr[ALL_SENSORS].to_numpy().astype(float)
    reg_mean = np.zeros((nreg, len(ALL_SENSORS)))
    for r in range(nreg):
        m = reg == r
        reg_mean[r] = x[m].mean(0) if m.any() else 0.0
    resid = x - reg_mean[reg]
    resid_std = np.maximum(resid.std(0), 1e-9)
    global_std = x.std(0)
    global_mean = x.mean(0)

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
        k=k,
        lambda_mono=lambda_mono,
        lambda_smooth=lambda_smooth,
        normalize=normalize,
        global_mean=global_mean,
        regime_rule=regime_rule,
        tag=tag,
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
