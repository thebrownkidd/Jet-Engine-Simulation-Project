from __future__ import annotations

from .config import (
    ALL_SENSORS,
    DATA_DIR,
    EPOCHS,
    FIG_BASE,
    HERE,
    K,
    LAMBDA_MONO,
    LAMBDA_SMOOTH,
    LR,
    MODEL_DIR,
    RESULTS_DIR,
    ROOT,
    SEED,
    SETTINGS,
    TABLE_DIR,
    TEST_SIZE,
    TREND_DYNAMIC,
    TREND_INFORMATIVE,
    WINDOW,
)
from .context import configure, discovery_info, fig_dir
from .data import load_rul, load_split, n_conditions, split_by_unit
from .denoise import denoise, same_engine_mask
from .health import per_engine_health, r2_pooled
from .model import HealthAE, Manifold
from .regimes import condition_normalize
from .train import get_manifold


# Keep runtime-mutated names backward compatible with prior mc.* usage.
def __getattr__(name: str):
    if name in {"FD", "DYNAMIC", "INFORMATIVE", "ART_DIR"}:
        from . import state

        return getattr(state, name)
    raise AttributeError(name)
