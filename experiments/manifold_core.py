"""Shared core for the discovery / rollout / forecasting / RUL experiments.

GENERALISED to all four C-MAPSS sub-datasets (FD001-FD004).

The original FD001 pipeline assumed a *single operating point*, so a global
standardisation of the sensors revealed the degradation trend directly. FD002
and FD004 run over **six operating regimes**, and FD003/FD004 contain **two
fault modes**. A global standardisation there is dominated by the regime steps
(the "staircase"), not by wear. The fix, standard in the C-MAPSS RUL
literature, is **operating-condition normalisation**:

    1. cluster the three operating settings into regimes (KMeans);
    2. for every sensor, remove the per-regime mean (kills the staircase) and
       divide by the pooled *within-regime* standard deviation (puts every
       sensor on a common scale);
    3. the residual is, by construction, the part of each sensor that is NOT
       explained by the operating condition -- i.e. the degradation signal.

After this transform, FD002/FD004 look like single-condition problems and the
same physics-constrained health-manifold autoencoder applies unchanged.

Public API (used by the experiment scripts)
-------------------------------------------
configure(fd, retrain=False) -> None          # select + prepare a dataset
load_split(name) -> DataFrame                  # 'train' | 'test' (+regime, +d)
load_rul() -> DataFrame
split_by_unit(df) -> (train_df, test_df)       # engine-disjoint 80/20
denoise(df, cols=None, window=15, causal=False)# condition-norm + rolling median
get_manifold(retrain=False) -> Manifold        # train/cache the k=2 manifold
per_engine_health(man, df_den) -> DataFrame
fig_dir() -> str                               # docs/figures/FD00<fd>/
r2_pooled(y_true, y_pred) -> float

Module globals re-bound by configure(): FD, DYNAMIC, INFORMATIVE, ART_DIR.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

ALL_SENSORS = [f"s{i}" for i in range(1, 22)]
SETTINGS = ["setting_1", "setting_2", "setting_3"]

# hyper-parameters (shared across datasets for a fair comparison)
WINDOW = 15
K = 2
EPOCHS = 4000
LR = 5e-3
LAMBDA_MONO = 5.0
LAMBDA_SMOOTH = 2.0
TEST_SIZE = 0.2
TREND_DYNAMIC = 0.20     # |corr(sensor, cycle)| to enter the AE input set
TREND_INFORMATIVE = 0.50  # |corr(sensor, cycle)| to be scored for reconstruction

HERE = os.path.dirname(os.path.abspath(__file__))

# Module globals re-bound by configure() -- defaults are placeholders.
FD = 1
DYNAMIC: List[str] = []
INFORMATIVE: List[str] = []
ART_DIR = os.path.join(HERE, "artifacts")


# --------------------------------------------------------------------------- #
# Per-dataset configuration object
# --------------------------------------------------------------------------- #
@dataclass
class _Config:
    fd: int
    n_regimes: int
    km: KMeans
    reg_mean: np.ndarray          # (n_regimes, 21)  per-regime sensor means
    resid_std: np.ndarray         # (21,)            pooled within-regime std
    global_std: np.ndarray        # (21,)            total sensor std
    dynamic: List[str]
    informative: List[str]
    trend: Dict[str, float]
    weights: np.ndarray           # ceiling-style weights over `dynamic`
    art_dir: str
    fig_dir: str


_CFG: _Config | None = None


# --------------------------------------------------------------------------- #
# Raw data
# --------------------------------------------------------------------------- #
def _parquet(split: str, fd: int) -> str:
    return os.path.join(HERE, os.pardir, "Data", f"{split}_FD00{fd}.parquet")


def _read(split: str, fd: int) -> pd.DataFrame:
    return (pd.read_parquet(_parquet(split, fd))
            .sort_values(["unit_id", "cycle"]).reset_index(drop=True))


def n_conditions(fd: int) -> int:
    """Number of distinct operating regimes (1 for FD001/FD003, 6 for 002/004)."""
    tr = _read("train", fd)
    return min(6, tr[SETTINGS].round(0).drop_duplicates().shape[0])


# --------------------------------------------------------------------------- #
# Operating-condition normalisation
# --------------------------------------------------------------------------- #
def _assign_regime(df: pd.DataFrame) -> np.ndarray:
    return _CFG.km.predict(df[SETTINGS].to_numpy())


def condition_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the per-regime mean and scale by within-regime std.

    The result's sensor columns hold the *degradation residual* on a common
    scale; sensors fully explained by the operating condition collapse to ~0.
    """
    out = df.copy()
    reg = out["regime"].to_numpy() if "regime" in out else _assign_regime(out)
    X = out[ALL_SENSORS].to_numpy().astype(float)
    X = X - _CFG.reg_mean[reg]                     # kill the regime staircase
    X = X / _CFG.resid_std[None, :]                # common within-regime scale
    out[ALL_SENSORS] = X
    return out


def load_split(name: str) -> pd.DataFrame:
    """name in {'train','test'}; adds regime label and life-fraction d."""
    df = _read(name, _CFG.fd)
    df["regime"] = _assign_regime(df)
    maxc = df.groupby("unit_id")["cycle"].transform("max")
    df["d"] = df["cycle"] / maxc
    return df


def load_rul() -> pd.DataFrame:
    return pd.read_parquet(
        os.path.join(HERE, os.pardir, "Data", f"RUL_FD00{_CFG.fd}.parquet"))


def split_by_unit(df: pd.DataFrame):
    units = sorted(df["unit_id"].unique().tolist())
    tr_u, te_u = train_test_split(units, test_size=TEST_SIZE, random_state=SEED)
    tr = df[df["unit_id"].isin(tr_u)].sort_values(
        ["unit_id", "cycle"]).reset_index(drop=True)
    te = df[df["unit_id"].isin(te_u)].sort_values(
        ["unit_id", "cycle"]).reset_index(drop=True)
    return tr, te


def denoise(df: pd.DataFrame, cols: List[str] = None,
            window: int = WINDOW, causal: bool = False) -> pd.DataFrame:
    """Condition-normalise, then per-engine rolling-median trend.

    causal=True -> trailing median (no future leakage), used at forecast cut-offs.
    """
    cols = cols or _CFG.dynamic
    out = condition_normalize(df)
    for s in cols:
        if causal:
            out[s] = (out.groupby("unit_id")[s]
                      .transform(lambda v: v.rolling(window, min_periods=1).median()))
        else:
            out[s] = (out.groupby("unit_id")[s]
                      .transform(lambda v: v.rolling(window, center=True,
                                                     min_periods=1).median()))
    return out


def same_engine_mask(df: pd.DataFrame) -> np.ndarray:
    uid = df["unit_id"].to_numpy()
    return uid[:-1] == uid[1:]


# --------------------------------------------------------------------------- #
# Sensor-trend based selection (replaces the hand-coded FD001 lists)
# --------------------------------------------------------------------------- #
def _sensor_trends(df_den: pd.DataFrame) -> Dict[str, float]:
    """Mean over engines of |Pearson corr(denoised sensor, cycle)|.

    A pure degradation sensor has |corr| -> 1; a condition-only sensor (now
    residual noise) has |corr| -> 0. Robust to the number of regimes.
    """
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


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class HealthAE(nn.Module):
    def __init__(self, n_in: int, k: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_in, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16, k),
        )
        self.dec = nn.Sequential(
            nn.Linear(k, 16), nn.Tanh(),
            nn.Linear(16, 32), nn.Tanh(),
            nn.Linear(32, n_in),
        )

    def encode(self, x):
        return torch.sigmoid(self.enc(x))

    def forward(self, x):
        h = self.encode(x)
        return self.dec(h), h


@dataclass
class Manifold:
    model: HealthAE
    mu: np.ndarray
    sd: np.ndarray
    flip0: bool
    dynamic: List[str] = field(default_factory=list)

    def encode(self, df_denoised: pd.DataFrame) -> np.ndarray:
        x = ((df_denoised[self.dynamic].to_numpy() - self.mu)
             / self.sd).astype(np.float32)
        with torch.no_grad():
            h = self.model.encode(torch.tensor(x)).numpy()
        if self.flip0:
            h = h.copy()
            h[:, 0] = 1.0 - h[:, 0]
        return h

    def decode(self, h_oriented: np.ndarray) -> np.ndarray:
        """h (oriented frame) -> sensors in the condition-NORMALISED frame."""
        h = np.asarray(h_oriented, dtype=np.float32).reshape(-1, K)
        if self.flip0:
            h = h.copy()
            h[:, 0] = 1.0 - h[:, 0]
        with torch.no_grad():
            x = self.model.dec(torch.tensor(h)).numpy()
        return x * self.sd + self.mu


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _train(train_den: pd.DataFrame) -> Manifold:
    dyn = _CFG.dynamic
    mu = train_den[dyn].mean().to_numpy()
    sd = train_den[dyn].std().to_numpy() + 1e-12
    X = ((train_den[dyn].to_numpy() - mu) / sd).astype(np.float32)
    Xt = torch.tensor(X)
    w = torch.tensor((_CFG.weights / _CFG.weights.sum()
                      * len(_CFG.weights)).astype(np.float32))
    mask = torch.tensor(same_engine_mask(train_den))

    model = HealthAE(len(dyn), K)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        opt.zero_grad()
        recon, h = model(Xt)
        rec = (w * (recon - Xt) ** 2).mean()
        h0 = h[:, 0]
        dh = h0[1:] - h0[:-1]
        mono = torch.relu(-dh)[mask].mean()
        smooth = (dh[mask] ** 2).mean()
        loss = rec + LAMBDA_MONO * mono + LAMBDA_SMOOTH * smooth
        loss.backward()
        opt.step()
    model.eval()

    with torch.no_grad():
        h0 = model.encode(Xt).numpy()[:, 0]
    corr = np.corrcoef(h0, train_den["cycle"].to_numpy())[0, 1]
    flip0 = bool(corr < 0)
    return Manifold(model=model, mu=mu, sd=sd, flip0=flip0, dynamic=list(dyn))


def get_manifold(retrain: bool = False) -> Manifold:
    os.makedirs(_CFG.art_dir, exist_ok=True)
    sd_path = os.path.join(_CFG.art_dir, "manifold_k2.pt")
    st_path = os.path.join(_CFG.art_dir, "norm_stats.npz")
    if (not retrain) and os.path.exists(sd_path) and os.path.exists(st_path):
        stats = np.load(st_path, allow_pickle=True)
        model = HealthAE(len(_CFG.dynamic), K)
        model.load_state_dict(torch.load(sd_path))
        model.eval()
        return Manifold(model=model, mu=stats["mu"], sd=stats["sd"],
                        flip0=bool(stats["flip0"]), dynamic=list(_CFG.dynamic))

    df = load_split("train")
    tr, _ = split_by_unit(df)
    tr_den = denoise(tr)
    man = _train(tr_den)
    torch.save(man.model.state_dict(), sd_path)
    np.savez(st_path, mu=man.mu, sd=man.sd, flip0=man.flip0)
    return man


# --------------------------------------------------------------------------- #
# configure(): build (or load) everything a dataset needs
# --------------------------------------------------------------------------- #
def configure(fd: int, retrain: bool = False) -> None:
    """Select dataset FD00<fd>, fit the regime model + normalisation, and pick
    the dynamic / informative sensor sets. Must be called before any other API.
    """
    global _CFG, FD, DYNAMIC, INFORMATIVE, ART_DIR

    art_dir = os.path.join(HERE, "artifacts", f"FD00{fd}")
    fig_d = os.path.join(HERE, os.pardir, "docs", "figures", f"FD00{fd}")
    os.makedirs(art_dir, exist_ok=True)
    os.makedirs(fig_d, exist_ok=True)

    tr = _read("train", fd)
    nreg = min(6, tr[SETTINGS].round(0).drop_duplicates().shape[0])
    km = KMeans(n_clusters=nreg, n_init=10, random_state=SEED).fit(
        tr[SETTINGS].to_numpy())
    reg = km.predict(tr[SETTINGS].to_numpy())

    X = tr[ALL_SENSORS].to_numpy().astype(float)
    reg_mean = np.zeros((nreg, len(ALL_SENSORS)))
    for r in range(nreg):
        m = reg == r
        reg_mean[r] = X[m].mean(0) if m.any() else 0.0
    resid = X - reg_mean[reg]
    resid_std = resid.std(0)
    global_std = X.std(0)
    # floor: keep normalisation well-defined; constants collapse via trend test
    resid_std = np.maximum(resid_std, 1e-9)

    # provisional config so denoise() works during sensor selection
    _CFG = _Config(fd=fd, n_regimes=nreg, km=km, reg_mean=reg_mean,
                   resid_std=resid_std, global_std=global_std,
                   dynamic=ALL_SENSORS, informative=[], trend={},
                   weights=np.ones(len(ALL_SENSORS)),
                   art_dir=art_dir, fig_dir=fig_d)

    tr_df = load_split("train")
    tr_den = denoise(tr_df, cols=ALL_SENSORS)
    trend = _sensor_trends(tr_den)

    dynamic = [s for s in ALL_SENSORS if trend[s] >= TREND_DYNAMIC]
    informative = [s for s in ALL_SENSORS if trend[s] >= TREND_INFORMATIVE]
    if len(dynamic) < 3:                       # safety net for hard datasets
        dynamic = sorted(ALL_SENSORS, key=lambda s: -trend[s])[:8]
        informative = [s for s in dynamic if trend[s] >= 0.3] or dynamic[:5]
    weights = np.array([max(trend[s], 0.05) for s in dynamic])

    _CFG.dynamic = dynamic
    _CFG.informative = informative
    _CFG.trend = trend
    _CFG.weights = weights

    FD = fd
    DYNAMIC = dynamic
    INFORMATIVE = informative
    ART_DIR = art_dir

    if retrain:
        get_manifold(retrain=True)


# --------------------------------------------------------------------------- #
# Helpers shared by the experiments
# --------------------------------------------------------------------------- #
def per_engine_health(man: Manifold, df_den: pd.DataFrame) -> pd.DataFrame:
    h = man.encode(df_den)
    out = df_den[["unit_id", "cycle"]].copy()
    if "d" in df_den:
        out["d"] = df_den["d"].to_numpy()
    out["h0"] = h[:, 0]
    out["h1"] = h[:, 1]
    return out


def fig_dir() -> str:
    os.makedirs(_CFG.fig_dir, exist_ok=True)
    return _CFG.fig_dir


def discovery_info() -> dict:
    return dict(fd=_CFG.fd, n_regimes=_CFG.n_regimes,
                n_dynamic=len(_CFG.dynamic), n_informative=len(_CFG.informative),
                dynamic=list(_CFG.dynamic), informative=list(_CFG.informative),
                trend=_CFG.trend)


def r2_pooled(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))
