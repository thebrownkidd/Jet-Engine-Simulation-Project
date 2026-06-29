"""
ACML upgrade — shared utilities.

This module provides the *theory-matched* and *k-aware* building blocks that the
ACML experiment scripts depend on. It deliberately does NOT modify the core
`src/manifold` package; instead it composes the existing preprocessing
(regime normalisation, denoising, sensor selection) with new, clearly named
model/rollout variants so that no production behaviour is silently changed.

Key components
--------------
FlexAE             autoencoder with an explicit `bounded` flag. bounded=True
                   reproduces the production sigmoid latent in (0,1)^k;
                   bounded=False removes the sigmoid -> unbounded latent.

TrainedAE          wrapper exposing k-aware `encode`/`decode` for arbitrary k,
                   with orientation handled per bounded/unbounded case.

train_flex_ae      trains a FlexAE on denoised training data using the active
                   `mc.configure(...)` context (dynamic sensors + weights),
                   with a fixed seed and configurable epoch budget.

rollout_latent     latent constant-velocity rollout with three named projection
                   modes:
                     - "h0_clip"  : legacy production behaviour (clip coord 0)
                     - "full_box" : theory-matched projection of ALL coords
                     - "none"     : no projection (for unbounded AE contrast)

Metric helpers     recon_r2, curvature_kappa, mono_violation_fraction,
                   forecast_skill_cv, freerun_growth, rollout_nrmse_by_horizon,
                   rul_metrics_kaware.

All horizons, windows and thresholds match the existing experiments so results
are comparable to the production pipeline.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import manifold as mc  # noqa: E402

# --------------------------------------------------------------------------- #
# Shared constants (kept consistent with the production experiments)
# --------------------------------------------------------------------------- #
ACML_EPOCHS = 1500          # reduced, documented budget for sweeps (prod=4000)
LR = 5e-3
VEL_WINDOW = 25             # trailing window for health velocity (RUL)
ROLLOUT_VEL_WINDOW = 20     # window for rollout velocity
RUL_CAP = 125
CUTOFF_FRAC = 0.40
FREE_STEPS = 400
SCORE_H = [1, 10, 25, 50]
MAX_H = 200
BOUNDED_GROWTH_THRESH = 5.0   # freerun growth below this => "bounded" flag

ACML_TAB = os.path.join(ROOT, "results", "acml", "tables")
ACML_FIG = os.path.join(ROOT, "results", "acml", "figures")
ACML_MODEL = os.path.join(ROOT, "results", "acml", "models")
for _d in (ACML_TAB, ACML_FIG, ACML_MODEL):
    os.makedirs(_d, exist_ok=True)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class FlexAE(nn.Module):
    """Same architecture as the production HealthAE, with a `bounded` switch.

    bounded=True  -> latent = sigmoid(enc(x)) in (0,1)^k   (production)
    bounded=False -> latent = enc(x)          in R^k       (unbounded variant)
    """

    def __init__(self, n_in: int, k: int, bounded: bool = True):
        super().__init__()
        self.bounded = bounded
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
        z = self.enc(x)
        return torch.sigmoid(z) if self.bounded else z

    def forward(self, x):
        h = self.encode(x)
        return self.dec(h), h


@dataclass
class TrainedAE:
    """k-aware wrapper around a trained FlexAE (bounded or unbounded)."""

    model: FlexAE
    mu: np.ndarray
    sd: np.ndarray
    flip0: bool
    dynamic: List[str]
    k: int
    bounded: bool

    def encode(self, df_denoised: pd.DataFrame) -> np.ndarray:
        x = ((df_denoised[self.dynamic].to_numpy() - self.mu) / self.sd).astype(np.float32)
        with torch.no_grad():
            h = self.model.encode(torch.tensor(x)).numpy()
        if self.flip0:
            h = h.copy()
            h[:, 0] = self._flip(h[:, 0])
        return h

    def decode(self, h_oriented: np.ndarray) -> np.ndarray:
        h = np.asarray(h_oriented, dtype=np.float32).reshape(-1, self.k)
        if self.flip0:
            h = h.copy()
            h[:, 0] = self._flip(h[:, 0])
        with torch.no_grad():
            x = self.model.dec(torch.tensor(h)).numpy()
        return x * self.sd + self.mu

    def _flip(self, c0: np.ndarray) -> np.ndarray:
        # bounded latent lives in (0,1) -> reflect; unbounded -> negate.
        return (1.0 - c0) if self.bounded else (-c0)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_flex_ae(tr_den: pd.DataFrame, *, k: int, bounded: bool,
                  lambda_mono: float, lambda_smooth: float, seed: int,
                  epochs: int = ACML_EPOCHS) -> TrainedAE:
    """Train a FlexAE on denoised training data using the active mc context.

    The monotonicity/smoothness penalties act on the primary latent coordinate
    only, exactly as in production. Penalties are well-defined for both the
    bounded and unbounded parameterisations.
    """
    cfg = mc.require_cfg() if hasattr(mc, "require_cfg") else None
    # dynamic + weights come from the configured context
    dyn = mc.DYNAMIC
    from manifold.state import require_cfg
    cfg = require_cfg()
    weights = cfg.weights

    mu = tr_den[dyn].mean().to_numpy()
    sd = tr_den[dyn].std().to_numpy() + 1e-12
    x = ((tr_den[dyn].to_numpy() - mu) / sd).astype(np.float32)
    xt = torch.tensor(x)
    w = torch.tensor((weights / weights.sum() * len(weights)).astype(np.float32))
    uid = tr_den["unit_id"].to_numpy()
    mask = torch.tensor(uid[:-1] == uid[1:])

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FlexAE(len(dyn), k, bounded=bounded)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        opt.zero_grad()
        recon, h = model(xt)
        rec = (w * (recon - xt) ** 2).mean()
        h0 = h[:, 0]
        dh = h0[1:] - h0[:-1]
        mono = torch.relu(-dh)[mask].mean()
        smooth = (dh[mask] ** 2).mean()
        loss = rec + lambda_mono * mono + lambda_smooth * smooth
        loss.backward()
        opt.step()
    model.eval()

    with torch.no_grad():
        h0 = model.encode(xt).numpy()[:, 0]
    corr = np.corrcoef(h0, tr_den["cycle"].to_numpy())[0, 1]
    flip0 = bool(corr < 0)
    return TrainedAE(model=model, mu=mu, sd=sd, flip0=flip0,
                     dynamic=list(dyn), k=k, bounded=bounded)


# --------------------------------------------------------------------------- #
# k-aware health features
# --------------------------------------------------------------------------- #
def _trailing_slope(y: np.ndarray, w: int) -> np.ndarray:
    n = len(y)
    out = np.zeros(n)
    for t in range(n):
        lo = max(0, t - w + 1)
        seg = y[lo:t + 1]
        m = len(seg)
        if m < 2:
            continue
        tau = np.arange(m, dtype=float)
        tau -= tau.mean()
        denom = (tau * tau).sum()
        out[t] = float((tau * (seg - seg.mean())).sum() / denom) if denom > 0 else 0.0
    return out


def kaware_health(man: TrainedAE, df_den: pd.DataFrame) -> pd.DataFrame:
    """Return per-cycle latent coordinates h0..h(k-1) for every engine."""
    h = man.encode(df_den)
    out = df_den[["unit_id", "cycle"]].copy()
    for j in range(man.k):
        out[f"h{j}"] = h[:, j]
    return out


def kaware_features(man: TrainedAE, df_den: pd.DataFrame) -> pd.DataFrame:
    """Per-cycle causal health state + velocity for ALL k coordinates."""
    H = kaware_health(man, df_den)
    rows = []
    for _, g in H.sort_values("cycle").groupby("unit_id"):
        g = g.copy()
        for j in range(man.k):
            g[f"v{j}"] = _trailing_slope(g[f"h{j}"].to_numpy(), VEL_WINDOW)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def feature_columns(k: int) -> List[str]:
    return [f"h{j}" for j in range(k)] + [f"v{j}" for j in range(k)]


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #
def rollout_latent(man: TrainedAE, h_hist: np.ndarray, steps: int,
                   projection: str = "full_box") -> np.ndarray:
    """k-aware constant-velocity latent rollout, then decode to raw sensors.

    projection:
      "h0_clip"  legacy: clip ONLY coordinate 0 to [0, 1.5]
      "full_box" theory-matched: project ALL coords to [0, 1]
      "none"     no projection (used for the unbounded-AE contrast)
    """
    k = man.k
    w = min(ROLLOUT_VEL_WINDOW, len(h_hist) - 1)
    if w < 1:
        v = np.zeros(k)
    else:
        recent = h_hist[-w - 1:]
        t = np.arange(len(recent))
        v = np.array([np.polyfit(t, recent[:, j], 1)[0] for j in range(k)])
    h0 = h_hist[-1]
    future_h = np.array([h0 + v * (s + 1) for s in range(steps)])
    if projection == "h0_clip":
        future_h[:, 0] = np.clip(future_h[:, 0], 0.0, 1.5)
    elif projection == "full_box":
        future_h = np.clip(future_h, 0.0, 1.0)
    elif projection == "none":
        pass
    else:
        raise ValueError(f"unknown projection mode: {projection}")
    return man.decode(future_h)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def recon_r2(man: TrainedAE, te_den: pd.DataFrame) -> tuple[float, float]:
    h = man.encode(te_den)
    recon = man.decode(h)
    idx = [man.dynamic.index(s) for s in mc.INFORMATIVE]
    true = te_den[mc.INFORMATIVE].to_numpy()
    r2s = [mc.r2_pooled(true[:, j], recon[:, kk]) for j, kk in enumerate(idx)]
    return float(np.mean(r2s)), float(np.min(r2s))


def curvature_kappa(man: TrainedAE, te_den: pd.DataFrame) -> float:
    """Median |2nd difference| of the primary health coordinate (robust)."""
    H = kaware_health(man, te_den)
    second_diffs = []
    for _, g in H.sort_values("cycle").groupby("unit_id"):
        h0 = g["h0"].to_numpy()
        if len(h0) >= 3:
            second_diffs.append(np.abs(np.diff(h0, 2)))
    if not second_diffs:
        return float("nan")
    return float(np.median(np.concatenate(second_diffs)))


def mono_violation_fraction(man: TrainedAE, te_den: pd.DataFrame) -> float:
    H = kaware_health(man, te_den)
    viols, total = 0, 0
    for _, g in H.sort_values("cycle").groupby("unit_id"):
        dh = np.diff(g["h0"].to_numpy())
        viols += int((dh < 0).sum())
        total += len(dh)
    return float(viols / max(total, 1))


def forecast_skill_cv(man: TrainedAE, te_den: pd.DataFrame, horizon: int = 20) -> float:
    """Constant-velocity forecast skill of h0 vs persistence at `horizon`."""
    H = kaware_health(man, te_den)
    mse_cv, mse_pers, n = 0.0, 0.0, 0
    for _, g in H.sort_values("cycle").groupby("unit_id"):
        y = g["h0"].to_numpy()
        T = len(y)
        for c0 in [int(0.5 * T), int(0.65 * T), int(0.8 * T)]:
            if c0 < ROLLOUT_VEL_WINDOW + 1 or c0 + horizon >= T:
                continue
            lo = max(0, c0 - ROLLOUT_VEL_WINDOW)
            t = np.arange(lo, c0)
            seg = y[lo:c0]
            vel = np.polyfit(t, seg, 1)[0] if len(seg) >= 2 else 0.0
            pred_cv = y[c0 - 1] + vel * horizon
            pred_pers = y[c0 - 1]
            target = y[c0 - 1 + horizon]
            mse_cv += (pred_cv - target) ** 2
            mse_pers += (pred_pers - target) ** 2
            n += 1
    if n == 0 or mse_pers == 0:
        return float("nan")
    return float(1.0 - mse_cv / mse_pers)


def _fit_sensor_var(tr_den: pd.DataFrame):
    mu = tr_den[mc.DYNAMIC].mean().to_numpy()
    sd = tr_den[mc.DYNAMIC].std().to_numpy() + 1e-12
    Xs, Ys = [], []
    for _, g in tr_den.groupby("unit_id"):
        z = (g[mc.DYNAMIC].to_numpy() - mu) / sd
        Xs.append(z[:-1])
        Ys.append(z[1:])
    reg = LinearRegression().fit(np.vstack(Xs), np.vstack(Ys))
    return mu, sd


def freerun_growth(man: TrainedAE, tr_den: pd.DataFrame, te_den: pd.DataFrame,
                   projection: str) -> tuple[float, float, bool]:
    """Standardized state-norm growth over FREE_STEPS closed-loop steps.

    Returns (freerun_norm_last, growth, bounded_flag).
    """
    mu, sd = _fit_sensor_var(tr_den)
    longest = max(te_den["unit_id"].unique(),
                  key=lambda u: (te_den["unit_id"] == u).sum())
    g = te_den[te_den["unit_id"] == longest].sort_values("cycle").reset_index(drop=True)
    c0 = int(CUTOFF_FRAC * len(g))
    h_hist = man.encode(g.iloc[:c0 + 1])
    roll = rollout_latent(man, h_hist, FREE_STEPS, projection=projection)
    norms = np.array([float(np.linalg.norm((roll[s] - mu) / sd)) for s in range(FREE_STEPS)])
    growth = float(norms[-1] / (norms[0] + 1e-12))
    return float(norms[-1]), growth, bool(growth < BOUNDED_GROWTH_THRESH)


def latent_freerun_growth(man: TrainedAE, te_den: pd.DataFrame,
                          projection: str, steps: int = FREE_STEPS) -> tuple[float, float, bool]:
    """LATENT-space norm growth over the rollout (architectural boundedness).

    This isolates the bounded-geometry claim from the bounded-output (tanh)
    decoder: a bounded latent projected to [0,1]^k has norm <= sqrt(k) for ALL
    horizons, whereas an unbounded latent under constant-velocity extrapolation
    grows linearly without limit. Returns (latent_norm_last, growth, bounded).
    """
    longest = max(te_den["unit_id"].unique(),
                  key=lambda u: (te_den["unit_id"] == u).sum())
    g = te_den[te_den["unit_id"] == longest].sort_values("cycle").reset_index(drop=True)
    c0 = int(CUTOFF_FRAC * len(g))
    h_hist = man.encode(g.iloc[:c0 + 1])
    k = man.k
    w = min(ROLLOUT_VEL_WINDOW, len(h_hist) - 1)
    if w < 1:
        v = np.zeros(k)
    else:
        recent = h_hist[-w - 1:]
        t = np.arange(len(recent))
        v = np.array([np.polyfit(t, recent[:, j], 1)[0] for j in range(k)])
    h0 = h_hist[-1]
    future_h = np.array([h0 + v * (s + 1) for s in range(steps)])
    if projection == "h0_clip":
        future_h[:, 0] = np.clip(future_h[:, 0], 0.0, 1.5)
    elif projection == "full_box":
        future_h = np.clip(future_h, 0.0, 1.0)
    norms = np.linalg.norm(future_h, axis=1)
    growth = float(norms[-1] / (norms[0] + 1e-12))
    return float(norms[-1]), growth, bool(growth < BOUNDED_GROWTH_THRESH)


def rollout_nrmse_by_horizon(man: TrainedAE, tr_den: pd.DataFrame,
                             te_den: pd.DataFrame, projection: str) -> Dict[int, float]:
    """Cross-engine NRMSE (informative-sensor std units) at SCORE_H horizons."""
    inf_idx = [mc.DYNAMIC.index(s) for s in mc.INFORMATIVE]
    sigma = tr_den[mc.INFORMATIVE].std().to_numpy() + 1e-9
    per_h = {h: {"T": [], "P": []} for h in SCORE_H}
    for _, g in te_den.groupby("unit_id"):
        g = g.sort_values("cycle").reset_index(drop=True)
        n = len(g)
        c0 = int(CUTOFF_FRAC * n)
        if c0 < ROLLOUT_VEL_WINDOW + 2 or c0 >= n - 5:
            continue
        steps = min(n - 1 - c0, MAX_H)
        truth = g[mc.DYNAMIC].to_numpy()
        roll = rollout_latent(man, man.encode(g.iloc[:c0 + 1]), steps, projection=projection)
        for h in SCORE_H:
            if h <= steps:
                per_h[h]["T"].append(truth[c0 + h, inf_idx])
                per_h[h]["P"].append(roll[h - 1, inf_idx])
    out = {}
    for h in SCORE_H:
        if len(per_h[h]["T"]) >= 5:
            T = np.array(per_h[h]["T"]) / sigma
            P = np.array(per_h[h]["P"]) / sigma
            out[h] = float(np.sqrt(np.mean((T - P) ** 2)))
        else:
            out[h] = float("nan")
    return out


def nasa_score(err: np.ndarray) -> float:
    err = np.asarray(err, float)
    s = np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)
    return float(s.sum())


def rul_metrics_kaware(man: TrainedAE, seed: int = 42) -> dict:
    """k-aware RUL: feature vector = [h0..h(k-1), v0..v(k-1)] (no discarding)."""
    feat = feature_columns(man.k)
    train = mc.load_split("train")
    train_den = mc.denoise(train, causal=True)
    Ftr = kaware_features(man, train_den)
    maxc = Ftr.groupby("unit_id")["cycle"].transform("max")
    Ftr["rul"] = np.minimum(maxc - Ftr["cycle"], RUL_CAP)

    reg = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                        max_iter=400, l2_regularization=1.0,
                                        random_state=seed)
    reg.fit(Ftr[feat].to_numpy(), Ftr["rul"].to_numpy())

    test = mc.load_split("test")
    test_den = mc.denoise(test, causal=True)
    Fte = kaware_features(man, test_den)
    rul_true = mc.load_rul()["rul"].to_numpy()
    units = sorted(Fte["unit_id"].unique().tolist())
    last = (Fte.sort_values("cycle").groupby("unit_id").tail(1)
            .set_index("unit_id").loc[units])
    pred = np.clip(reg.predict(last[feat].to_numpy()), 0, RUL_CAP)
    base = np.full_like(rul_true, float(Ftr["rul"].mean()), dtype=float)
    err = pred - rul_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    r2 = mc.r2_pooled(rul_true, pred)
    base_rmse = float(np.sqrt(np.mean((base - rul_true) ** 2)))
    return dict(rul_rmse=rmse, rul_mae=mae, rul_r2=r2,
                rul_nasa=nasa_score(err), base_rmse=base_rmse)


# --------------------------------------------------------------------------- #
# Shared-sensor-set helpers (Task 7: specialist vs generalized)
# --------------------------------------------------------------------------- #
def _fit_var_shared(tr_den: pd.DataFrame, shared: List[str]):
    mu = tr_den[shared].mean().to_numpy()
    sd = tr_den[shared].std().to_numpy() + 1e-12
    return mu, sd


def freerun_growth_shared(man: TrainedAE, tr_den: pd.DataFrame, te_den: pd.DataFrame,
                          shared: List[str], projection: str) -> tuple[float, float, bool]:
    """Decoded free-run growth over a shared sensor set (Task 7)."""
    mu, sd = _fit_var_shared(tr_den, shared)
    longest = max(te_den["unit_id"].unique(),
                  key=lambda u: (te_den["unit_id"] == u).sum())
    g = te_den[te_den["unit_id"] == longest].sort_values("cycle").reset_index(drop=True)
    c0 = int(CUTOFF_FRAC * len(g))
    h_hist = man.encode(g.iloc[:c0 + 1])
    roll = rollout_latent(man, h_hist, FREE_STEPS, projection=projection)
    norms = np.array([float(np.linalg.norm((roll[s] - mu) / sd)) for s in range(FREE_STEPS)])
    growth = float(norms[-1] / (norms[0] + 1e-12))
    return float(norms[-1]), growth, bool(growth < BOUNDED_GROWTH_THRESH)


def rul_metrics_shared(man: TrainedAE, shared: List[str], seed: int = 42) -> dict:
    """k-aware RUL using a shared sensor set; mc must be configured for the dataset."""
    feat = feature_columns(man.k)
    train = mc.load_split("train")
    train_den = mc.denoise(train, cols=shared, causal=True)
    Ftr = kaware_features(man, train_den)
    maxc = Ftr.groupby("unit_id")["cycle"].transform("max")
    Ftr["rul"] = np.minimum(maxc - Ftr["cycle"], RUL_CAP)
    reg = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                        max_iter=400, l2_regularization=1.0,
                                        random_state=seed)
    reg.fit(Ftr[feat].to_numpy(), Ftr["rul"].to_numpy())
    test = mc.load_split("test")
    test_den = mc.denoise(test, cols=shared, causal=True)
    Fte = kaware_features(man, test_den)
    rul_true = mc.load_rul()["rul"].to_numpy()
    units = sorted(Fte["unit_id"].unique().tolist())
    last = (Fte.sort_values("cycle").groupby("unit_id").tail(1)
            .set_index("unit_id").loc[units])
    pred = np.clip(reg.predict(last[feat].to_numpy()), 0, RUL_CAP)
    err = pred - rul_true
    return dict(rul_rmse=float(np.sqrt(np.mean(err ** 2))),
                rul_r2=mc.r2_pooled(rul_true, pred))


# --------------------------------------------------------------------------- #
# Convenience: configure + load denoised splits
# --------------------------------------------------------------------------- #
def setup_dataset(fd: int, *, k: int = 2, normalize: bool = True,
                  regime_rule: str = "heuristic"):
    """Configure mc context for FD00<fd> and return denoised train/test splits."""
    mc.configure(fd, k=k, normalize=normalize, regime_rule=regime_rule, tag=f"acml_k{k}")
    df = mc.load_split("train")
    tr, te = mc.split_by_unit(df)
    tr_den = mc.denoise(tr)
    te_den = mc.denoise(te)
    return tr_den, te_den


def latex_table(df: pd.DataFrame, caption: str, label: str,
                float_fmt: str = "%.3f") -> str:
    """Minimal booktabs LaTeX table from a DataFrame."""
    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(str(c).replace("_", r"\_") for c in cols) + r" \\",
             r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("nan" if (v != v) else (float_fmt % v))
            else:
                cells.append(str(v).replace("_", r"\_"))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)
