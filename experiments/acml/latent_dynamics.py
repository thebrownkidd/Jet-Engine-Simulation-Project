"""
Context-conditioned bounded latent dynamics — reusable head + baselines.

This module replaces the constant-velocity (CV) latent rollout used in the
external-dataset experiments with a *learned* residual dynamics model operating
in the frozen bounded latent space of a trained autoencoder:

    h_{t+1} = Pi_[0,1]^k ( h_t + alpha * tanh( g_psi([h_t, c_t]) ) )

where c_t are deterministic (and therefore known-into-the-future) context
features (e.g. cyclic calendar encodings), g_psi is a small residual MLP, and
Pi is the projection onto the compact latent box. The projection makes the
rollout bounded for *any* g_psi (generalised boundedness theorem), so the
scientific question shifts from "is it bounded" (trivially yes) to "is the
bounded latent space a better place to learn rollout dynamics than sensor
space, and does conditioning on context beat CV under seasonality/regimes".

Design goals
------------
* The autoencoder is trained and then FROZEN; only g_psi is learned. This keeps
  the latent geometry fixed across every ablation for a clean comparison.
* Projection modes are explicit and separable:
    - "hard" : torch.clamp inside the training graph (zero grad outside box)
    - "soft" : no clamp in graph; an out-of-box penalty supplies gradient, and
               a hard clamp is applied only at inference (recommended default)
    - "none" : no projection anywhere (the no-projection ablation / unbounded)
* Training uses one-step + multi-step (free-run) rollout loss + a small-step
  penalty, so the model is optimised for the recursive rollout it is used in.

The module is dataset-agnostic: it consumes latent trajectories produced by the
frozen AE plus optional per-row context, so the same code serves air quality,
PHM milling, C-MAPSS, etc.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Projection helpers
# --------------------------------------------------------------------------- #
def _project_torch(h_raw: torch.Tensor, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (projected_state, out_of_box_penalty) for a training step.

    "hard": clamp inside the graph -> exact boundedness but zero grad outside.
    "soft": identity in the graph + quadratic out-of-box penalty (differentiable
            everywhere); the caller applies a hard clamp only at inference.
    "none": identity, no penalty (unbounded latent dynamics).
    """
    if mode == "hard":
        return h_raw.clamp(0.0, 1.0), h_raw.new_zeros(())
    if mode == "soft":
        oob = (torch.relu(h_raw - 1.0) ** 2 + torch.relu(-h_raw) ** 2).mean()
        return h_raw, oob
    if mode == "none":
        return h_raw, h_raw.new_zeros(())
    raise ValueError(f"unknown projection mode: {mode}")


def _project_np(h_raw: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return h_raw
    return np.clip(h_raw, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ResidualLatentDynamics(nn.Module):
    """Small residual MLP dynamics: Delta h = alpha * tanh(g_psi([h, c]))."""

    def __init__(self, k: int, c_dim: int, *, hidden: int = 64,
                 alpha: float = 0.05, use_context: bool = True):
        super().__init__()
        self.k = k
        self.c_dim = c_dim
        self.use_context = bool(use_context and c_dim > 0)
        self.alpha = alpha
        in_dim = k + (c_dim if self.use_context else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, k),
        )

    def delta(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([h, c], dim=-1) if self.use_context else h
        return self.alpha * torch.tanh(self.net(inp))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class DynamicsResult:
    model: ResidualLatentDynamics
    projection: str
    train_time: float
    n_params: int
    final_loss: float


# --------------------------------------------------------------------------- #
# Trajectory assembly
# --------------------------------------------------------------------------- #
@dataclass
class Traj:
    h: np.ndarray        # (T, k) oriented latent state (in [0,1]^k for bounded AE)
    c: np.ndarray        # (T, c_dim) context (empty second dim allowed)
    x_std: np.ndarray    # (T, n) standardized true sensors (for evaluation)
    unit: object


def build_trajectories(encode_fn: Callable, df, feature_cols: List[str],
                       context_cols: List[str], mu: np.ndarray, sd: np.ndarray
                       ) -> List[Traj]:
    """Encode every unit to a latent trajectory with aligned context + truth."""
    trajs: List[Traj] = []
    for unit, g in df.groupby("unit_id"):
        g = g.sort_values("cycle")
        h = encode_fn(g).astype(np.float32)
        if context_cols:
            c = g[context_cols].to_numpy().astype(np.float32)
        else:
            c = np.zeros((len(g), 0), np.float32)
        x_std = ((g[feature_cols].to_numpy() - mu) / sd).astype(np.float32)
        trajs.append(Traj(h=h, c=c, x_std=x_std, unit=unit))
    return trajs


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _make_windows(trajs: List[Traj], horizon: int, max_windows: int,
                  seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample multi-step rollout windows across units.

    Returns (H0 (M,k), Cseq (M,horizon,c), Htgt (M,horizon,k)). The step at
    index t predicts h_{s+t+1} from state h_{s+t} and context c_{s+t}, so the
    context sequence is the *current-time* context over the window.
    """
    rng = np.random.RandomState(seed)
    k = trajs[0].h.shape[1]
    c_dim = trajs[0].c.shape[1]
    H0, Cseq, Htgt = [], [], []
    starts = []
    for ti, tr in enumerate(trajs):
        T = len(tr.h)
        if T <= horizon + 1:
            continue
        for s in range(0, T - horizon - 1):
            starts.append((ti, s))
    if not starts:
        return (np.zeros((0, k), np.float32),
                np.zeros((0, horizon, c_dim), np.float32),
                np.zeros((0, horizon, k), np.float32))
    idx = rng.permutation(len(starts))[:max_windows]
    for j in idx:
        ti, s = starts[j]
        tr = trajs[ti]
        H0.append(tr.h[s])
        Cseq.append(tr.c[s:s + horizon])
        Htgt.append(tr.h[s + 1:s + horizon + 1])
    return (np.asarray(H0, np.float32),
            np.asarray(Cseq, np.float32),
            np.asarray(Htgt, np.float32))


def train_dynamics(trajs: List[Traj], k: int, c_dim: int, *,
                   use_context: bool, projection: str,
                   hidden: int = 64, alpha: float = 0.05,
                   horizon: int = 16, multistep: bool = True,
                   beta: float = 1.0, eta_step: float = 1e-3,
                   oob_weight: float = 10.0, epochs: int = 400,
                   lr: float = 3e-3, max_windows: int = 4000,
                   seed: int = 0) -> DynamicsResult:
    """Train the residual dynamics on frozen latent trajectories.

    Loss = one-step MSE (teacher-forced)
         + beta * multi-step free-run MSE (only if `multistep`)
         + eta_step * mean ||Delta h||^2
         + oob_weight * out-of-box penalty (only for projection="soft").
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # One-step (teacher-forced) tensors across all consecutive pairs.
    Ht, Ct, Htp1 = [], [], []
    for tr in trajs:
        if len(tr.h) < 2:
            continue
        Ht.append(tr.h[:-1])
        Ct.append(tr.c[:-1])
        Htp1.append(tr.h[1:])
    Ht = torch.tensor(np.concatenate(Ht))
    Ct = torch.tensor(np.concatenate(Ct))
    Htp1 = torch.tensor(np.concatenate(Htp1))

    # Multi-step windows.
    if multistep:
        H0n, Cseqn, Htgtn = _make_windows(trajs, horizon, max_windows, seed)
        H0 = torch.tensor(H0n)
        Cseq = torch.tensor(Cseqn)
        Htgt = torch.tensor(Htgtn)
        has_win = H0.shape[0] > 0
    else:
        has_win = False

    model = ResidualLatentDynamics(k, c_dim, hidden=hidden, alpha=alpha,
                                   use_context=use_context)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    final = float("nan")
    for _ in range(epochs):
        opt.zero_grad()

        # --- one-step teacher-forced ---
        d1 = model.delta(Ht, Ct)
        h_raw1 = Ht + d1
        h1, oob1 = _project_torch(h_raw1, projection)
        loss_one = ((h1 - Htp1) ** 2).mean()
        step_pen = (d1 ** 2).mean()
        loss = loss_one + eta_step * step_pen + oob_weight * oob1

        # --- multi-step free-run ---
        if has_win:
            h = H0
            oob_acc = h.new_zeros(())
            step_acc = h.new_zeros(())
            errs = []
            for t in range(horizon):
                d = model.delta(h, Cseq[:, t, :])
                h_raw = h + d
                h, oob = _project_torch(h_raw, projection)
                errs.append(((h - Htgt[:, t, :]) ** 2).mean())
                oob_acc = oob_acc + oob
                step_acc = step_acc + (d ** 2).mean()
            loss_multi = torch.stack(errs).mean()
            loss = (loss + beta * loss_multi
                    + eta_step * step_acc / horizon
                    + oob_weight * oob_acc / horizon)

        loss.backward()
        opt.step()
        final = float(loss.detach())

    model.eval()
    return DynamicsResult(model=model, projection=projection,
                          train_time=time.time() - t0,
                          n_params=model.n_params(), final_loss=final)


# --------------------------------------------------------------------------- #
# Inference rollout (numpy)
# --------------------------------------------------------------------------- #
def dyn_rollout(res: DynamicsResult, h0: np.ndarray, c_future: np.ndarray,
                steps: int) -> np.ndarray:
    """Roll the learned dynamics forward `steps` from h0 given future context.

    A hard box projection is applied for bounded modes ("hard"/"soft"); "none"
    leaves the state unprojected. Returns an (steps, k) latent trajectory.
    """
    model = res.model
    infer_mode = "none" if res.projection == "none" else "hard"
    h = np.asarray(h0, np.float32).reshape(1, -1)
    out = np.empty((steps, model.k), np.float32)
    with torch.no_grad():
        for t in range(steps):
            ct = c_future[t] if (c_future is not None and t < len(c_future)) \
                else np.zeros(model.c_dim, np.float32)
            ct = np.asarray(ct, np.float32).reshape(1, -1)
            d = model.delta(torch.tensor(h), torch.tensor(ct)).numpy()
            h = _project_np(h + d, infer_mode)
            out[t] = h[0]
    return out


# --------------------------------------------------------------------------- #
# Latent AR(1) baseline (spectral-radius constrained)
# --------------------------------------------------------------------------- #
@dataclass
class LatentAR1:
    A: np.ndarray
    b: np.ndarray
    rho: float
    projection: str = "hard"


def fit_latent_ar1(trajs: List[Traj], *, max_rho: float = 0.99,
                   projection: str = "hard") -> LatentAR1:
    Xs, Ys = [], []
    for tr in trajs:
        if len(tr.h) < 2:
            continue
        Xs.append(tr.h[:-1])
        Ys.append(tr.h[1:])
    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    Xa = np.hstack([X, np.ones((len(X), 1))])
    W, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
    A = W[:-1].T
    b = W[-1]
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    if rho >= max_rho and rho > 0:
        A = A * (max_rho / rho)
        rho = max_rho
    return LatentAR1(A=A.astype(np.float32), b=b.astype(np.float32),
                     rho=rho, projection=projection)


def ar1_rollout(ar: LatentAR1, h0: np.ndarray, steps: int) -> np.ndarray:
    h = np.asarray(h0, np.float32).reshape(-1)
    out = np.empty((steps, len(h)), np.float32)
    for t in range(steps):
        h = ar.A @ h + ar.b
        h = _project_np(h, "none" if ar.projection == "none" else "hard")
        out[t] = h
    return out


# --------------------------------------------------------------------------- #
# Sensor-space matched-capacity MLP dynamics (fairness control)
# --------------------------------------------------------------------------- #
def train_sensor_dynamics(trajs: List[Traj], n_sensors: int, c_dim: int, *,
                          use_context: bool, hidden: int = 64,
                          alpha: float = 0.25, horizon: int = 16,
                          multistep: bool = True, beta: float = 1.0,
                          eta_step: float = 1e-3, epochs: int = 400,
                          lr: float = 3e-3, max_windows: int = 4000,
                          seed: int = 0) -> DynamicsResult:
    """Same residual-MLP dynamics, but in standardized SENSOR space (unbounded).

    This is the essential control: if the bounded-latent head beats this
    matched-capacity sensor-space head at long horizons while staying bounded,
    the win is attributable to the bounded latent *space*, not to model capacity.
    """
    # Re-use the latent trainer by swapping the "latent" arrays for sensors.
    sensor_trajs = [Traj(h=tr.x_std, c=tr.c, x_std=tr.x_std, unit=tr.unit)
                    for tr in trajs]
    return train_dynamics(sensor_trajs, n_sensors, c_dim,
                          use_context=use_context, projection="none",
                          hidden=hidden, alpha=alpha, horizon=horizon,
                          multistep=multistep, beta=beta, eta_step=eta_step,
                          oob_weight=0.0, epochs=epochs, lr=lr,
                          max_windows=max_windows, seed=seed)


def sensor_rollout(res: DynamicsResult, x0: np.ndarray, c_future: np.ndarray,
                   steps: int) -> np.ndarray:
    """Roll the sensor-space dynamics; returns (steps, n_sensors) standardized."""
    return dyn_rollout(res, x0, c_future, steps)
