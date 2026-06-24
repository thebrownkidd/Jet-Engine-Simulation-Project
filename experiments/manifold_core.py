"""Shared core for the rollout / forecasting / RUL experiments.

This module re-builds and caches the k=2 physics-constrained health manifold
from `fd001_thermo_health_manifold.py` and exposes a clean API so the three
experiment scripts share *exactly* the same encoder, decoder, standardization
statistics, and denoising convention.

Public API
----------
get_manifold(retrain=False) -> Manifold
    Train (or load cached) the k=2 health manifold. Returns a `Manifold` with
    .encode(df_denoised), .decode(h), and the standardization stats.
load_split(name) -> DataFrame                 # 'train' | 'test'
denoise(df, window=15, causal=False)          # per-engine rolling median
split_by_unit(df)                             # engine-disjoint 80/20

Design notes (skeptical-reviewer friendly)
------------------------------------------
* Standardization stats (mu, sd) come from the TRAIN denoised sensors only.
* The encoder/decoder weights are frozen after training and reused everywhere.
* `causal=True` denoising uses a trailing median (no future leakage) and is
  what the RUL experiment uses at its forecast cutoff.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

T_REF = 518.67
ALL_SENSORS = [f"s{i}" for i in range(1, 22)]
STATIONARY = ["s1", "s5", "s10", "s16", "s18", "s19"]
DYNAMIC = [s for s in ALL_SENSORS if s not in STATIONARY]
INFORMATIVE = ["s2", "s3", "s4", "s7", "s8", "s9", "s11",
               "s12", "s13", "s14", "s15", "s17", "s20", "s21"]

WINDOW = 15
K = 2
EPOCHS = 4000
LR = 5e-3
LAMBDA_MONO = 5.0
LAMBDA_SMOOTH = 2.0
TEST_SIZE = 0.2

HERE = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(HERE, "artifacts")
SUMMARY_JSON = os.path.join(
    HERE, os.pardir, "physics_hypothesis_outputs_v3", "summary_v3.json")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _parquet(name: str) -> str:
    return os.path.join(HERE, os.pardir, "Data", f"{name}_FD001.parquet")


def load_split(name: str) -> pd.DataFrame:
    """name in {'train', 'test'}; adds life fraction d (train only is exact)."""
    df = pd.read_parquet(_parquet(name)).sort_values(
        ["unit_id", "cycle"]).reset_index(drop=True)
    maxc = df.groupby("unit_id")["cycle"].transform("max")
    df["d"] = df["cycle"] / maxc
    return df


def load_rul() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(HERE, os.pardir, "Data", "RUL_FD001.parquet"))


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
    """Per-engine rolling-median trend. causal=True -> trailing (no leakage)."""
    cols = cols or DYNAMIC
    out = df.copy()
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
    flip0: bool          # whether to present h0 as (1 - h0) so it increases

    # ---- encode / decode (h presented in the *oriented* frame) ---------- #
    def encode(self, df_denoised: pd.DataFrame) -> np.ndarray:
        x = ((df_denoised[DYNAMIC].to_numpy() - self.mu) / self.sd).astype(np.float32)
        with torch.no_grad():
            h = self.model.encode(torch.tensor(x)).numpy()
        if self.flip0:
            h = h.copy()
            h[:, 0] = 1.0 - h[:, 0]
        return h

    def decode(self, h_oriented: np.ndarray) -> np.ndarray:
        """h in the oriented frame -> sensors in ORIGINAL physical units."""
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
def _ceiling_weights() -> np.ndarray:
    with open(SUMMARY_JSON, "r", encoding="utf-8") as fh:
        ceil = json.load(fh)["ceilings"]
    return np.array([max(ceil.get(s, 0.05), 0.05) for s in DYNAMIC])


def _train(train_den: pd.DataFrame) -> Manifold:
    mu = train_den[DYNAMIC].mean().to_numpy()
    sd = train_den[DYNAMIC].std().to_numpy() + 1e-12
    X = ((train_den[DYNAMIC].to_numpy() - mu) / sd).astype(np.float32)
    Xt = torch.tensor(X)
    w = _ceiling_weights()
    w = torch.tensor((w / w.sum() * len(w)).astype(np.float32))
    mask = torch.tensor(same_engine_mask(train_den))

    model = HealthAE(len(DYNAMIC), K)
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

    # orient h0 so that it INCREASES with cycle (wear up). Decided on train.
    with torch.no_grad():
        h0 = model.encode(Xt).numpy()[:, 0]
    corr = np.corrcoef(h0, train_den["cycle"].to_numpy())[0, 1]
    flip0 = corr < 0
    return Manifold(model=model, mu=mu, sd=sd, flip0=bool(flip0))


def get_manifold(retrain: bool = False) -> Manifold:
    os.makedirs(ART_DIR, exist_ok=True)
    sd_path = os.path.join(ART_DIR, "manifold_k2.pt")
    st_path = os.path.join(ART_DIR, "norm_stats.npz")
    if (not retrain) and os.path.exists(sd_path) and os.path.exists(st_path):
        stats = np.load(st_path)
        model = HealthAE(len(DYNAMIC), K)
        model.load_state_dict(torch.load(sd_path))
        model.eval()
        return Manifold(model=model, mu=stats["mu"], sd=stats["sd"],
                        flip0=bool(stats["flip0"]))

    df = load_split("train")
    tr, _ = split_by_unit(df)
    tr_den = denoise(tr)
    man = _train(tr_den)
    torch.save(man.model.state_dict(), sd_path)
    np.savez(st_path, mu=man.mu, sd=man.sd, flip0=man.flip0)
    return man


# --------------------------------------------------------------------------- #
# Small helpers shared by experiments
# --------------------------------------------------------------------------- #
def per_engine_health(man: Manifold, df_den: pd.DataFrame) -> pd.DataFrame:
    """Return [unit_id, cycle, d, h0, h1] for a denoised dataframe."""
    h = man.encode(df_den)
    out = df_den[["unit_id", "cycle"]].copy()
    if "d" in df_den:
        out["d"] = df_den["d"].to_numpy()
    out["h0"] = h[:, 0]
    out["h1"] = h[:, 1]
    return out


def fig_dir() -> str:
    d = os.path.join(HERE, os.pardir, "docs", "figures")
    os.makedirs(d, exist_ok=True)
    return d


def r2_pooled(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))
