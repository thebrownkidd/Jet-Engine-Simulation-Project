from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import K


class HealthAE(nn.Module):
    def __init__(self, n_in: int, k: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_in, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, k),
        )
        self.dec = nn.Sequential(
            nn.Linear(k, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
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
    k: int = K

    def encode(self, df_denoised: pd.DataFrame) -> np.ndarray:
        x = ((df_denoised[self.dynamic].to_numpy() - self.mu) / self.sd).astype(np.float32)
        with torch.no_grad():
            h = self.model.encode(torch.tensor(x)).numpy()
        if self.flip0:
            h = h.copy()
            h[:, 0] = 1.0 - h[:, 0]
        return h

    def decode(self, h_oriented: np.ndarray) -> np.ndarray:
        h = np.asarray(h_oriented, dtype=np.float32).reshape(-1, self.k)
        if self.flip0:
            h = h.copy()
            h[:, 0] = 1.0 - h[:, 0]
        with torch.no_grad():
            x = self.model.dec(torch.tensor(h)).numpy()
        return x * self.sd + self.mu
