from __future__ import annotations

import os

import numpy as np
import torch

from .config import EPOCHS, K, LAMBDA_MONO, LAMBDA_SMOOTH, LR
from .data import load_split, split_by_unit
from .denoise import denoise, same_engine_mask
from .model import HealthAE, Manifold
from .state import require_cfg


def _train(train_den) -> Manifold:
    cfg = require_cfg()
    dyn = cfg.dynamic
    mu = train_den[dyn].mean().to_numpy()
    sd = train_den[dyn].std().to_numpy() + 1e-12
    x = ((train_den[dyn].to_numpy() - mu) / sd).astype(np.float32)
    xt = torch.tensor(x)
    w = torch.tensor((cfg.weights / cfg.weights.sum() * len(cfg.weights)).astype(np.float32))
    mask = torch.tensor(same_engine_mask(train_den))

    model = HealthAE(len(dyn), K)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        opt.zero_grad()
        recon, h = model(xt)
        rec = (w * (recon - xt) ** 2).mean()
        h0 = h[:, 0]
        dh = h0[1:] - h0[:-1]
        mono = torch.relu(-dh)[mask].mean()
        smooth = (dh[mask] ** 2).mean()
        loss = rec + LAMBDA_MONO * mono + LAMBDA_SMOOTH * smooth
        loss.backward()
        opt.step()
    model.eval()

    with torch.no_grad():
        h0 = model.encode(xt).numpy()[:, 0]
    corr = np.corrcoef(h0, train_den["cycle"].to_numpy())[0, 1]
    flip0 = bool(corr < 0)
    return Manifold(model=model, mu=mu, sd=sd, flip0=flip0, dynamic=list(dyn))


def get_manifold(retrain: bool = False) -> Manifold:
    cfg = require_cfg()
    os.makedirs(cfg.model_dir, exist_ok=True)
    sd_path = os.path.join(cfg.model_dir, "manifold_k2.pt")
    st_path = os.path.join(cfg.model_dir, "norm_stats.npz")

    if (not retrain) and os.path.exists(sd_path) and os.path.exists(st_path):
        stats = np.load(st_path, allow_pickle=True)
        model = HealthAE(len(cfg.dynamic), K)
        model.load_state_dict(torch.load(sd_path))
        model.eval()
        return Manifold(
            model=model,
            mu=stats["mu"],
            sd=stats["sd"],
            flip0=bool(stats["flip0"]),
            dynamic=list(cfg.dynamic),
        )

    df = load_split("train")
    tr, _ = split_by_unit(df)
    tr_den = denoise(tr)
    man = _train(tr_den)
    torch.save(man.model.state_dict(), sd_path)
    np.savez(st_path, mu=man.mu, sd=man.sd, flip0=man.flip0)
    return man
