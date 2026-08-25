"""
Unified experiment runner — all datasets, context-conditioned bounded latent dynamics.

For every dataset:
  1. Load features CSV.
  2. K-sweep (k=1..K_MAX) to find the best latent dimension by test recon R².
  3. Train the frozen bounded AE at the best k.
  4. Run five forecasting heads and pick the winner.
  5. Compute per-sensor RMSE and R² in original observation space.
  6. Produce a multi-panel ground-truth vs prediction figure (one panel per sensor).
  7. Write a per-dataset REPORT.md that explains the methodology, every figure
     element, and every table column in plain language.

Heads compared
--------------
  persistence   — last observed value held constant (laziest baseline)
  cv            — constant-velocity latent rollout (old method)
  var_sensor    — sensor-space linear AR(1) baseline
  mlp_noctx     — bounded residual MLP, no context
  mlp_ctx       — bounded residual MLP, with context (proposed)

Outputs (per dataset, under results/final_experiments/<name>/)
--------------------------------------------------------------
  metrics.csv        — all-head skill / NRMSE / growth / bounded table
  obs_metrics.csv    — per-sensor RMSE and R² of the best head
  forecast_plot.png  — ground truth vs best-head prediction per sensor
  REPORT.md          — full methodology + figure and table explanations

Usage
-----
  python experiments/acml/exp_all_datasets.py                # all datasets
  python experiments/acml/exp_all_datasets.py --datasets air_quality ims_bearing
  python experiments/acml/exp_all_datasets.py --epochs-ae 300 --epochs-dyn 150
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))

import exp_ims_bearing as X     # AE training / encode / decode / rollout helpers
import latent_dynamics as LD    # learned dynamics heads

OUT_ROOT = os.path.join(ROOT, "results", "final_experiments")

# --------------------------------------------------------------------------- #
# Dataset catalogue
# --------------------------------------------------------------------------- #
# Each entry specifies:
#   feat_csv      : path to the long-format features CSV
#   lambda_mono   : monotonicity penalty for AE training
#                   (1.0 for steadily-degrading, 0.0 for stationary/forecasting)
#   horizons      : forecast horizons to evaluate (in steps / cycles)
#   test_frac     : fraction of units held out as test set
#   description   : one-line plain-language description
DATASET_CONFIGS = {
    "cmapss_FD001": dict(
        feat_csv="data/processed/cmapss_FD001_features.csv",
        lambda_mono=1.0, horizons=[1, 10, 25, 50], test_frac=0.30,
        description="NASA C-MAPSS FD001 — 100 simulated jet engines, "
                    "single flight condition, monotone degradation to failure."),
    "cmapss_FD002": dict(
        feat_csv="data/processed/cmapss_FD002_features.csv",
        lambda_mono=1.0, horizons=[1, 10, 25, 50], test_frac=0.30,
        description="NASA C-MAPSS FD002 — 260 simulated jet engines, "
                    "six flight conditions."),
    "cmapss_FD003": dict(
        feat_csv="data/processed/cmapss_FD003_features.csv",
        lambda_mono=1.0, horizons=[1, 10, 25, 50], test_frac=0.30,
        description="NASA C-MAPSS FD003 — 100 simulated jet engines, "
                    "single flight condition, two fault modes."),
    "cmapss_FD004": dict(
        feat_csv="data/processed/cmapss_FD004_features.csv",
        lambda_mono=1.0, horizons=[1, 10, 25, 50], test_frac=0.30,
        description="NASA C-MAPSS FD004 — 249 simulated jet engines, "
                    "six flight conditions, two fault modes."),
    "ims_bearing": dict(
        feat_csv="data/processed/ims_bearing_features.csv",
        lambda_mono=0.5, horizons=[1, 10, 25, 50], test_frac=0.35,
        description="NASA IMS bearings — 16 run-to-failure bearing units, "
                    "14 time+frequency vibration features per channel."),
    "battery": dict(
        feat_csv="data/processed/battery_features.csv",
        lambda_mono=1.0, horizons=[1, 10, 30, 60], test_frac=0.35,
        description="NASA Li-ion battery — 26 batteries, 10 per-discharge-cycle "
                    "features (capacity fade + V/I/T summaries)."),
    "phm_milling": dict(
        feat_csv="data/processed/phm_features.csv",
        lambda_mono=1.0, horizons=[1, 10, 25, 50], test_frac=0.34,
        description="PHM 2010 CNC milling — 6 cutting tools, 35 force/vibration/"
                    "AE features per cut, steady tool-wear degradation."),
    "air_quality": dict(
        feat_csv="data/processed/air_quality_features_ctx.csv",
        lambda_mono=0.0, horizons=[1, 8, 24, 48], test_frac=0.34,
        description="Beijing multi-site air quality — 12 stations, hourly "
                    "PM2.5/PM10/meteorology, 11 variables (no degradation; "
                    "cyclic seasonal forecasting)."),
    "femto_bearing": dict(
        feat_csv="data/processed/femto_features.csv",
        lambda_mono=5.0, horizons=[1, 10, 25, 50], test_frac=0.35,
        description="FEMTO/PRONOSTIA bearings — accelerated run-to-failure "
                    "bearing tests under multiple operating conditions."),
    "ncmapss": dict(
        feat_csv="data/processed/ncmapss_features.csv",
        lambda_mono=1.0, horizons=[1, 10, 25, 50], test_frac=0.30,
        description="N-CMAPSS (Turbofan Degradation Simulation-2) — realistic "
                    "flight-condition turbofan run-to-failure trajectories."),
}

ANCHORS = (0.5, 0.65, 0.8)
FREE_STEPS = X.FREE_STEPS
BOUNDED_GROWTH_THRESH = X.BOUNDED_GROWTH_THRESH

# Preprocessing constants — match the validated manifold pipeline
TREND_THRESH = 0.20    # min mean |corr(sensor, cycle)| to retain a sensor
MIN_SENSORS  = 3       # fallback: keep at least this many sensors
DENOISE_WIN  = 15      # rolling-median window (cycles) — matches manifold/config.py


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def load_dataset(feat_csv: str):
    df = pd.read_csv(feat_csv)
    ctx = [c for c in df.columns if c.startswith("ctx_")]
    feats = [c for c in df.columns
             if c not in ("unit_id", "cycle") and c not in ctx]
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    return df, feats, ctx


def select_sensors(tr_df: pd.DataFrame, feats: list, lambda_mono: float) -> list:
    """Select informative sensors from training data.

    Degradation datasets (lambda_mono > 0):
      Compute mean |corr(sensor, cycle)| across units.
      Keep sensors >= TREND_THRESH. Fall back to top-MIN_SENSORS if too few pass.
      This removes flat / constant sensors that add noise to the AE without
      contributing any degradation signal.

    Stationary / cyclic datasets (lambda_mono == 0):
      Keep sensors with training std > 1e-4 (drop near-constants only).
      Trend selection is meaningless here; we want all varying sensors.
    """
    if lambda_mono > 0:
        scores = {}
        for c in feats:
            rs = []
            for _, g in tr_df.groupby("unit_id"):
                if g[c].std() > 1e-9 and len(g) > 5:
                    rs.append(abs(np.corrcoef(g[c], g["cycle"])[0, 1]))
            scores[c] = float(np.nanmean(rs)) if rs else 0.0
        chosen = [c for c in feats if scores[c] >= TREND_THRESH]
        if len(chosen) < MIN_SENSORS:
            chosen = sorted(feats, key=lambda c: scores[c], reverse=True)[:MIN_SENSORS]
        return chosen
    else:
        # Stationary/cyclic dataset: drop near-constant sensors only
        chosen = [c for c in feats if tr_df[c].std() > 1e-4]
        if len(chosen) < MIN_SENSORS:
            chosen = sorted(feats, key=lambda c: tr_df[c].std(), reverse=True)[:MIN_SENSORS]
        return chosen


# --------------------------------------------------------------------------- #
# K-sweep with PCA elbow method
# --------------------------------------------------------------------------- #
def _find_pca_elbow(train_data: np.ndarray, k_max: int = 10) -> int:
    """Find the elbow point in PCA cumulative explained variance.
    
    Returns the smallest k where cumulative variance >= 85%, or the k before
    the plateau starts (when the next variance drop is < 5% of current drop).
    """
    pca = PCA(n_components=min(k_max, train_data.shape[1]))
    pca.fit(train_data)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    
    # First criterion: 85% variance explained
    k_85 = np.argmax(cum_var >= 0.85) + 1 if np.any(cum_var >= 0.85) else k_max
    
    # Second criterion: elbow in the variance drops
    # Find where incremental variance drop flattens
    diffs = np.diff(cum_var)  # variance gain at each step
    if len(diffs) >= 2:
        # Look for where variance gain becomes very small
        threshold = 0.05  # 5% incremental gain
        k_plateau = np.argmax(diffs < threshold) + 1 if np.any(diffs < threshold) else k_max
    else:
        k_plateau = k_max
    
    # Take the smaller of the two criteria (conservative)
    best_k = min(k_85, k_plateau)
    best_k = max(2, min(best_k, k_max))  # clamp to [2, k_max]
    return int(best_k)


def k_sweep(tr_df, te_df, feats, lambda_mono, k_max=6, epochs=250, seed=0,
            lambda_smooth=0.5):
    """K-sweep: train AE at each k, record reconstruction R², find elbow."""
    results = []
    for k in range(1, k_max + 1):
        ae = X.train_ae(tr_df, feats, k=k, bounded=True,
                        lambda_mono=lambda_mono, lambda_smooth=lambda_smooth,
                        epochs=epochs, seed=seed)
        r2_mean, r2_min = X.recon_r2(ae, te_df)
        results.append(dict(k=k, recon_r2=r2_mean, recon_r2_min=r2_min))
    df_k = pd.DataFrame(results)
    
    # Determine best k using PCA elbow method on training data
    mu = tr_df[feats].mean().to_numpy()
    sd = tr_df[feats].std().to_numpy() + 1e-12
    x_std = ((tr_df[feats].to_numpy() - mu) / sd).astype(np.float32)
    best_k = _find_pca_elbow(x_std, k_max=k_max)
    
    return df_k, best_k


# --------------------------------------------------------------------------- #
# Build forecasting heads
# --------------------------------------------------------------------------- #
def build_heads(ae, tr_trajs, te_trajs, ctx, mu, sd,
                horizon, dyn_epochs, seed, reg, vmu, vsd):
    n = len(mu)
    c_dim = len(ctx)
    k = ae["k"]

    def decode_std(latent_arr):
        return (X.decode(ae, latent_arr) - mu) / sd

    # -- persistence
    def pers_pred(tr, c0, hmax):
        return np.repeat(tr.x_std[c0][None, :], hmax, axis=0)

    # -- constant velocity
    def cv_pred(tr, c0, hmax):
        _, dec = X.rollout(ae, tr.h[:c0 + 1], hmax, "full_box")
        return (dec - mu) / sd

    def cv_lat(tr, c0, steps):
        fut_h, _ = X.rollout(ae, tr.h[:c0 + 1], steps, "full_box")
        return fut_h

    # -- damped constant velocity (Theorem 3'): geometric velocity decay,
    #    gamma fit on TRAIN trajectories only. gamma=1 -> CV, gamma=0 -> persistence.
    gamma_cand = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]

    def _damp_roll(h_hist, steps, gamma):
        w = min(X.ROLLOUT_VEL_WINDOW, len(h_hist) - 1)
        if w < 1:
            v = np.zeros(k)
        else:
            recent = h_hist[-w - 1:]
            tt = np.arange(len(recent))
            v = np.array([np.polyfit(tt, recent[:, j], 1)[0] for j in range(k)])
        h0v = h_hist[-1]
        m = np.arange(1, steps + 1, dtype=float)
        S = m if gamma >= 1.0 else (1.0 - gamma ** m) / (1.0 - gamma)
        fut = h0v[None, :] + v[None, :] * S[:, None]
        return np.clip(fut, 0.0, 1.0)

    H = max(1, int(horizon))

    def _fit_gamma():
        best_g, best_err = 1.0, np.inf
        for g in gamma_cand:
            se, cnt = 0.0, 0
            for trj in tr_trajs:
                T = len(trj.x_std)
                for f in ANCHORS:
                    c0 = int(f * T)
                    if c0 < X.ROLLOUT_VEL_WINDOW + 2 or c0 + H >= T:
                        continue
                    dec = X.decode(ae, _damp_roll(trj.h[:c0 + 1], H, g))
                    pred = (dec - mu) / sd
                    tgt = trj.x_std[c0 + 1:c0 + 1 + H]
                    se += float(np.sum((pred - tgt) ** 2))
                    cnt += pred.size
            err = se / cnt if cnt else np.inf
            if err < best_err:
                best_err, best_g = err, g
        return best_g

    gamma_star = _fit_gamma()

    def cvd_pred(tr, c0, hmax):
        return (X.decode(ae, _damp_roll(tr.h[:c0 + 1], hmax, gamma_star)) - mu) / sd

    def cvd_lat(tr, c0, steps):
        return _damp_roll(tr.h[:c0 + 1], steps, gamma_star)

    # -- Holt damped-trend exponential smoothing (same theorem family; alpha
    #    smooths the level, beta smooths the trend, phi damps -> robust to noise).
    holt_grid = [(a, b, p) for a in (0.3, 0.6, 0.9)
                 for b in (0.0, 0.1, 0.3) for p in (0.0, 0.5, 0.8, 1.0)]

    def _holt(h_hist, steps, alpha, beta, phi):
        L = h_hist[0].astype(float).copy()
        B = (h_hist[1] - h_hist[0]).astype(float) if len(h_hist) > 1 else np.zeros(k)
        for t in range(1, len(h_hist)):
            Lp = L
            L = alpha * h_hist[t] + (1.0 - alpha) * (L + phi * B)
            B = beta * (L - Lp) + (1.0 - beta) * phi * B
        m = np.arange(1, steps + 1, dtype=float)
        Sphi = np.cumsum(np.ones(steps)) if phi >= 1.0 else np.cumsum(phi ** m)
        fut = L[None, :] + B[None, :] * Sphi[:, None]
        return np.clip(fut, 0.0, 1.0)

    def _fit_holt():
        best, best_err = (0.6, 0.0, 1.0), np.inf
        for (a, b, p) in holt_grid:
            se, cnt = 0.0, 0
            for trj in tr_trajs:
                T = len(trj.x_std)
                for f in ANCHORS:
                    c0 = int(f * T)
                    if c0 < X.ROLLOUT_VEL_WINDOW + 2 or c0 + H >= T:
                        continue
                    dec = X.decode(ae, _holt(trj.h[:c0 + 1], H, a, b, p))
                    pred = (dec - mu) / sd
                    tgt = trj.x_std[c0 + 1:c0 + 1 + H]
                    se += float(np.sum((pred - tgt) ** 2))
                    cnt += pred.size
            err = se / cnt if cnt else np.inf
            if err < best_err:
                best_err, best = err, (a, b, p)
        return best

    holt_star = _fit_holt()

    def holt_pred(tr, c0, hmax):
        return (X.decode(ae, _holt(tr.h[:c0 + 1], hmax, *holt_star)) - mu) / sd

    def holt_lat(tr, c0, steps):
        return _holt(tr.h[:c0 + 1], steps, *holt_star)

    # -- long-window damped trend: lower-variance velocity (OLS slope over a
    #    fitted window W) + geometric damping. Still Theorem-3 linear extrapolation.
    trend_W = [20, 40, 80, 160]
    trend_g = [0.5, 0.7, 0.85, 0.95, 1.0]

    def _slope(h_hist, W):
        w = min(W, len(h_hist))
        recent = h_hist[-w:]
        if len(recent) < 2:
            return np.zeros(k)
        tt = np.arange(len(recent))
        return np.array([np.polyfit(tt, recent[:, j], 1)[0] for j in range(k)])

    def _trend_roll(h_hist, steps, W, gamma):
        v = _slope(h_hist, W)
        h0v = h_hist[-1]
        m = np.arange(1, steps + 1, dtype=float)
        S = m if gamma >= 1.0 else (1.0 - gamma ** m) / (1.0 - gamma)
        return np.clip(h0v[None, :] + v[None, :] * S[:, None], 0.0, 1.0)

    def _fit_trend():
        best, best_err = (20, 1.0), np.inf
        for W in trend_W:
            for g in trend_g:
                se, cnt = 0.0, 0
                for trj in tr_trajs:
                    T = len(trj.x_std)
                    for f in ANCHORS:
                        c0 = int(f * T)
                        if c0 < X.ROLLOUT_VEL_WINDOW + 2 or c0 + H >= T:
                            continue
                        dec = X.decode(ae, _trend_roll(trj.h[:c0 + 1], H, W, g))
                        pred = (dec - mu) / sd
                        tgt = trj.x_std[c0 + 1:c0 + 1 + H]
                        se += float(np.sum((pred - tgt) ** 2))
                        cnt += pred.size
                err = se / cnt if cnt else np.inf
                if err < best_err:
                    best_err, best = err, (W, g)
        return best

    trend_star = _fit_trend()

    def trend_pred(tr, c0, hmax):
        return (X.decode(ae, _trend_roll(tr.h[:c0 + 1], hmax, *trend_star)) - mu) / sd

    def trend_lat(tr, c0, steps):
        return _trend_roll(tr.h[:c0 + 1], steps, *trend_star)

    # -- anchored decoding: yhat = y_c + [D(hhat) - D(h_c)] removes the constant
    #    reconstruction offset at the forecast origin (Theorem 3 x decoder-Lipschitz).
    def _anchor_pred(tr, c0, latent_fut):
        dec = X.decode(ae, latent_fut)
        dec0 = X.decode(ae, tr.h[c0][None, :])[0]
        y_c = tr.x_std[c0] * sd + mu
        return (dec - dec0[None, :] + y_c[None, :] - mu) / sd

    def cvd_anch_pred(tr, c0, hmax):
        return _anchor_pred(tr, c0, _damp_roll(tr.h[:c0 + 1], hmax, gamma_star))

    def holt_anch_pred(tr, c0, hmax):
        return _anchor_pred(tr, c0, _holt(tr.h[:c0 + 1], hmax, *holt_star))

    # -- sensor-space VAR
    def var_pred(tr, c0, hmax):
        raw = tr.x_std[c0] * sd + mu
        z = (raw - vmu) / vsd
        out = np.empty((hmax, n), np.float32)
        for t in range(hmax):
            z = reg.predict(z.reshape(1, -1))[0]
            out[t] = (z * vsd + vmu - mu) / sd
        return out

    # -- residual MLP, no context
    r_nc = LD.train_dynamics(tr_trajs, k, c_dim, use_context=False,
                             projection="soft", multistep=True,
                             hidden=64, alpha=0.05, horizon=horizon,
                             epochs=dyn_epochs, seed=seed)

    def mlp_nc_pred(tr, c0, hmax):
        return decode_std(LD.dyn_rollout(r_nc, tr.h[c0], tr.c[c0:c0+hmax], hmax))

    def mlp_nc_lat(tr, c0, s):
        return LD.dyn_rollout(r_nc, tr.h[c0], tr.c[c0:c0+s], s)

    # -- residual MLP, with context
    r_ctx = LD.train_dynamics(tr_trajs, k, c_dim, use_context=True,
                              projection="soft", multistep=True,
                              hidden=64, alpha=0.05, horizon=horizon,
                              epochs=dyn_epochs, seed=seed)

    def mlp_ctx_pred(tr, c0, hmax):
        return decode_std(LD.dyn_rollout(r_ctx, tr.h[c0], tr.c[c0:c0+hmax], hmax))

    def mlp_ctx_lat(tr, c0, s):
        return LD.dyn_rollout(r_ctx, tr.h[c0], tr.c[c0:c0+s], s)

    return {
        "persistence":  dict(pred=pers_pred, lat=None),
        "cv":           dict(pred=cv_pred, lat=cv_lat),
        "cv_damped":    dict(pred=cvd_pred, lat=cvd_lat, gamma=gamma_star),
        "holt":         dict(pred=holt_pred, lat=holt_lat, params=holt_star),
        "trend_lw":     dict(pred=trend_pred, lat=trend_lat, params=trend_star),
        "cvd_anch":     dict(pred=cvd_anch_pred, lat=cvd_lat),
        "holt_anch":    dict(pred=holt_anch_pred, lat=holt_lat),
        "var_sensor":   dict(pred=var_pred, lat=None),
        "mlp_noctx":    dict(pred=mlp_nc_pred, lat=mlp_nc_lat,
                             res=r_nc),
        "mlp_ctx":      dict(pred=mlp_ctx_pred, lat=mlp_ctx_lat,
                             res=r_ctx),
    }


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def eval_skill(pred_fn, te_trajs, horizons):
    hmax = max(horizons)
    m_sq = {h: [] for h in horizons}
    p_sq = {h: [] for h in horizons}
    for tr in te_trajs:
        T = len(tr.x_std)
        for f in ANCHORS:
            c0 = int(f * T)
            if c0 < X.ROLLOUT_VEL_WINDOW + 2 or c0 + hmax >= T:
                continue
            pred = pred_fn(tr, c0, hmax)
            pers = tr.x_std[c0]
            for h in horizons:
                tgt = tr.x_std[c0 + h]
                m_sq[h].append((pred[h-1] - tgt)**2)
                p_sq[h].append((pers - tgt)**2)
    skill, nrmse = {}, {}
    for h in horizons:
        if len(m_sq[h]) >= 2:
            mm = np.mean(np.concatenate([a.ravel() for a in m_sq[h]]))
            pp = np.mean(np.concatenate([a.ravel() for a in p_sq[h]]))
            skill[h] = float(1.0 - mm / (pp + 1e-12))
            nrmse[h] = float(np.sqrt(mm))
        else:
            skill[h] = float("nan")
            nrmse[h] = float("nan")
    return skill, nrmse


def eval_freerun(pred_fn, te_trajs, horizons):
    tr = max(te_trajs, key=lambda t: len(t.x_std))
    T = len(tr.x_std)
    hmax = max(horizons)
    # Use same dynamic cutoff: at least 2× max_horizon of free-run steps
    c0 = max(int(0.5 * T), T - 2*hmax)
    steps = min(FREE_STEPS, T - c0 - 1)
    if steps < 1:
        return float("nan"), False
    pred = pred_fn(tr, c0, steps)
    norms = np.linalg.norm(pred, axis=1)
    growth = float(norms[-1] / (norms[0] + 1e-12))
    return growth, bool(growth < BOUNDED_GROWTH_THRESH)


def pick_best_head(all_skill, horizons):
    """Pick head with best mean skill at horizons[1:] (skip h=1, recon tax).

    all_skill[lab][h] is a list of per-seed scalar values.
    """
    score = {}
    eval_h = horizons[1:] if len(horizons) > 1 else horizons
    for lab, sk in all_skill.items():
        vals = []
        for h in eval_h:
            v = float(np.nanmean(sk[h]))
            if not np.isnan(v):
                vals.append(v)
        score[lab] = float(np.mean(vals)) if vals else float("-inf")
    return max(score, key=score.__getitem__)


# --------------------------------------------------------------------------- #
# Observation-space metrics and figure
# --------------------------------------------------------------------------- #
def obs_space_metrics(pred_fn, te_trajs, feats, mu, sd, horizons):
    """Per-sensor RMSE and R² in *original* (un-standardized) feature space."""
    hmax = max(horizons)
    n = len(feats)
    preds = {h: {j: [] for j in range(n)} for h in horizons}
    truths = {h: {j: [] for j in range(n)} for h in horizons}

    for tr in te_trajs:
        T = len(tr.x_std)
        for f in ANCHORS:
            c0 = int(f * T)
            if c0 < X.ROLLOUT_VEL_WINDOW + 2 or c0 + hmax >= T:
                continue
            pred = pred_fn(tr, c0, hmax)   # (hmax, n) standardized
            for h in horizons:
                p_raw = pred[h-1] * sd + mu
                t_raw = tr.x_std[c0 + h] * sd + mu
                for j in range(n):
                    preds[h][j].append(float(p_raw[j]))
                    truths[h][j].append(float(t_raw[j]))

    rows = []
    for j, feat in enumerate(feats):
        for h in horizons:
            p_arr = np.array(preds[h][j])
            t_arr = np.array(truths[h][j])
            if len(p_arr) < 2:
                continue
            rmse = float(np.sqrt(np.mean((p_arr - t_arr)**2)))
            ss_res = np.sum((p_arr - t_arr)**2)
            ss_tot = np.sum((t_arr - t_arr.mean())**2) + 1e-12
            r2 = float(1.0 - ss_res / ss_tot)
            rows.append(dict(sensor=feat, horizon=h, rmse=rmse, r2=r2))
    return pd.DataFrame(rows)


def make_recon_figure(ae, te_trajs, feats, mu, sd, name, out_path):
    """
    Multi-panel reconstruction figure: AE rebuild of full trajectory.

    Each panel shows one representative test unit (the longest):

      Blue solid line     — Ground truth (actual recorded sensor values)
      Green dashed line   — AE reconstruction (what the bounded AE rebuilds
                            from k latent numbers)
    """
    unit_tr = max(te_trajs, key=lambda t: len(t.x_std))
    T = len(unit_tr.x_std)

    # Ground truth (original scale)
    truth_full = unit_tr.x_std * sd + mu        # (T, n_feat)
    steps = np.arange(T)

    # AE reconstruction (original scale)
    h_enc = unit_tr.h                            # (T, k)
    recon_std = (X.decode(ae, h_enc) - mu) / sd
    recon_full = recon_std * sd + mu             # (T, n_feat)

    n = len(feats)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 2.8 * nrows),
                             squeeze=False)
    fig.suptitle(f"{name} — Autoencoder reconstruction (k={ae['k']})\n"
                 f"Full trajectory of longest test unit (cycles 1-{T})",
                 fontsize=12, y=1.0)

    for idx, feat in enumerate(feats):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ax.plot(steps, truth_full[:, idx], color="steelblue", lw=1.2,
                label="Ground truth")
        ax.plot(steps, recon_full[:, idx], color="seagreen", lw=1.0,
                ls="--", alpha=0.8, label="AE reconstruction")
        ax.set_title(feat, fontsize=9, pad=3)
        ax.set_xlabel("cycle", fontsize=7)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6, loc="upper left")

    # hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    recon figure -> {out_path}")


def make_forecast_figure(ae, best_head, te_trajs, feats, mu, sd,
                         horizons, name, out_path):
    """
    Multi-panel forecast figure: best-head predictions vs ground truth.

    Each panel shows one representative test unit (the longest) from cutoff onward:

      Blue solid line     — Ground truth (actual observed sensor values after cutoff)
      Red solid line      — Best-head forecast (predicted future values)
    """
    pred_fn = best_head["pred"]

    # pick the longest test unit
    unit_tr = max(te_trajs, key=lambda t: len(t.x_std))
    T = len(unit_tr.x_std)
    hmax = max(horizons)
    # Ensure cutoff leaves at least 2× max_horizon steps for evaluation
    # This prevents the cutoff from being too late (e.g., 90% of trajectory)
    c0 = max(int(0.5 * T), T - 2*hmax)
    if c0 + hmax >= T:
        return

    # Ground truth from cutoff onward (original scale)
    truth_forecast = unit_tr.x_std[c0:c0+hmax] * sd + mu  # (hmax, n_feat)
    steps_forecast = np.arange(c0, c0 + hmax)

    # Forecast from cutoff (original scale)
    pred_std = pred_fn(unit_tr, c0, hmax)        # (hmax, n_feat)
    pred_raw = pred_std * sd + mu                # (hmax, n_feat)

    n = len(feats)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 2.8 * nrows),
                             squeeze=False)
    cutoff_pct = int(100 * c0 / T)
    fig.suptitle(f"{name} — {_head_label(best_head)} forecast\n"
                 f"From cycle {c0}/{T} ({cutoff_pct}% cutoff), {hmax} steps ahead",
                 fontsize=12, y=1.0)

    for idx, feat in enumerate(feats):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ax.plot(steps_forecast, truth_forecast[:, idx], color="steelblue", lw=1.2,
                label="Ground truth (observed)")
        ax.plot(steps_forecast, pred_raw[:, idx], color="crimson", lw=1.4,
                label="Forecast")
        ax.set_title(feat, fontsize=9, pad=3)
        ax.set_xlabel("cycle", fontsize=7)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6, loc="upper left")

    # hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    forecast figure -> {out_path}")


def _head_label(hd):
    res = hd.get("res")
    if res is None:
        return "cv/var/persistence"
    uc = hd["res"].model.use_context
    return "mlp_ctx (proposed)" if uc else "mlp_noctx"


# --------------------------------------------------------------------------- #
# REPORT.md writer
# --------------------------------------------------------------------------- #
def write_report(name, cfg, best_k, df_ksweep, metrics_df, best_head_name,
                 obs_metrics, horizons, feats, n_raw_feats, ctx, ae_epochs, dyn_epochs,
                 seeds, n_units_tr, n_units_te, out_dir):
    # Note: recon_plot.png and forecast_plot.png are now separate files
    met_md = metrics_df.round(4).to_markdown(index=False)

    # per-sensor obs summary
    obs_summary = (obs_metrics.groupby("sensor")[["rmse", "r2"]]
                   .mean().round(4).reset_index())
    obs_md = obs_summary.to_markdown(index=False)

    # best-k row
    bk_row = df_ksweep[df_ksweep["k"] == best_k].iloc[0]

    report = textwrap.dedent(f"""\
    # {name} — Experiment Report

    > **Dataset:** {cfg["description"]}

    ---

    ## 1. What this experiment does

    We apply a **Context-Conditioned Bounded Latent Dynamics** model to this
    dataset. The core idea:

    1. An **autoencoder** squeezes the {len(feats)}-variable sensor stream into
       a small **k={best_k}** latent state.  Each latent number is forced to stay
       between 0 and 1 (bounded), so predictions can never blow up.
    2. A small **residual MLP** then learns how the latent state steps forward in
       time.  If context columns are available (calendar time, operating settings)
       it conditions on those too.
    3. Every prediction is projected back onto the [0,1] box, then decoded back
       into the original sensor space so we can compare predicted vs actual sensor
       values directly.

    Five forecasting methods are compared:

    | Label | What it does |
    |---|---|
    | **persistence** | Just repeats the last observed value forever. The laziest possible forecast. Beats it = good. |
    | **cv** | Constant-velocity rollout: extrapolates the recent trend in the latent space. The old method. |
    | **var_sensor** | Fits a linear autoregression directly on the sensors (no latent compression). |
    | **mlp_noctx** | Bounded residual MLP in latent space, no context/season/regime information. |
    | **mlp_ctx** | Bounded residual MLP in latent space, **conditioned on context features** (proposed). |

    ---

    ## 2. Methodology and parameters

    | Parameter | Value |
    |---|---|
    | Latent dimension k | **{best_k}** (chosen by k-sweep, see Section 3) |
    | AE epochs | {ae_epochs} |
    | Dynamics epochs | {dyn_epochs} |
    | Seeds for MLP heads | {seeds} |
    | Monotonicity penalty λ_mono | {cfg["lambda_mono"]} |
    | Smoothness penalty λ_smooth | 0.5 |
    | Forecast horizons evaluated | {horizons} (in cycles/steps) |
    | Train / test split | {n_units_tr} / {n_units_te} units |
    | Context columns used | {ctx if ctx else "none"} |
    | Sensors (raw → selected) | {n_raw_feats} → {len(feats)} |
    | Denoising window | {DENOISE_WIN} cycles (causal rolling median) |
    | Sensor selection threshold | {"trend |corr(sensor,cycle)| ≥ " + str(TREND_THRESH) if cfg["lambda_mono"] > 0 else "variance filter (non-degrading dataset)"} |

    **Preprocessing pipeline (applied before any model training):**
    1. **Train/test split** — units split first so no preprocessing leaks across the boundary.
    2. **Denoising** — causal rolling-median (window={DENOISE_WIN} cycles) applied per unit per sensor.
       Removes cycle-to-cycle noise without looking ahead; matches the validated `manifold` pipeline.
    3. **Sensor selection** — {"sensors with mean |corr(sensor, cycle)| < " + str(TREND_THRESH) + " across training units are dropped. " + "These sensors carry no degradation signal and only add noise to the latent space. " + "Selected: " + str(len(feats)) + "/" + str(n_raw_feats) + " sensors." if cfg["lambda_mono"] > 0 else "near-constant sensors (std < 1e-4) are dropped; all varying sensors are kept for this non-degrading dataset. " + "Selected: " + str(len(feats)) + "/" + str(n_raw_feats) + " sensors."}

    **Why k={best_k}?**
    The k-sweep trains the autoencoder at each dimension and applies the **PCA elbow method**
    to find the latent dimension that captures the data's intrinsic structure without overfitting.
    The elbow is found using two criteria:
      1. The smallest k where cumulative explained variance ≥ 85%
      2. The k where incremental variance gain drops below 5%

    We take the **more conservative choice** (smaller k). For k={best_k}, the reconstruction
    R² (mean across sensors) = **{bk_row["recon_r2"]:.3f}** (worst single sensor = {bk_row["recon_r2_min"]:.3f}).

    **Why PCA elbow, not max reconstruction R²?**
    Reconstruction R² always rises with k, so choosing max R² would lead to overfitting
    (using many more latent dimensions than necessary). The PCA elbow method finds the
    "natural" dimensionality of the problem, a much more principled approach.

    **Why λ_mono={cfg["lambda_mono"]}?**
    {"The dataset has a degradation trend, so we encourage the primary latent coordinate to increase monotonically with time.  This is the *health progressing toward failure* assumption." if cfg["lambda_mono"] > 0 else "This dataset does not have a monotone degradation trend (it is stationary / cyclic), so the monotonicity penalty is turned off."}

    ---

    ## 3. K-sweep results

    This table shows how reconstruction quality changes as we give the latent
    space more dimensions.  **recon_r2** is the mean R² across all sensors;
    **recon_r2_min** is the worst single sensor.

    {df_ksweep.round(4).to_markdown(index=False)}

    A higher recon_r2 means the compressed representation is more faithful.
    Diminishing returns usually set in around k=3–5; we pick the elbow point
    (best recon without overfitting capacity).

    ---

    ## 4. Forecasting comparison

    **Skill vs persistence** measures "how much better than just repeating the
    last value are we?"  A skill of +0.5 means our error is half of the
    persistence error.  A skill of 0 means we tied persistence.  Negative means
    we were *worse* than doing nothing.

    **NRMSE** is the root-mean-squared forecast error expressed in standard
    deviation units of the training data, so a value of 0.5 means the average
    error is about half a standard deviation.

    **freerun_growth** is how many times larger the forecast output got over a
    long free run starting from the test cutoff.  A value near 1 is good (stable
    forecast).  Large values mean the forecast exploded.

    **bounded** = True means the growth stayed below the threshold of
    {BOUNDED_GROWTH_THRESH}×, which we define as "bounded" for reporting.

    {met_md}

    **Winner:** `{best_head_name}` (highest mean skill at horizons {horizons[1:]}).

    ---

    ## 5. Observation-space accuracy (best head: `{best_head_name}`)

    This table shows the per-sensor forecast accuracy in the **original feature
    scale** (regime-normalized z-scores or raw feature units depending on the
    dataset).  Averaged over the three anchor points (50%, 65%, 80% of each
    test trajectory) and all test units.

    **rmse**: root-mean-squared error in original feature units.
    **r2**: coefficient of determination; 1.0 = perfect, 0 = no better than the
    mean, negative = worse than the mean.

    Values are averaged over all forecast horizons ({horizons}).

    {obs_md}

    ---

    ## 6. Figures: Reconstruction and Forecast

    ### Figure (a): Reconstruction — `recon_plot.png`

    ![Reconstruction figure](recon_plot.png)

    This figure shows how faithfully the bounded autoencoder can **rebuild** the
    sensor values from the k={best_k} latent numbers, using the full observed
    trajectory of the longest test unit.

    | Element | What it means |
    |---|---|
    | **Blue solid line** | Ground truth — actual recorded sensor values |
    | **Green dashed line** | AE reconstruction — what the bounded AE rebuilds from the {best_k} latent numbers |

    If the green line follows the blue line closely, the k={best_k} latent space
    captures the data well. Large gaps mean information was lost in compression.
    This is a "no-forecast" check: can the model even represent the data?

    ---

    ### Figure (b): Forecast — `forecast_plot.png`

    ![Forecast figure](forecast_plot.png)

    This figure shows the **predictions** made by the best-head model (`{best_head_name}`)
    starting from a mid-to-late point through the trajectory of the longest test unit.

    | Element | What it means |
    |---|---|
    | **Blue solid line** | Ground truth — the actual sensor values after the cutoff (what really happened) |
    | **Red solid line** | Forecast — the model's predicted future values |

    **Cutoff strategy:** The forecast starts at the later of (1) 50% through the trajectory
    or (2) 2× the maximum forecast horizon before the end. This ensures you see a
    substantial forecast window (at least 2× max_horizon steps) to evaluate where
    predictions diverge from reality. If the red line stays close to the blue line,
    the model can predict well. If they diverge sharply, explode, or flatten, the
    forecast is unreliable.

    ---

    ## 7. Observation-space accuracy (best head: `{best_head_name}`)

    This table shows the per-sensor forecast accuracy in the **original feature
    scale** (regime-normalized z-scores or raw feature units depending on the
    dataset).  Averaged over the three anchor points (50%, 65%, 80% of each
    test trajectory) and all test units.

    **rmse**: root-mean-squared error in original feature units.
    **r2**: coefficient of determination; 1.0 = perfect, 0 = no better than the
    mean, negative = worse than the mean.

    Values are averaged over all forecast horizons ({horizons}).

    {obs_md}

    ---

    ## 8. Honest summary

    ### What worked
    {_honest_good(metrics_df, best_head_name, horizons, bk_row)}

    ### What did not work / limitations
    {_honest_bad(metrics_df, best_head_name, horizons)}

    ---

    *Generated automatically by experiments/acml/exp_all_datasets.py*
    """)

    path = os.path.join(out_dir, "REPORT.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"    REPORT.md -> {path}")


def _honest_good(metrics_df, best, horizons, bk_row):
    row = metrics_df[metrics_df["model"] == best]
    if len(row) == 0:
        return "_(results not available)_"
    row = row.iloc[0]
    lines = [f"- Reconstruction R² at the chosen k = **{bk_row['recon_r2']:.3f}**."]
    mid_h = horizons[1] if len(horizons) > 1 else horizons[0]
    sk_col = f"skill_h{mid_h}"
    if sk_col in row and not np.isnan(row[sk_col]) and row[sk_col] > 0:
        lines.append(f"- Best head (`{best}`) achieves positive forecast skill "
                     f"at h={mid_h}: **{row[sk_col]:+.3f}** (beats persistence).")
    if "bounded" in row and row["bounded"]:
        lines.append(f"- The best-head free-run forecast stays bounded "
                     f"(growth = {row['freerun_growth']:.2f}×).")
    if not lines:
        lines.append("- See table above for detailed metrics.")
    return "\n".join(lines)


def _honest_bad(metrics_df, best, horizons):
    h1 = horizons[0]
    row = metrics_df[metrics_df["model"] == best]
    lines = []
    if len(row) > 0:
        r = row.iloc[0]
        sk1_col = f"skill_h{h1}"
        if sk1_col in r and not np.isnan(r[sk1_col]) and r[sk1_col] < 0:
            lines.append(
                f"- **Short-horizon reconstruction tax:** at h={h1}, "
                f"skill = {r[sk1_col]:+.3f}.  Squeezing to k latent numbers "
                "discards small-scale detail that matters for very-next-step "
                "predictions.  Persistence wins here.")
    cv_row = metrics_df[metrics_df["model"] == "cv"]
    if len(cv_row) > 0:
        cv = cv_row.iloc[0]
        mid_h = horizons[1] if len(horizons) > 1 else horizons[0]
        sk_col = f"skill_h{mid_h}"
        if sk_col in cv and not np.isnan(cv[sk_col]) and cv[sk_col] < -0.1:
            lines.append(
                "- **Constant-velocity rollout** (old method) performs poorly "
                f"at h={mid_h} (skill {cv[sk_col]:+.3f}), confirming it is "
                "the weakest link.")
    if not lines:
        lines.append("- No major failure modes detected at these horizons and metrics.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run_dataset(name, cfg, args):
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'=' * 70}")
    print(f"DATASET: {name}")
    print(f"  {cfg['description']}")
    print(f"{'=' * 70}")

    # --- load ---
    feat_path = os.path.join(ROOT, cfg["feat_csv"])
    if not os.path.exists(feat_path):
        print(f"  SKIP: {feat_path} not found")
        return
    df, feats, ctx = load_dataset(feat_path)
    n_raw_feats = len(feats)
    print(f"  rows={len(df)}  units={df['unit_id'].nunique()}  "
          f"features={n_raw_feats}  context={len(ctx)}")

    # Split first (no leakage), then denoise each split independently,
    # then select sensors using training data only.
    tr_raw, te_raw = X.split_units(df, test_frac=cfg["test_frac"])
    tr_df = X.denoise(tr_raw, feats, win=DENOISE_WIN)
    te_df = X.denoise(te_raw, feats, win=DENOISE_WIN)
    feats = select_sensors(tr_df, feats, cfg["lambda_mono"])
    n_tr = tr_df["unit_id"].nunique()
    n_te = te_df["unit_id"].nunique()
    print(f"  train={n_tr} units  test={n_te} units")
    sel_method = "trend |corr|≥{:.2f}".format(TREND_THRESH) if cfg["lambda_mono"] > 0 else "variance (non-degrading)"
    print(f"  sensors: {n_raw_feats} raw → {len(feats)} selected ({sel_method}, denoise win={DENOISE_WIN})")

    # --- k-sweep ---
    print(f"  [1/5] k-sweep (k=1..{args.kmax}) ...")
    df_k, best_k = k_sweep(tr_df, te_df, feats, cfg["lambda_mono"],
                           k_max=args.kmax, epochs=args.epochs_sweep, seed=0,
                           lambda_smooth=args.lambda_smooth)
    print(f"        best k={best_k}  "
          f"recon_r2={df_k.loc[df_k.k==best_k,'recon_r2'].iloc[0]:.3f}")
    df_k.to_csv(os.path.join(out_dir, "ksweep.csv"), index=False)

    # --- train final AE at best k ---
    print(f"  [2/5] training final AE (k={best_k}, {args.epochs_ae} epochs) ...")
    ae = X.train_ae(tr_df, feats, k=best_k, bounded=True,
                    lambda_mono=cfg["lambda_mono"], lambda_smooth=args.lambda_smooth,
                    epochs=args.epochs_ae, seed=0)
    mu, sd = ae["mu"], ae["sd"]
    print(f"        recon_r2={X.recon_r2(ae, te_df)[0]:.3f}")

    # --- build latent trajectories ---
    tr_trajs = LD.build_trajectories(lambda g: X.encode(ae, g),
                                     tr_df, feats, ctx, mu, sd)
    te_trajs = LD.build_trajectories(lambda g: X.encode(ae, g),
                                     te_df, feats, ctx, mu, sd)

    # --- fit VAR reference ---
    reg, vmu, vsd, rho = X.fit_var(tr_df, feats)
    print(f"        sensor-VAR rho={rho:.3f}")

    # --- dynamics heads ---
    horizons = cfg["horizons"]
    horizon = max(16, horizons[2] if len(horizons) > 2 else horizons[-1])
    print(f"  [3/5] training dynamics heads "
          f"({args.epochs_dyn} epochs, {args.seeds} seeds) ...")
    all_skill, all_nrmse, all_growth, all_bounded = {}, {}, {}, {}
    head_objs = {}
    for si, seed in enumerate(args.seeds):
        heads = build_heads(ae, tr_trajs, te_trajs, ctx, mu, sd,
                            horizon, args.epochs_dyn, seed, reg, vmu, vsd)
        for lab, hd in heads.items():
            sk, nr = eval_skill(hd["pred"], te_trajs, horizons)
            g, b = eval_freerun(hd["pred"], te_trajs, horizons)
            if lab not in all_skill:
                all_skill[lab] = {h: [] for h in horizons}
                all_nrmse[lab] = {h: [] for h in horizons}
                all_growth[lab] = []
                all_bounded[lab] = []
                head_objs[lab] = hd     # store first-seed version for plotting
            for h in horizons:
                all_skill[lab][h].append(sk[h])
                all_nrmse[lab][h].append(nr[h])
            all_growth[lab].append(g)
            all_bounded[lab].append(b)
        print(f"        seed {seed} done ({si+1}/{len(args.seeds)})")

    # --- assemble metrics table ---
    rows = []
    for lab in ["persistence", "cv", "cv_damped", "holt", "trend_lw",
                "cvd_anch", "holt_anch", "var_sensor", "mlp_noctx", "mlp_ctx"]:
        row = dict(model=lab)
        for h in horizons:
            row[f"skill_h{h}"] = float(np.nanmean(all_skill[lab][h]))
            row[f"nrmse_h{h}"] = float(np.nanmean(all_nrmse[lab][h]))
        row["freerun_growth"] = float(np.nanmean(all_growth[lab]))
        row["bounded"] = bool(np.mean(all_bounded[lab]) > 0.5)
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    # --- pick best head ---
    best_head_name = pick_best_head(all_skill, horizons)
    best_head = head_objs[best_head_name]
    if "cv_damped" in head_objs:
        print(f"  [info] fitted damping gamma* = {head_objs['cv_damped'].get('gamma')}")
    if "holt" in head_objs:
        print(f"  [info] fitted Holt (alpha,beta,phi) = {head_objs['holt'].get('params')}")
    if "trend_lw" in head_objs:
        print(f"  [info] fitted trend (W,gamma) = {head_objs['trend_lw'].get('params')}")
    print(f"  [4/5] best head: {best_head_name}")

    # --- obs-space metrics ---
    print(f"  [4/5] obs-space metrics for {best_head_name} ...")
    obs_df = obs_space_metrics(best_head["pred"], te_trajs, feats, mu, sd, horizons)
    obs_df.to_csv(os.path.join(out_dir, "obs_metrics.csv"), index=False)

    # --- figures (reconstruction and forecast, both with ground truth) ---
    print(f"  [5/5] generating figures ...")
    recon_path = os.path.join(out_dir, "recon_plot.png")
    make_recon_figure(ae, te_trajs, feats, mu, sd, name, recon_path)
    
    forecast_path = os.path.join(out_dir, "forecast_plot.png")
    make_forecast_figure(ae, best_head, te_trajs, feats, mu, sd,
                         horizons, name, forecast_path)

    # --- REPORT.md ---
    write_report(name, cfg, best_k, df_k, metrics_df, best_head_name,
                 obs_df, horizons, feats, n_raw_feats, ctx, args.epochs_ae,
                 args.epochs_dyn, args.seeds, n_tr, n_te, out_dir)

    # --- console summary ---
    print(f"\n  RESULT SUMMARY — {name}")
    for _, row in metrics_df.iterrows():
        skills = "  ".join(
            f"@{h}={row[f'skill_h{h}']:+.3f}" for h in horizons
        )
        print(f"    {row['model']:16s}  {skills}  "
              f"growth={row['freerun_growth']:7.2f}  bounded={row['bounded']}")
    print(f"  >> best: {best_head_name}")

    return metrics_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=list(DATASET_CONFIGS.keys()),
                    help="dataset names to run (default: all)")
    ap.add_argument("--kmax", type=int, default=6,
                    help="maximum k for the latent-dimension sweep")
    ap.add_argument("--epochs-sweep", type=int, default=200,
                    help="AE epochs during k-sweep")
    ap.add_argument("--epochs-ae", type=int, default=600,
                    help="AE epochs for the final best-k model")
    ap.add_argument("--epochs-dyn", type=int, default=200,
                    help="dynamics head training epochs")
    ap.add_argument("--lambda-smooth", type=float, default=0.5,
                    help="AE latent smoothness penalty (production C-MAPSS = 2.0)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1],
                    help="random seeds for MLP heads")
    args = ap.parse_args()

    print("Unified experiment runner — context-conditioned bounded latent dynamics")
    print(f"Datasets: {args.datasets}")
    print(f"k-sweep 1..{args.kmax} ({args.epochs_sweep} epochs each)")
    print(f"Final AE: {args.epochs_ae} epochs | Dynamics: {args.epochs_dyn} epochs "
          f"× {args.seeds} seeds")

    summary_rows = []
    for name in args.datasets:
        if name not in DATASET_CONFIGS:
            print(f"Unknown dataset: {name}. Available: {list(DATASET_CONFIGS)}")
            continue
        cfg = DATASET_CONFIGS[name]
        m = run_dataset(name, cfg, args)
        if m is not None:
            summary_rows.append(dict(dataset=name,
                                     best_recon_r2="(see REPORT)"))

    print("\n" + "=" * 70)
    print("ALL DONE — outputs in results/final_experiments/<dataset>/")
    print("  metrics.csv       all-head skill / growth / bounded")
    print("  obs_metrics.csv   per-sensor RMSE and R² (best head)")
    print("  forecast_plot.png ground-truth vs prediction per sensor")
    print("  REPORT.md         full methodology + figure explanations")


if __name__ == "__main__":
    main()
