"""
Context-conditioned bounded latent dynamics — ablation driver.

Replaces the constant-velocity (CV) latent rollout with a learned residual
context-conditioned dynamics head and runs the full ablation the reviewer asks
for, all in the frozen bounded latent space of one autoencoder:

  1. cv                 constant-velocity latent rollout (old baseline)
  2. ar1                latent AR(1), spectral-radius constrained
  3. mlp_noctx          residual MLP dynamics, NO context           (bounded, soft proj)
  4. mlp_ctx            residual MLP dynamics, WITH context         (PROPOSED)
  5. mlp_ctx_noproj     with context, NO projection (unbounded)     -> tests projection
  6. mlp_ctx_onestep    with context, one-step loss only            -> tests multi-step loss
  7. sensor_mlp_ctx     matched-capacity MLP in SENSOR space        -> tests bounded space
  8. var_sensor         sensor-space VAR                            (reference)
  9. persistence        last-value hold                             (reference)

Everything is compared in standardized sensor space at horizons {1,8,24,48}:
skill vs persistence, NRMSE/horizon, closed-loop free-run growth + bounded flag,
and (for latent heads) latent max-norm and box-saturation fraction. MLP heads
are trained over several seeds; skill is reported as mean +/- std.

Inputs
------
  data/processed/air_quality_features_ctx.csv   (air_quality_prep.py --context)

Outputs
-------
  results/acml/tables/<name>_latent_dynamics.csv (+ .tex)
  results/acml/figures/<name>_latent_dynamics.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))

import exp_ims_bearing as X          # frozen-AE building blocks (train/encode/decode/rollout)
import latent_dynamics as LD         # learned dynamics heads

TAB = os.path.join(ROOT, "results", "acml", "tables")
FIG = os.path.join(ROOT, "results", "acml", "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

FREE_STEPS = X.FREE_STEPS
CUTOFF_FRAC = X.CUTOFF_FRAC
BOUNDED_GROWTH_THRESH = X.BOUNDED_GROWTH_THRESH
DEFAULT_FEAT = os.path.join(ROOT, "data", "processed",
                            "air_quality_features_ctx.csv")
ANCHORS = (0.5, 0.65, 0.8)
SAT_EPS = 1e-3


# --------------------------------------------------------------------------- #
# Loading (separate feature columns from ctx_ context columns)
# --------------------------------------------------------------------------- #
def load_with_context(path: str):
    df = pd.read_csv(path)
    ctx = [c for c in df.columns if c.startswith("ctx_")]
    feats = [c for c in df.columns if c not in ("unit_id", "cycle") and c not in ctx]
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    return df, feats, ctx


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def eval_skill(predict_std, trajs, horizons):
    """predict_std(tr, c0, hmax) -> (hmax, n_std). Returns (skill{h}, nrmse{h})."""
    hmax = max(horizons)
    m_sq = {h: [] for h in horizons}
    p_sq = {h: [] for h in horizons}
    for tr in trajs:
        T = len(tr.x_std)
        for f in ANCHORS:
            c0 = int(f * T)
            if c0 < X.ROLLOUT_VEL_WINDOW + 2 or c0 + hmax >= T:
                continue
            pred = predict_std(tr, c0, hmax)
            pers = tr.x_std[c0]
            for h in horizons:
                tgt = tr.x_std[c0 + h]
                m_sq[h].append((pred[h - 1] - tgt) ** 2)
                p_sq[h].append((pers - tgt) ** 2)
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


def longest(trajs):
    return max(trajs, key=lambda t: len(t.x_std))


def eval_freerun(predict_std, trajs, latent_fn=None):
    """Closed-loop growth on the longest unit; latent stats if latent_fn given."""
    tr = longest(trajs)
    T = len(tr.x_std)
    c0 = int(CUTOFF_FRAC * T)
    steps = min(FREE_STEPS, T - c0 - 1)
    pred = predict_std(tr, c0, steps)
    norms = np.linalg.norm(pred, axis=1)
    growth = float(norms[-1] / (norms[0] + 1e-12))
    bounded = bool(growth < BOUNDED_GROWTH_THRESH and np.isfinite(growth))
    lat_max, sat = float("nan"), float("nan")
    if latent_fn is not None:
        lat = latent_fn(tr, c0, steps)
        lat_max = float(np.max(np.linalg.norm(lat, axis=1)))
        near0 = np.abs(lat) < SAT_EPS
        near1 = np.abs(lat - 1.0) < SAT_EPS
        sat = float(np.mean(near0 | near1))
    return growth, bounded, lat_max, sat


# --------------------------------------------------------------------------- #
# Head factory (returns predict_std, latent_fn, meta) for a given seed
# --------------------------------------------------------------------------- #
def make_heads(ae, trajs, ctx, mu, sd, horizon, epochs, seed, reg, vmu, vsd, ar):
    n = len(mu)
    c_dim = len(ctx)

    def decode_std(latent):
        return (X.decode(ae, latent) - mu) / sd

    # --- CV latent rollout ---
    def cv_pred(tr, c0, hmax):
        fut_h, dec_raw = X.rollout(ae, tr.h[:c0 + 1], hmax, "full_box")
        return (dec_raw - mu) / sd

    def cv_lat(tr, c0, steps):
        fut_h, _ = X.rollout(ae, tr.h[:c0 + 1], steps, "full_box")
        return fut_h

    # --- AR(1) ---
    def ar_pred(tr, c0, hmax):
        return decode_std(LD.ar1_rollout(ar, tr.h[c0], hmax))

    def ar_lat(tr, c0, steps):
        return LD.ar1_rollout(ar, tr.h[c0], steps)

    heads = {}
    heads["cv"] = dict(predict=cv_pred, latent=cv_lat, res=None, kind="cv")
    heads["ar1"] = dict(predict=ar_pred, latent=ar_lat, res=None, kind="ar1")

    # --- learned MLP dynamics variants ---
    mlp_specs = {
        "mlp_noctx":        dict(use_context=False, projection="soft", multistep=True),
        "mlp_ctx":          dict(use_context=True,  projection="soft", multistep=True),
        "mlp_ctx_noproj":   dict(use_context=True,  projection="none", multistep=True),
        "mlp_ctx_onestep":  dict(use_context=True,  projection="soft", multistep=False),
    }
    for label, sp in mlp_specs.items():
        res = LD.train_dynamics(trajs, ae["k"], c_dim, hidden=64, alpha=0.05,
                                horizon=horizon, epochs=epochs, seed=seed, **sp)

        def make_pred(r):
            def _p(tr, c0, hmax):
                lat = LD.dyn_rollout(r, tr.h[c0], tr.c[c0:c0 + hmax], hmax)
                return decode_std(lat)
            def _l(tr, c0, steps):
                return LD.dyn_rollout(r, tr.h[c0], tr.c[c0:c0 + steps], steps)
            return _p, _l
        p, l = make_pred(res)
        heads[label] = dict(predict=p, latent=l, res=res, kind="mlp")

    # --- sensor-space matched MLP (with context) ---
    res_s = LD.train_sensor_dynamics(trajs, n, c_dim, use_context=True,
                                     hidden=64, alpha=0.25, horizon=horizon,
                                     epochs=epochs, seed=seed)

    def sensor_pred(tr, c0, hmax):
        return LD.sensor_rollout(res_s, tr.x_std[c0], tr.c[c0:c0 + hmax], hmax)
    heads["sensor_mlp_ctx"] = dict(predict=sensor_pred, latent=None,
                                   res=res_s, kind="sensor")

    # --- sensor VAR reference ---
    def var_pred(tr, c0, hmax):
        raw = tr.x_std[c0] * sd + mu
        z = (raw - vmu) / vsd
        out = np.empty((hmax, n), np.float32)
        for t in range(hmax):
            z = reg.predict(z.reshape(1, -1))[0]
            out[t] = (z * vsd + vmu - mu) / sd
        return out
    heads["var_sensor"] = dict(predict=var_pred, latent=None, res=None, kind="var")

    # --- persistence reference ---
    def pers_pred(tr, c0, hmax):
        return np.repeat(tr.x_std[c0][None, :], hmax, axis=0)
    heads["persistence"] = dict(predict=pers_pred, latent=None, res=None,
                                kind="pers")
    return heads


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default=DEFAULT_FEAT)
    ap.add_argument("--name", type=str, default="air_quality")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--epochs-ae", type=int, default=800)
    ap.add_argument("--epochs-dyn", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--horizons", type=int, nargs="*", default=[1, 8, 24, 48])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--test-frac", type=float, default=0.34)
    args = ap.parse_args()
    name = args.name
    horizons = args.horizons

    print(f"{name} — context-conditioned bounded latent dynamics ablation")
    print("=" * 70)
    df, feats, ctx = load_with_context(args.features)
    print(f"loaded {len(df)} rows, {df['unit_id'].nunique()} units, "
          f"{len(feats)} features, {len(ctx)} context cols: {ctx}")
    if not ctx:
        print("WARNING: no ctx_ columns found; context ablations will be trivial. "
              "Regenerate features with air_quality_prep.py --context.")

    dfd = X.denoise(df, feats, win=5)
    tr_df, te_df = X.split_units(dfd, test_frac=args.test_frac)
    print(f"train units: {tr_df['unit_id'].nunique()}  "
          f"test units: {te_df['unit_id'].nunique()}")

    # --- frozen bounded AE (no monotonicity: stationary/cyclic data) ---
    print(f"\n[1/4] training frozen bounded AE (k={args.k}) ...")
    ae = X.train_ae(tr_df, feats, k=args.k, bounded=True, lambda_mono=0.0,
                    lambda_smooth=0.5, epochs=args.epochs_ae)
    mu, sd = ae["mu"], ae["sd"]

    # --- latent trajectories on test units (evaluation) + train (fitting) ---
    tr_trajs = LD.build_trajectories(lambda g: X.encode(ae, g), tr_df, feats, ctx, mu, sd)
    te_trajs = LD.build_trajectories(lambda g: X.encode(ae, g), te_df, feats, ctx, mu, sd)

    # --- deterministic references (fit once) ---
    reg, vmu, vsd, rho = X.fit_var(tr_df, feats)
    ar = LD.fit_latent_ar1(tr_trajs)
    print(f"      sensor-VAR rho={rho:.3f}   latent-AR(1) rho={ar.rho:.3f}")

    # --- train + evaluate every head over seeds ---
    print(f"[2/4] training dynamics heads over seeds {args.seeds} ...")
    order = ["persistence", "cv", "ar1", "var_sensor", "mlp_noctx", "mlp_ctx",
             "mlp_ctx_noproj", "mlp_ctx_onestep", "sensor_mlp_ctx"]
    seed_skill = {lab: {h: [] for h in horizons} for lab in order}
    seed_nrmse = {lab: {h: [] for h in horizons} for lab in order}
    freerun = {}
    meta = {}
    det_done = set()

    for si, seed in enumerate(args.seeds):
        heads = make_heads(ae, tr_trajs, ctx, mu, sd, args.horizon,
                           args.epochs_dyn, seed, reg, vmu, vsd, ar)
        for lab in order:
            hd = heads[lab]
            deterministic = hd["kind"] in ("cv", "ar1", "var", "pers")
            if deterministic and lab in det_done:
                continue
            sk, nr = eval_skill(hd["predict"], te_trajs, horizons)
            for h in horizons:
                seed_skill[lab][h].append(sk[h])
                seed_nrmse[lab][h].append(nr[h])
            if lab not in freerun:
                g, b, lm, sat = eval_freerun(hd["predict"], te_trajs, hd["latent"])
                freerun[lab] = dict(growth=g, bounded=b, lat_max=lm, sat=sat)
                res = hd["res"]
                meta[lab] = dict(n_params=(res.n_params if res else 0),
                                 train_s=(res.train_time if res else 0.0))
            if deterministic:
                det_done.add(lab)
        print(f"    seed {seed} done ({si + 1}/{len(args.seeds)})")

    # --- assemble table ---
    rows = []
    for lab in order:
        row = dict(model=lab)
        for h in horizons:
            row[f"skill_h{h}"] = float(np.nanmean(seed_skill[lab][h]))
        hmid = horizons[min(2, len(horizons) - 1)]
        row[f"skill_h{hmid}_std"] = float(np.nanstd(seed_skill[lab][hmid]))
        row["nrmse_h1"] = float(np.nanmean(seed_nrmse[lab][horizons[0]]))
        row[f"nrmse_h{horizons[-1]}"] = float(np.nanmean(seed_nrmse[lab][horizons[-1]]))
        row["freerun_growth"] = freerun[lab]["growth"]
        row["bounded"] = freerun[lab]["bounded"]
        row["lat_max_norm"] = freerun[lab]["lat_max"]
        row["sat_frac"] = freerun[lab]["sat"]
        row["n_params"] = meta[lab]["n_params"]
        row["train_s"] = round(meta[lab]["train_s"], 2)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(TAB, f"{name}_latent_dynamics.csv"), index=False)
    with open(os.path.join(TAB, f"{name}_latent_dynamics.tex"), "w") as fh:
        fh.write(X.latex_table(
            summary.round(4),
            f"{name}: context-conditioned bounded latent dynamics vs baselines. "
            "Forecast skill vs persistence (mean over seeds), closed-loop "
            "free-run growth, boundedness and latent box-saturation.",
            f"tab:{name}_latentdyn"))

    print("[3/4] wrote table -> "
          f"{os.path.join(TAB, f'{name}_latent_dynamics.csv')}")

    # --- figure ---
    print("[4/4] rendering figure ...")
    make_figure(summary, seed_skill, horizons, name, ae, te_trajs, mu, sd,
                make_heads(ae, tr_trajs, ctx, mu, sd, args.horizon,
                           args.epochs_dyn, args.seeds[0], reg, vmu, vsd, ar))

    # --- console verdict ---
    print("\n" + "=" * 70)
    print(f"LATENT-DYNAMICS ABLATION — {name.upper()}")
    def g(lab, h):
        return summary.loc[summary.model == lab, f"skill_h{h}"].iloc[0]
    hmid = horizons[min(2, len(horizons) - 1)]
    for lab in order:
        r = summary[summary.model == lab].iloc[0]
        print(f"  {lab:16s} skill@{hmid}={g(lab, hmid):+.3f}  "
              f"growth={r.freerun_growth:8.2f}  bounded={bool(r.bounded)}  "
              f"sat={r.sat_frac if r.sat_frac == r.sat_frac else float('nan'):.3f}")
    print("=" * 70)
    print("Key contrasts:")
    print(f"  context helps?    mlp_ctx@{hmid}={g('mlp_ctx', hmid):+.3f} vs "
          f"mlp_noctx@{hmid}={g('mlp_noctx', hmid):+.3f}")
    print(f"  bounded space?    mlp_ctx@{hmid}={g('mlp_ctx', hmid):+.3f} vs "
          f"sensor_mlp@{hmid}={g('sensor_mlp_ctx', hmid):+.3f}")
    print(f"  multi-step helps? mlp_ctx growth={freerun['mlp_ctx']['growth']:.2f} vs "
          f"onestep growth={freerun['mlp_ctx_onestep']['growth']:.2f}")
    print(f"  projection?       mlp_ctx bounded={freerun['mlp_ctx']['bounded']} vs "
          f"noproj bounded={freerun['mlp_ctx_noproj']['bounded']} "
          f"(growth {freerun['mlp_ctx_noproj']['growth']:.2f})")
    print(f"  beats CV?         mlp_ctx@{hmid}={g('mlp_ctx', hmid):+.3f} vs "
          f"cv@{hmid}={g('cv', hmid):+.3f}")


def make_figure(summary, seed_skill, horizons, name, ae, te_trajs, mu, sd, heads):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    # (a) skill vs horizon for the key heads
    key = ["cv", "ar1", "mlp_noctx", "mlp_ctx", "sensor_mlp_ctx"]
    for lab in key:
        ys = [summary.loc[summary.model == lab, f"skill_h{h}"].iloc[0] for h in horizons]
        ax[0, 0].plot(horizons, ys, "o-", label=lab)
    ax[0, 0].axhline(0.0, color="gray", ls=":")
    ax[0, 0].set_title("(a) Forecast skill vs persistence")
    ax[0, 0].set_xlabel("horizon (steps)"); ax[0, 0].set_ylabel("skill")
    ax[0, 0].legend(fontsize=8)

    # (b) closed-loop free-run growth per head (log)
    labs = list(summary.model)
    growth = summary["freerun_growth"].to_numpy()
    colors = ["tab:green" if b else "tab:red" for b in summary["bounded"]]
    ax[0, 1].bar(range(len(labs)), np.maximum(growth, 1e-3), color=colors)
    ax[0, 1].axhline(BOUNDED_GROWTH_THRESH, color="gray", ls=":")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_xticks(range(len(labs)))
    ax[0, 1].set_xticklabels(labs, rotation=60, ha="right", fontsize=7)
    ax[0, 1].set_title("(b) Free-run growth (green=bounded)")

    # (c) skill@mid bar with seed error bars
    hmid = horizons[min(2, len(horizons) - 1)]
    means = [np.nanmean(seed_skill[lab][hmid]) for lab in labs]
    errs = [np.nanstd(seed_skill[lab][hmid]) for lab in labs]
    ax[1, 0].bar(range(len(labs)), means, yerr=errs, capsize=3)
    ax[1, 0].axhline(0.0, color="gray", ls=":")
    ax[1, 0].set_xticks(range(len(labs)))
    ax[1, 0].set_xticklabels(labs, rotation=60, ha="right", fontsize=7)
    ax[1, 0].set_title(f"(d) Skill @ h={hmid} (mean +/- seed std)")

    # (d) latent free-run: proposed vs no-projection latent norm
    tr = longest_traj(te_trajs)
    T = len(tr.x_std); c0 = int(CUTOFF_FRAC * T)
    steps = min(FREE_STEPS, T - c0 - 1)
    for lab, style in [("mlp_ctx", "-"), ("mlp_ctx_noproj", "--"), ("cv", ":")]:
        lf = heads[lab]["latent"]
        if lf is None:
            continue
        lat = lf(tr, c0, steps)
        ax[1, 1].plot(np.linalg.norm(lat, axis=1), style, label=lab)
    ax[1, 1].axhline(np.sqrt(ae["k"]), color="gray", ls=":", label="sqrt(k) box bound")
    ax[1, 1].set_title("(c) Latent free-run norm")
    ax[1, 1].set_xlabel("rollout step"); ax[1, 1].set_ylabel("||h||")
    ax[1, 1].legend(fontsize=8)

    fig.suptitle(f"{name}: context-conditioned bounded latent dynamics")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{name}_latent_dynamics.png"), dpi=130)
    plt.close(fig)


def longest_traj(trajs):
    return max(trajs, key=lambda t: len(t.x_std))


if __name__ == "__main__":
    main()
