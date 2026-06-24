"""
FD001 Thermodynamic Health-Manifold Experiment (v3).

Goal: stable R2 >= 0.9 reconstruction for all *informative* sensors, using a
physics-structured model that is robust under autoregressive rollout.

First-principles premise (see master log Section "First-Principles Framing"):
FD001 runs at a single operating point (sea level, M~0, TRA=100), so the only
quantity evolving over an engine's life is the HPC degradation. Physically there
is therefore essentially ONE latent health coordinate theta(t); every dynamic
sensor is a smooth, monotone-in-time function of theta plus measurement noise.
The achievable R2 per sensor is bounded by its signal-to-noise ratio.

Three stages:
  Stage 0  Reconstructability ceiling (leave-one-out HistGBM) + linear PCA
           intrinsic dimensionality. Defines the feasible 0.9+ target set.
  Stage 1  PINN-style physics-constrained health-manifold autoencoder solved
           with torch autograd. A low-dim latent h(t) is forced (by penalties)
           to be smooth and monotone non-decreasing in cycle -- the physical
           signature of irreversible degradation. Per-sensor train/test R2.
  Stage 2  Interpretable thermodynamic inter-sensor equations (conservation of
           mass/energy, spool power balance, choked-orifice flow) fit with and
           without the solved health latent h. Shows which physical law governs
           each sensor and confirms rollout-stable algebraic links.

References grounding the physics (fetched, see master log):
  - MIT 16.unified Thermodynamics & Propulsion, Brayton cycle work/efficiency
    (compressor temperature ratio, spool work balance: W_compressor = W_turbine
    + W_useful). node28.
  - Corrected-parameter similarity: N/sqrt(theta), W*sqrt(theta)/delta.
"""

import json
import os
from datetime import datetime
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

T_REF = 518.67
P_REF = 14.696
GAMMA = 1.4
EXP_ISEN = (GAMMA - 1.0) / GAMMA  # 0.2857

ALL_SENSORS = [f"s{i}" for i in range(1, 22)]
STATIONARY = ["s1", "s5", "s10", "s16", "s18", "s19"]
DYNAMIC = [s for s in ALL_SENSORS if s not in STATIONARY]

CONFIG = {
    "fd": 1,
    "test_size": 0.2,
    "target_r2": 0.90,
    "ceiling_informative": 0.50,   # sensor is "informative" if ceiling >= this
    "smooth_window": 15,
    "latent_dims": [1, 2, 3],
    "epochs": 4000,
    "lr": 5e-3,
    "lambda_mono": 5.0,
    "lambda_smooth": 2.0,
    "out_dir": "physics_hypothesis_outputs_v3",
    "plot_dir": "plotting/physics_v3",
    "master_log": "FD001_PHYSICS_MASTER_LOG.md",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def ensure_dirs() -> None:
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    os.makedirs(CONFIG["plot_dir"], exist_ok=True)


def load_fd001() -> pd.DataFrame:
    path = f"Data/train_FD00{CONFIG['fd']}.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing parquet file: {path}")
    df = pd.read_parquet(path).sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df["d"] = df["cycle"] / max_cycle           # life fraction in (0, 1]
    return df


def split_by_unit(df: pd.DataFrame):
    units = sorted(df["unit_id"].unique().tolist())
    tr_units, te_units = train_test_split(
        units, test_size=CONFIG["test_size"], random_state=SEED
    )
    tr = df[df["unit_id"].isin(tr_units)].sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    te = df[df["unit_id"].isin(te_units)].sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    return tr, te


def same_engine_mask(df: pd.DataFrame) -> np.ndarray:
    """Boolean for consecutive rows (t, t+1) belonging to the same engine."""
    uid = df["unit_id"].to_numpy()
    return uid[:-1] == uid[1:]


def denoise_per_engine(df: pd.DataFrame, cols: List[str], window: int) -> pd.DataFrame:
    """Per-engine centered rolling median -> the slow degradation trend.

    C-MAPSS adds zero-mean high-frequency measurement noise to each sensor. The
    physically meaningful, *predictable* quantity is the health-driven trend;
    the noise is irreducible. We model the trend and report the noise floor
    separately. Centered smoothing is used for offline trend extraction.
    """
    out = df.copy()
    for s in cols:
        out[s] = (
            out.groupby("unit_id")[s]
            .transform(lambda v: v.rolling(window, center=True, min_periods=1).median())
        )
    return out


def noise_floor(df_raw: pd.DataFrame, df_smooth: pd.DataFrame) -> Dict[str, float]:
    """Per-sensor fraction of variance that is high-frequency noise, and the
    implied R2 ceiling when predicting the *raw* sensor from any trend model:
    R2_ceiling_raw = 1 - var(noise)/var(raw)."""
    out = {}
    for s in DYNAMIC:
        resid = df_raw[s].to_numpy() - df_smooth[s].to_numpy()
        var_raw = np.var(df_raw[s].to_numpy())
        out[s] = float(1.0 - np.var(resid) / (var_raw + 1e-12))
    return out


# --------------------------------------------------------------------------- #
# Stage 0 - reconstructability ceiling + PCA dimensionality
# --------------------------------------------------------------------------- #
def stage0_ceilings(train_df, test_df) -> Dict[str, float]:
    ceilings = {}
    for s in DYNAMIC:
        if np.var(test_df[s].to_numpy()) < 1e-8:   # (near-)constant target
            ceilings[s] = 0.0
            continue
        feats = [c for c in DYNAMIC if c != s]
        model = HistGradientBoostingRegressor(
            max_depth=4, max_iter=300, learning_rate=0.05, random_state=SEED
        )
        model.fit(train_df[feats], train_df[s])
        ceilings[s] = float(r2_score(test_df[s], model.predict(test_df[feats])))
    return ceilings


def stage0_pca(train_df) -> List[float]:
    X = train_df[DYNAMIC].to_numpy()
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    evr = PCA().fit(X).explained_variance_ratio_
    return evr.tolist()


# --------------------------------------------------------------------------- #
# Stage 1 - PINN-style health-manifold autoencoder
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


def train_health_ae(train_df, test_df, k: int, weights: np.ndarray):
    """Fit a k-dim physics-constrained health manifold; return per-sensor R2
    (train/test) and the solved latent h on both splits."""
    mu = train_df[DYNAMIC].mean().to_numpy()
    sd = train_df[DYNAMIC].std().to_numpy() + 1e-12
    Xtr = ((train_df[DYNAMIC].to_numpy() - mu) / sd).astype(np.float32)
    Xte = ((test_df[DYNAMIC].to_numpy() - mu) / sd).astype(np.float32)

    Xtr_t = torch.tensor(Xtr)
    Xte_t = torch.tensor(Xte)
    w = torch.tensor((weights / weights.sum() * len(weights)).astype(np.float32))
    mask_tr = torch.tensor(same_engine_mask(train_df))

    model = HealthAE(len(DYNAMIC), k)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    for epoch in range(CONFIG["epochs"]):
        opt.zero_grad()
        recon, h = model(Xtr_t)
        rec_loss = (w * (recon - Xtr_t) ** 2).mean()

        h0 = h[:, 0]
        dh = h0[1:] - h0[:-1]
        # Monotone non-decreasing in cycle within an engine (irreversible wear).
        mono = torch.relu(-dh)[mask_tr].mean()
        # Smoothness: penalize curvature of the health trajectory.
        smooth = (dh[mask_tr] ** 2).mean()

        loss = rec_loss + CONFIG["lambda_mono"] * mono + CONFIG["lambda_smooth"] * smooth
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        recon_tr, h_tr = model(Xtr_t)
        recon_te, h_te = model(Xte_t)
    recon_tr = recon_tr.numpy() * sd + mu
    recon_te = recon_te.numpy() * sd + mu

    r2_tr, r2_te = {}, {}
    for j, s in enumerate(DYNAMIC):
        r2_tr[s] = float(r2_score(train_df[s], recon_tr[:, j]))
        r2_te[s] = float(r2_score(test_df[s], recon_te[:, j]))

    return dict(
        k=k, r2_train=r2_tr, r2_test=r2_te,
        h_train=h_tr.numpy(), h_test=h_te.numpy(),
    )


# --------------------------------------------------------------------------- #
# Stage 2 - interpretable thermodynamic inter-sensor equations
# --------------------------------------------------------------------------- #
def f_T30_euler(df):           # HPC energy: T30 = a + b*T24 + c*Nc^2
    return np.column_stack([df["s2"], df["s9"] ** 2])


def f_T30_polytropic(df):      # isentropic + efficiency
    isen = df["s2"] * (np.power(df["s7"] / df["s5"], EXP_ISEN) - 1.0)
    return np.column_stack([df["s2"], isen])


def f_T50_energy(df):          # turbine/combustor energy: T50 = a + b*T30 + c*phi
    return np.column_stack([df["s3"], df["s12"]])


def f_phi_fuel(df):            # thrust-hold fuel schedule
    return np.column_stack([df["s4"] - df["s2"], df["s3"]])


def f_P30_static(df):          # gas dynamics: P30 = a + b*Ps30
    return df["s11"].to_numpy().reshape(-1, 1)


def f_htBleed(df):             # enthalpy ~ cp*T30
    return df["s3"].to_numpy().reshape(-1, 1)


def f_W31_choked(df):          # choked orifice: W ~ Ps30/sqrt(T30)
    return (df["s11"] / np.sqrt(df["s3"])).to_numpy().reshape(-1, 1)


def f_W32_choked(df):
    return (df["s11"] / np.sqrt(df["s3"])).to_numpy().reshape(-1, 1)


def f_BPR_split(df):           # flow split: BPR = a + b*(Nf/Nc)
    return (df["s8"] / df["s9"]).to_numpy().reshape(-1, 1)


def f_NRf(df):                 # corrected fan speed identity
    return (df["s8"] / np.sqrt(df["s1"] / T_REF)).to_numpy().reshape(-1, 1)


def f_NRc(df):                 # corrected core speed identity
    return (df["s9"] / np.sqrt(df["s2"] / T_REF)).to_numpy().reshape(-1, 1)


STAGE2_EQS = [
    dict(name="E_T30_euler",   target="s3",  builder=f_T30_euler,
         law="HPC energy (Euler work)",       desc="T30 = a + b*T24 + c*Nc^2"),
    dict(name="E_T30_polytropic", target="s3", builder=f_T30_polytropic,
         law="isentropic + eta_HPC",          desc="T30 = a + b*T24 + c*[T24*((P30/P2)^0.2857-1)]"),
    dict(name="E_T50_energy",  target="s4",  builder=f_T50_energy,
         law="turbine/combustor energy",      desc="T50 = a + b*T30 + c*phi"),
    dict(name="E_phi_fuel",    target="s12", builder=f_phi_fuel,
         law="thrust-hold fuel schedule",     desc="phi = a + b*(T50-T24) + c*T30"),
    dict(name="E_P30_static",  target="s7",  builder=f_P30_static,
         law="gas dynamics total/static",     desc="P30 = a + b*Ps30"),
    dict(name="E_htBleed",     target="s17", builder=f_htBleed,
         law="bleed enthalpy ~ cp*T30",       desc="htBleed = a + b*T30"),
    dict(name="E_W31_choked",  target="s20", builder=f_W31_choked,
         law="choked orifice flow",           desc="W31 = a + b*(Ps30/sqrt(T30))"),
    dict(name="E_W32_choked",  target="s21", builder=f_W32_choked,
         law="choked orifice flow",           desc="W32 = a + b*(Ps30/sqrt(T30))"),
    dict(name="E_BPR_split",   target="s15", builder=f_BPR_split,
         law="bypass flow split",             desc="BPR = a + b*(Nf/Nc)"),
    dict(name="E_NRf_identity", target="s13", builder=f_NRf,
         law="corrected-speed similarity",    desc="NRf = Nf/sqrt(T2/T_ref)"),
    dict(name="E_NRc_identity", target="s14", builder=f_NRc,
         law="corrected-speed similarity",    desc="NRc = Nc/sqrt(T24/T_ref)"),
]


def stage2_equations(train_df, test_df, h_tr, h_te) -> List[Dict]:
    results = []
    for eq in STAGE2_EQS:
        ytr = train_df[eq["target"]].to_numpy()
        yte = test_df[eq["target"]].to_numpy()
        Xtr = eq["builder"](train_df)
        Xte = eq["builder"](test_df)

        m = LinearRegression().fit(Xtr, ytr)
        r2_phys = float(r2_score(yte, m.predict(Xte)))

        Xtr_h = np.column_stack([Xtr, h_tr, h_tr ** 2])
        Xte_h = np.column_stack([Xte, h_te, h_te ** 2])
        mh = LinearRegression().fit(Xtr_h, ytr)
        r2_phys_h = float(r2_score(yte, mh.predict(Xte_h)))

        results.append(dict(
            name=eq["name"], target=eq["target"], law=eq["law"], desc=eq["desc"],
            r2_physics=r2_phys, r2_physics_plus_h=r2_phys_h,
            delta=r2_phys_h - r2_phys,
        ))
    return results


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def plot_health(h_te_1d, d_te, ae_results, ceilings, informative) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    axes[0].scatter(d_te, h_te_1d, s=6, alpha=0.25)
    rho = np.corrcoef(h_te_1d, d_te)[0, 1]
    axes[0].set_xlabel("life fraction d")
    axes[0].set_ylabel("health latent h (k=1)")
    axes[0].set_title(f"Solved health vs life (corr={rho:.3f})")

    best = ae_results[max(ae_results,
                          key=lambda k: np.mean([ae_results[k]["r2_test"][s]
                                                 for s in informative]))]
    sensors = DYNAMIC
    r2v = [best["r2_test"][s] for s in sensors]
    cv = [ceilings[s] for s in sensors]
    x = np.arange(len(sensors))
    axes[1].bar(x - 0.2, cv, width=0.4, label="ceiling (GBM LOO)")
    axes[1].bar(x + 0.2, r2v, width=0.4, label=f"manifold k={best['k']}")
    axes[1].axhline(0.9, color="k", ls="--", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sensors, rotation=90, fontsize=7)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("test R2 (denoised trend)")
    axes[1].set_title("Per-sensor R2: ceiling vs health manifold")
    axes[1].legend(fontsize=8)

    for k in sorted(ae_results):
        mean_r2 = np.mean([ae_results[k]["r2_test"][s] for s in informative])
        axes[2].bar(k, mean_r2)
        axes[2].text(k, mean_r2 + 0.01, f"{mean_r2:.3f}", ha="center", fontsize=8)
    axes[2].axhline(0.9, color="k", ls="--", lw=0.8)
    axes[2].set_xlabel("latent dimension k")
    axes[2].set_ylabel("mean test R2 (informative)")
    axes[2].set_title("Intrinsic dimensionality")
    axes[2].set_xticks(sorted(ae_results))
    axes[2].set_ylim(0, 1.05)

    fig.tight_layout()
    path = os.path.join(CONFIG["plot_dir"], "health_manifold.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def monotonicity(df, h1d):
    tmp = df[["unit_id", "cycle"]].copy()
    tmp["h"] = h1d
    vals = []
    for _, g in tmp.groupby("unit_id"):
        g = g.sort_values("cycle")
        if len(g) > 5:
            vals.append(np.corrcoef(g["cycle"], g["h"])[0, 1])
    return float(np.nanmean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ensure_dirs()
    df = load_fd001()
    train_df, test_df = split_by_unit(df)
    win = CONFIG["smooth_window"]
    train_s = denoise_per_engine(train_df, DYNAMIC, win)
    test_s = denoise_per_engine(test_df, DYNAMIC, win)

    print("=" * 80)
    print("FD001 THERMODYNAMIC HEALTH-MANIFOLD EXPERIMENT (v3)")
    print("=" * 80)
    print(f"train engines={train_df['unit_id'].nunique()} "
          f"test engines={test_df['unit_id'].nunique()} rows={len(df)}")

    # ---- Stage 0 -------------------------------------------------------- #
    print("\n[Stage 0] Noise floor + reconstructability ceilings")
    nfloor = noise_floor(test_df, test_s)
    ceilings_raw = stage0_ceilings(train_df, test_df)
    ceilings = stage0_ceilings(train_s, test_s)        # ceilings on the trend
    evr = stage0_pca(train_s)                          # dimensionality of trend
    informative = [s for s in DYNAMIC if ceilings[s] >= CONFIG["ceiling_informative"]]
    print(f"  {'sensor':6s}{'raw_ceil':>9s}{'noise_R2cap':>12s}{'trend_ceil':>11s}  tag")
    for s in DYNAMIC:
        tag = "INFORMATIVE" if ceilings[s] >= CONFIG["ceiling_informative"] else "noise-limited"
        print(f"  {s:6s}{ceilings_raw[s]:9.4f}{nfloor[s]:12.4f}{ceilings[s]:11.4f}  {tag}")
    print(f"  PCA EVR of denoised trend (first 5): {[round(v,4) for v in evr[:5]]}")
    print(f"  cumulative: PC1={evr[0]:.3f}  PC1-2={sum(evr[:2]):.3f}  "
          f"PC1-3={sum(evr[:3]):.3f}")
    print(f"  informative sensors ({len(informative)}): {informative}")

    # decoder weights focus the manifold on reconstructable sensors
    weights = np.array([max(ceilings[s], 0.05) for s in DYNAMIC])

    # ---- Stage 1 -------------------------------------------------------- #
    print("\n[Stage 1] PINN-style health-manifold autoencoder on denoised trend")
    ae_results = {}
    for k in CONFIG["latent_dims"]:
        res = train_health_ae(train_s, test_s, k, weights)
        ae_results[k] = res
        mean_inf = np.mean([res["r2_test"][s] for s in informative])
        n_pass = sum(res["r2_test"][s] >= CONFIG["target_r2"] for s in informative)
        print(f"  k={k}: mean test R2 (informative)={mean_inf:.4f}  "
              f"informative sensors >=0.9: {n_pass}/{len(informative)}")

    # choose smallest k achieving all informative >= target (else best mean)
    chosen_k = None
    for k in CONFIG["latent_dims"]:
        if all(ae_results[k]["r2_test"][s] >= CONFIG["target_r2"] for s in informative):
            chosen_k = k
            break
    if chosen_k is None:
        chosen_k = max(CONFIG["latent_dims"],
                       key=lambda k: np.mean([ae_results[k]["r2_test"][s] for s in informative]))
    chosen = ae_results[chosen_k]
    print(f"  -> chosen latent dimension k={chosen_k}")

    # orient h (k=1 trajectory) to increase with life for reporting
    h_tr_1d = ae_results[1]["h_train"][:, 0]
    h_te_1d = ae_results[1]["h_test"][:, 0]
    if np.corrcoef(h_te_1d, test_df["d"].to_numpy())[0, 1] < 0:
        h_tr_1d, h_te_1d = 1 - h_tr_1d, 1 - h_te_1d
    corr_life = float(np.corrcoef(h_te_1d, test_df["d"].to_numpy())[0, 1])
    mono = monotonicity(test_df, h_te_1d)
    print(f"  health latent h: corr(life)={corr_life:.3f}  "
          f"per-engine monotonicity={mono:.3f}")

    # ---- Stage 2 -------------------------------------------------------- #
    print("\n[Stage 2] Interpretable thermodynamic inter-sensor equations (+h)")
    stage2 = stage2_equations(train_s, test_s,
                              h_tr_1d.reshape(-1, 1), h_te_1d.reshape(-1, 1))
    for r in stage2:
        print(f"  {r['name']:18s} {r['target']:4s} "
              f"R2_phys={r['r2_physics']:7.4f}  +h={r['r2_physics_plus_h']:7.4f}  "
              f"d={r['delta']:+.4f}  [{r['law']}]")

    # ---- Persist -------------------------------------------------------- #
    plot_path = plot_health(h_te_1d, test_df["d"].to_numpy(), ae_results,
                            ceilings, informative)

    rows = []
    for s in DYNAMIC:
        rows.append(dict(
            sensor=s, ceiling=ceilings[s], test_std=float(test_df[s].std()),
            informative=s in informative,
            manifold_r2_k1=ae_results[1]["r2_test"][s],
            manifold_r2_k2=ae_results[2]["r2_test"][s],
            manifold_r2_k3=ae_results[3]["r2_test"][s],
            chosen_k=chosen_k, chosen_r2=chosen["r2_test"][s],
        ))
    pd.DataFrame(rows).to_csv(
        os.path.join(CONFIG["out_dir"], "manifold_per_sensor.csv"), index=False)
    pd.DataFrame(stage2).to_csv(
        os.path.join(CONFIG["out_dir"], "stage2_equations.csv"), index=False)

    degr = test_df[["unit_id", "cycle", "d"]].copy()
    degr["health_latent"] = h_te_1d
    degr.to_csv(os.path.join(CONFIG["out_dir"], "health_latent_test.csv"), index=False)

    summary = dict(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        ceilings=ceilings,
        pca_evr=evr[:6],
        informative_sensors=informative,
        chosen_k=chosen_k,
        manifold_mean_r2_informative={
            k: float(np.mean([ae_results[k]["r2_test"][s] for s in informative]))
            for k in CONFIG["latent_dims"]},
        informative_pass_at_chosen_k=[
            s for s in informative if chosen["r2_test"][s] >= CONFIG["target_r2"]],
        informative_fail_at_chosen_k=[
            s for s in informative if chosen["r2_test"][s] < CONFIG["target_r2"]],
        health_corr_life=corr_life,
        health_monotonicity=mono,
        plot=plot_path,
    )
    with open(os.path.join(CONFIG["out_dir"], "summary_v3.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nSaved artifacts in {CONFIG['out_dir']}/ and plot {plot_path}")
    n_pass = len(summary["informative_pass_at_chosen_k"])
    print(f"RESULT: {n_pass}/{len(informative)} informative sensors reach "
          f"R2>=0.9 at k={chosen_k}")
    if summary["informative_fail_at_chosen_k"]:
        print(f"  still below 0.9: {summary['informative_fail_at_chosen_k']}")
    return summary


if __name__ == "__main__":
    main()
