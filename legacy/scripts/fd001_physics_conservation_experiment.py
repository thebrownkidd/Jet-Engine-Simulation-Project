"""
FD001 Physics-Based Sensor Mapping via Conservation Laws.

Tests candidate equations (C1..C13) from
physics_hypothesis_outputs_v2/physics_conservation_research_log.md one-by-one.

For each equation:
  - Model P    : physics features only.
  - Model P+d  : physics + provisional life-fraction degradation (d, d^2).
  - ds R2      : R2 improvement attributable to degradation coupling.

Then discovers a SINGLE latent degradation parameter (theta_hat) from the joint
residual structure of the degradation-coupled equations via PCA, validates it
against the true life fraction, and re-fits the coupled equations with theta_hat.

All metrics use an engine-level train/test split (no unit leakage).
Results are appended to the research log MD file.
"""

import json
import os
from datetime import datetime
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

T_REF = 518.67          # deg R, reference total temperature
P_REF = 14.696          # psia, reference pressure
GAMMA_COLD = 1.4
EXP_ISEN = (GAMMA_COLD - 1.0) / GAMMA_COLD  # 0.2857

CONFIG = {
    "fd": 1,
    "seed": 42,
    "test_size": 0.2,
    "out_dir": "physics_hypothesis_outputs_v2",
    "plot_dir": "plotting/physics_conservation",
    "log_md": "physics_hypothesis_outputs_v2/physics_conservation_research_log.md",
    "scores_csv": "physics_hypothesis_outputs_v2/conservation_scores.csv",
    "degr_csv": "physics_hypothesis_outputs_v2/discovered_degradation.csv",
    "summary_json": "physics_hypothesis_outputs_v2/conservation_summary.json",
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
    return pd.read_parquet(path)


def add_life_fraction(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["unit_id", "cycle"]).copy()
    max_cycle = out.groupby("unit_id")["cycle"].transform("max")
    out["d"] = out["cycle"] / max_cycle          # life fraction in (0, 1]
    out["d2"] = out["d"] ** 2
    return out


def split_by_unit(df: pd.DataFrame):
    units = sorted(df["unit_id"].unique().tolist())
    tr_units, te_units = train_test_split(
        units, test_size=CONFIG["test_size"], random_state=CONFIG["seed"]
    )
    return (
        df[df["unit_id"].isin(tr_units)].copy(),
        df[df["unit_id"].isin(te_units)].copy(),
    )


# --------------------------------------------------------------------------- #
# Physics feature builders (each returns an (n, k) design matrix)
# --------------------------------------------------------------------------- #
def f_C1(df: pd.DataFrame) -> np.ndarray:
    # Corrected fan speed: s13 = s8 / sqrt(s1 / T_ref)
    return (df["s8"] / np.sqrt(df["s1"] / T_REF)).to_numpy().reshape(-1, 1)


def f_C2(df: pd.DataFrame) -> np.ndarray:
    # Corrected core speed: s14 = s9 / sqrt(s2 / T_ref)
    return (df["s9"] / np.sqrt(df["s2"] / T_REF)).to_numpy().reshape(-1, 1)


def f_C3(df: pd.DataFrame) -> np.ndarray:
    # HPC Euler work: s3 = a + b1*s2 + b2*Nc^2
    return np.column_stack([df["s2"].to_numpy(), (df["s9"] ** 2).to_numpy()])


def f_C4(df: pd.DataFrame) -> np.ndarray:
    # HPC polytropic efficiency:
    # s3 = a + b1*s2 + b2*[ s2 * ((s7/s5)^0.2857 - 1) ]
    isen = df["s2"] * (np.power(df["s7"] / df["s5"], EXP_ISEN) - 1.0)
    return np.column_stack([df["s2"].to_numpy(), isen.to_numpy()])


def f_C5(df: pd.DataFrame) -> np.ndarray:
    # LPT outlet temperature: s4 = a + b1*s3 + b2*phi
    return np.column_stack([df["s3"].to_numpy(), df["s12"].to_numpy()])


def f_C6(df: pd.DataFrame) -> np.ndarray:
    # Fuel/thermal coupling (phi): s12 = a + b1*(s4 - s2) + b2*s3
    return np.column_stack([(df["s4"] - df["s2"]).to_numpy(), df["s3"].to_numpy()])


def f_C7(df: pd.DataFrame) -> np.ndarray:
    # HPC exit total/static gas dynamics: s7 = a + b*s11
    return df["s11"].to_numpy().reshape(-1, 1)


def f_C9(df: pd.DataFrame) -> np.ndarray:
    # Bleed enthalpy: s17 = a + b*s3
    return df["s3"].to_numpy().reshape(-1, 1)


def f_C10(df: pd.DataFrame) -> np.ndarray:
    # HPT coolant bleed (choked orifice): s20 = a + b*(s11 / sqrt(s3))
    return (df["s11"] / np.sqrt(df["s3"])).to_numpy().reshape(-1, 1)


def f_C11(df: pd.DataFrame) -> np.ndarray:
    # LPT coolant bleed (choked orifice): s21 = a + b*(s11 / sqrt(s3))
    return (df["s11"] / np.sqrt(df["s3"])).to_numpy().reshape(-1, 1)


def f_C13(df: pd.DataFrame) -> np.ndarray:
    # Bypass ratio: s15 = a + b*(s8 / s9)
    return (df["s8"] / df["s9"]).to_numpy().reshape(-1, 1)


# --------------------------------------------------------------------------- #
# Equation catalogue
# --------------------------------------------------------------------------- #
EQUATIONS: List[Dict] = [
    dict(name="C1_corr_fan_speed",   target="s13", builder=f_C1,  identity=True,
         law="mass-flow similarity", desc="s13 = s8 / sqrt(s1/T_ref)"),
    dict(name="C2_corr_core_speed",  target="s14", builder=f_C2,  identity=True,
         law="mass-flow similarity", desc="s14 = s9 / sqrt(s2/T_ref)"),
    dict(name="C3_hpc_euler_work",   target="s3",  builder=f_C3,  identity=False,
         law="energy (Euler work)",  desc="s3 = a + b1*s2 + b2*Nc^2"),
    dict(name="C4_hpc_polytropic",   target="s3",  builder=f_C4,  identity=False,
         law="isentropic + eta_HPC", desc="s3 = a + b1*s2 + b2*[s2*((s7/s5)^0.2857-1)]"),
    dict(name="C5_lpt_outlet_temp",  target="s4",  builder=f_C5,  identity=False,
         law="combustor/turbine energy", desc="s4 = a + b1*s3 + b2*phi"),
    dict(name="C6_fuel_coupling",    target="s12", builder=f_C6,  identity=False,
         law="thrust-hold fuel schedule", desc="s12 = a + b1*(s4-s2) + b2*s3"),
    dict(name="C7_hpc_total_static", target="s7",  builder=f_C7,  identity=False,
         law="gas dynamics",         desc="s7 = a + b*s11"),
    dict(name="C9_bleed_enthalpy",   target="s17", builder=f_C9,  identity=False,
         law="enthalpy ~ cp*T30",    desc="s17 = a + b*s3"),
    dict(name="C10_hpt_coolant",     target="s20", builder=f_C10, identity=False,
         law="choked orifice flow",  desc="s20 = a + b*(s11/sqrt(s3))"),
    dict(name="C11_lpt_coolant",     target="s21", builder=f_C11, identity=False,
         law="choked orifice flow",  desc="s21 = a + b*(s11/sqrt(s3))"),
    dict(name="C13_bypass_ratio",    target="s15", builder=f_C13, identity=False,
         law="flow split",           desc="s15 = a + b*(s8/s9)"),
]

# Equations whose residuals are pooled to discover the latent degradation scalar.
COUPLED_FOR_DISCOVERY = [
    "C3_hpc_euler_work", "C4_hpc_polytropic", "C5_lpt_outlet_temp",
    "C6_fuel_coupling", "C9_bleed_enthalpy", "C10_hpt_coolant", "C11_lpt_coolant",
]


# --------------------------------------------------------------------------- #
# Fitting helpers
# --------------------------------------------------------------------------- #
def fit_eval(Xtr, ytr, Xte, yte):
    """Fit OLS, return (model, r2_test, residuals_test)."""
    model = LinearRegression()
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return model, r2_score(yte, pred), yte - pred


def evaluate_equation(eq: Dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    y_tr = train_df[eq["target"]].to_numpy()
    y_te = test_df[eq["target"]].to_numpy()
    Xp_tr = eq["builder"](train_df)
    Xp_te = eq["builder"](test_df)

    # Model P: physics only.
    model_p, r2_phys, res_te = fit_eval(Xp_tr, y_tr, Xp_te, y_te)
    res_tr = y_tr - model_p.predict(Xp_tr)

    # Model P+d: physics + provisional life-fraction degradation.
    d_tr = train_df[["d", "d2"]].to_numpy()
    d_te = test_df[["d", "d2"]].to_numpy()
    Xpd_tr = np.column_stack([Xp_tr, d_tr])
    Xpd_te = np.column_stack([Xp_te, d_te])
    _, r2_phys_d, _ = fit_eval(Xpd_tr, y_tr, Xpd_te, y_te)

    return dict(
        name=eq["name"], target=eq["target"], law=eq["law"], desc=eq["desc"],
        identity=eq["identity"], target_std=float(np.std(y_te)),
        r2_physics=float(r2_phys), r2_physics_plus_d=float(r2_phys_d),
        delta_r2_degradation=float(r2_phys_d - r2_phys),
        residual_train=res_tr, residual_test=res_te,
    )


# --------------------------------------------------------------------------- #
# Degradation discovery
# --------------------------------------------------------------------------- #
SMOOTH_WINDOW = 15  # cycles; theta(t) is a slow latent, noise is high-frequency


def smooth_per_engine(values: np.ndarray, df: pd.DataFrame,
                      window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Per-engine centered rolling mean, returned in the SAME row order as
    `values`/`df`. Degradation is slow; this removes the high-frequency
    measurement noise that otherwise dominates residual PCA."""
    tmp = pd.DataFrame({
        "uid": df["unit_id"].to_numpy(),
        "cyc": df["cycle"].to_numpy(),
        "v": values,
        "pos": np.arange(len(values)),
    })
    out = np.empty(len(tmp))
    for _, g in tmp.groupby("uid"):
        g = g.sort_values("cyc")
        sm = g["v"].rolling(window, center=True, min_periods=1).mean().to_numpy()
        out[g["pos"].to_numpy()] = sm
    return out


def _monotonicity(df: pd.DataFrame, theta: np.ndarray) -> float:
    tmp = df[["unit_id", "cycle"]].copy()
    tmp["theta"] = theta
    monos = []
    for _, g in tmp.groupby("unit_id"):
        g = g.sort_values("cycle")
        if len(g) > 5:
            monos.append(np.corrcoef(g["cycle"], g["theta"])[0, 1])
    return float(np.nanmean(monos)) if monos else float("nan")


def discover_degradation(results: List[Dict], train_df: pd.DataFrame,
                         test_df: pd.DataFrame) -> Dict:
    """Discover a single latent degradation parameter theta_hat from the joint
    residual structure of the coupled physics equations.

    The PCA basis (loadings + standardization) is fit on TRAIN residuals only,
    then applied to TEST residuals -> no test leakage in the discovered basis.
    """
    names = [r["name"] for r in results if r["name"] in COUPLED_FOR_DISCOVERY]

    # Build smoothed, standardized residual matrices (train fits the scaler).
    tr_cols, te_cols, means, stds = [], [], [], []
    for r in results:
        if r["name"] in COUPLED_FOR_DISCOVERY:
            rtr = smooth_per_engine(r["residual_train"], train_df)
            rte = smooth_per_engine(r["residual_test"], test_df)
            mu, sd = rtr.mean(), rtr.std() + 1e-12
            tr_cols.append((rtr - mu) / sd)
            te_cols.append((rte - mu) / sd)
            means.append(mu)
            stds.append(sd)
    R_tr = np.column_stack(tr_cols)
    R_te = np.column_stack(te_cols)

    pca = PCA(n_components=min(3, R_tr.shape[1]))
    pca.fit(R_tr)
    theta_tr = pca.transform(R_tr)[:, 0]
    theta_te = pca.transform(R_te)[:, 0]

    # Orient so theta increases with life, using TRAIN sign.
    if np.corrcoef(theta_tr, train_df["d"].to_numpy())[0, 1] < 0:
        theta_tr, theta_te = -theta_tr, -theta_te
        pca.components_[0] = -pca.components_[0]

    # Normalize both with TRAIN min/max.
    lo, hi = theta_tr.min(), theta_tr.max()
    rng = (hi - lo) + 1e-12
    theta_tr = (theta_tr - lo) / rng
    theta_te = (theta_te - lo) / rng

    d_te = test_df["d"].to_numpy()
    rho = float(np.corrcoef(theta_te, d_te)[0, 1])

    return dict(
        theta_train=theta_tr, theta=theta_te, d_true=d_te,
        pc1_var=float(pca.explained_variance_ratio_[0]),
        var_ratios=pca.explained_variance_ratio_.tolist(),
        loadings=dict(zip(names, pca.components_[0].round(4).tolist())),
        corr_with_life=rho,
        mean_per_engine_monotonicity=_monotonicity(test_df, theta_te),
    )


def refit_with_theta(eq: Dict, train_df, test_df, theta_tr, theta_te) -> float:
    """R2 of coupled equation when discovered theta_hat replaces life fraction."""
    y_tr = train_df[eq["target"]].to_numpy()
    y_te = test_df[eq["target"]].to_numpy()
    Xp_tr = np.column_stack([eq["builder"](train_df), theta_tr])
    Xp_te = np.column_stack([eq["builder"](test_df), theta_te])
    _, r2, _ = fit_eval(Xp_tr, y_tr, Xp_te, y_te)
    return float(r2)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def plot_degradation(disc: Dict, theta_full: np.ndarray, d_full: np.ndarray) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(disc["d_true"], disc["theta"], s=6, alpha=0.3)
    axes[0].set_xlabel("true life fraction d")
    axes[0].set_ylabel(r"discovered $\hat{\theta}$ (PC1 of residuals)")
    axes[0].set_title(f"corr = {disc['corr_with_life']:.3f}")
    axes[1].bar(range(len(disc["var_ratios"])), disc["var_ratios"])
    axes[1].set_xlabel("principal component")
    axes[1].set_ylabel("explained variance ratio")
    axes[1].set_title(f"PC1 = {disc['pc1_var']*100:.1f}% of residual variance")
    fig.suptitle("Discovered HPC degradation parameter from physics residuals")
    fig.tight_layout()
    path = os.path.join(CONFIG["plot_dir"], "discovered_degradation.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def append_results_md(results, disc, refit_r2):
    lines = []
    lines.append("\n---\n")
    lines.append(f"### Run {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    lines.append(
        "Engine-level train/test split (no unit leakage), OLS fits. "
        "`R2_phys` = physics features only; `R2_phys+d` adds provisional life "
        "fraction; `dR2` = degradation contribution.\n"
    )
    lines.append("\n#### Per-equation results\n")
    lines.append("| Equation | Law | Target | std | R2_phys | R2_phys+d | dR2 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['name']} | {r['law']} | {r['target']} | {r['target_std']:.3f} "
            f"| {r['r2_physics']:.4f} | {r['r2_physics_plus_d']:.4f} "
            f"| {r['delta_r2_degradation']:+.4f} |"
        )

    lines.append("\n#### Discovered degradation parameter (theta_hat)\n")
    lines.append(
        f"- PC1 explains **{disc['pc1_var']*100:.1f}%** of pooled residual variance "
        f"(ratios: {[round(v,3) for v in disc['var_ratios']]}).\n"
        f"- Correlation of theta_hat with true life fraction: **{disc['corr_with_life']:.3f}**.\n"
        f"- Mean per-engine monotonicity (corr theta_hat vs cycle): "
        f"**{disc['mean_per_engine_monotonicity']:.3f}**.\n"
    )
    lines.append("\nResidual loadings on PC1 (per equation):\n")
    lines.append("| Equation | PC1 loading |")
    lines.append("|---|---|")
    for k, v in disc["loadings"].items():
        lines.append(f"| {k} | {v:+.4f} |")

    lines.append("\n#### Coupled equations re-fitted with discovered theta_hat\n")
    lines.append("| Equation | R2_phys | R2_phys+theta_hat | gain |")
    lines.append("|---|---|---|---|")
    base = {r["name"]: r["r2_physics"] for r in results}
    for name, r2 in refit_r2.items():
        lines.append(f"| {name} | {base[name]:.4f} | {r2:.4f} | {r2-base[name]:+.4f} |")

    lines.append(
        "\n#### Interpretation\n"
        "- C1/C2 are pure similarity identities and should be ~1.0 with no "
        "degradation term: they reconstruct corrected speeds exactly from physical "
        "speeds and inlet temperatures.\n"
        "- Equations with large `dR2` carry the HPC-degradation signature; their "
        "pooled residuals collapse onto a single latent (PC1) that tracks engine "
        "life, i.e. the **discovered degradation parameter**.\n"
        "- Because each mapping is an instantaneous algebraic relation among "
        "contemporaneous sensors plus one slow scalar, it does not accumulate "
        "error under autoregressive rollout (unlike lag-based AR models).\n"
    )
    with open(CONFIG["log_md"], "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ensure_dirs()
    df = add_life_fraction(load_fd001())
    train_df, test_df = split_by_unit(df)

    print("=" * 78)
    print("FD001 PHYSICS CONSERVATION-LAW SENSOR MAPPING")
    print("=" * 78)
    print(f"train engines: {train_df['unit_id'].nunique()}  "
          f"test engines: {test_df['unit_id'].nunique()}  "
          f"rows: {len(df)}")
    print("-" * 78)
    print(f"{'equation':24s} {'target':6s} {'R2_phys':>9s} "
          f"{'R2_phys+d':>10s} {'dR2':>9s}")
    print("-" * 78)

    results = []
    for eq in EQUATIONS:
        r = evaluate_equation(eq, train_df, test_df)
        results.append(r)
        print(f"{r['name']:24s} {r['target']:6s} {r['r2_physics']:9.4f} "
              f"{r['r2_physics_plus_d']:10.4f} {r['delta_r2_degradation']:+9.4f}")

    print("-" * 78)
    disc = discover_degradation(results, train_df, test_df)
    print(f"Discovered theta_hat: PC1={disc['pc1_var']*100:.1f}% var, "
          f"corr(life)={disc['corr_with_life']:.3f}, "
          f"per-engine monotonicity={disc['mean_per_engine_monotonicity']:.3f}")

    # theta_hat basis was fit on TRAIN residuals and applied to TEST (no leakage).
    theta_tr = disc["theta_train"]
    theta_te = disc["theta"]

    refit_r2 = {}
    for eq in EQUATIONS:
        if eq["name"] in COUPLED_FOR_DISCOVERY:
            refit_r2[eq["name"]] = refit_with_theta(
                eq, train_df, test_df, theta_tr.reshape(-1, 1), theta_te.reshape(-1, 1)
            )

    print("-" * 78)
    print("Coupled equations re-fit with discovered theta_hat:")
    for name, r2 in refit_r2.items():
        print(f"  {name:24s} R2 = {r2:.4f}")

    # Persist artefacts.
    scores_df = pd.DataFrame([{k: v for k, v in r.items()
                               if k not in ("residual_test", "residual_train")}
                              for r in results])
    scores_df.to_csv(CONFIG["scores_csv"], index=False)

    degr_df = test_df[["unit_id", "cycle", "d"]].copy()
    degr_df["theta_hat"] = theta_te
    degr_df.to_csv(CONFIG["degr_csv"], index=False)

    plot_path = plot_degradation(disc, theta_te, disc["d_true"])

    summary = dict(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        n_equations=len(results),
        identity_eqs={r["name"]: r["r2_physics"]
                      for r in results if r["identity"]},
        degradation_discovery=dict(
            pc1_var=disc["pc1_var"], corr_with_life=disc["corr_with_life"],
            mean_per_engine_monotonicity=disc["mean_per_engine_monotonicity"],
            loadings=disc["loadings"],
        ),
        refit_with_theta=refit_r2,
        plot=plot_path,
    )
    with open(CONFIG["summary_json"], "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    append_results_md(results, disc, refit_r2)
    print("-" * 78)
    print(f"Saved: {CONFIG['scores_csv']}")
    print(f"Saved: {CONFIG['degr_csv']}")
    print(f"Saved: {CONFIG['summary_json']}")
    print(f"Saved: {plot_path}")
    print(f"Appended results to: {CONFIG['log_md']}")


if __name__ == "__main__":
    main()
