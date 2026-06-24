import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]

CONFIG = {
    "fd": 1,
    "seed": 42,
    "test_size": 0.2,
    "stationary_std_threshold": 1e-4,
    "target_r2": 0.90,
    "out_dir": "physics_hypothesis_outputs_v2",
    "scores_csv": "physics_hypothesis_outputs_v2/all_scores.csv",
    "best_csv": "physics_hypothesis_outputs_v2/best_per_sensor.csv",
    "summary_json": "physics_hypothesis_outputs_v2/summary.json",
    "equation_log_md": "physics_hypothesis_outputs_v2/equation_log.md",
    "top_plot": "plotting/physics_hypothesis_v2/top_r2_best.png",
}


def ensure_dirs() -> None:
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["top_plot"]), exist_ok=True)


def load_fd001() -> pd.DataFrame:
    path = f"Data/train_FD00{CONFIG['fd']}.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing parquet file: {path}")
    return pd.read_parquet(path)


def add_health_state_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    max_cycle = out.groupby("unit_id")["cycle"].transform("max")
    out["degr"] = out["cycle"] / max_cycle
    out["degr2"] = out["degr"] ** 2
    out["degr3"] = out["degr"] ** 3
    out["rul"] = (max_cycle - out["cycle"]).astype(float)
    out["rul_norm"] = out["rul"] / max_cycle
    out["inv_rul"] = 1.0 / (out["rul_norm"] + 1e-3)
    out["exp_neg_degr"] = np.exp(-3.0 * out["degr"])
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["unit_id", "cycle"]).copy()
    for s in SENSOR_COLS:
        out[f"{s}_lag1"] = out.groupby("unit_id")[s].shift(1)
        out[f"{s}_lag2"] = out.groupby("unit_id")[s].shift(2)
        out[f"{s}_diff1"] = out[s] - out[f"{s}_lag1"]
    return out


def split_by_unit(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    units = sorted(df["unit_id"].unique().tolist())
    tr_units, te_units = train_test_split(
        units, test_size=CONFIG["test_size"], random_state=CONFIG["seed"]
    )
    train_df = df[df["unit_id"].isin(tr_units)].copy()
    test_df = df[df["unit_id"].isin(te_units)].copy()
    return train_df, test_df


def hypotheses_catalog() -> List[Dict[str, str]]:
    return [
        {
            "name": "H0_settings_linear",
            "model": "linear",
            "equation": "y_s(t) = a0 + a1*u1(t) + a2*u2(t) + a3*u3(t)",
            "notes": "Settings-only baseline",
        },
        {
            "name": "H1_degr_poly_ridge",
            "model": "ridge_scaled",
            "equation": "y_s(t) = b0 + b1*d + b2*d^2 + b3*d^3, d=t/T_fail",
            "notes": "Monotone polynomial degradation",
        },
        {
            "name": "H2_settings_degr_xgb",
            "model": "xgb",
            "equation": "y_s(t) = F_xgb(u(t), d, d^2, d^3, 1/(rul+eps), exp(-3d))",
            "notes": "Nonlinear settings + degradation",
        },
        {
            "name": "H3_autoreg_targetlag_xgb",
            "model": "xgb",
            "equation": "y_s(t) = F_xgb(u(t), d, y_s(t-1), y_s(t-2), Delta y_s(t-1))",
            "notes": "Target autoregression + health state",
        },
        {
            "name": "H4_crosssensor_autoreg_xgb",
            "model": "xgb",
            "equation": "y_s(t) = F_xgb(u(t), d, {y_j(t-1)}_{j=1..21}, y_s(t-2), Delta y_s(t-1))",
            "notes": "Cross-sensor coupled autoregression",
        },
        {
            "name": "H5_physics_basis_xgb",
            "model": "xgb",
            "equation": "y_s(t) = F_xgb(u, d, d^2, d^3, sqrt(d), log(1+1/rul), exp(-3d), u*d)",
            "notes": "Physics-inspired nonlinear basis",
        },
    ]


def build_features_for_hypothesis(df: pd.DataFrame, sensor: str, hyp_name: str) -> pd.DataFrame:
    if hyp_name == "H0_settings_linear":
        return df[SETTING_COLS].copy()

    if hyp_name == "H1_degr_poly_ridge":
        return df[["degr", "degr2", "degr3"]].copy()

    if hyp_name == "H2_settings_degr_xgb":
        return df[SETTING_COLS + ["degr", "degr2", "degr3", "inv_rul", "exp_neg_degr"]].copy()

    if hyp_name == "H3_autoreg_targetlag_xgb":
        cols = [
            *SETTING_COLS,
            "degr",
            "degr2",
            f"{sensor}_lag1",
            f"{sensor}_lag2",
            f"{sensor}_diff1",
        ]
        return df[cols].copy()

    if hyp_name == "H4_crosssensor_autoreg_xgb":
        lag1_all = [f"{s}_lag1" for s in SENSOR_COLS]
        cols = [
            *SETTING_COLS,
            "degr",
            "degr2",
            *lag1_all,
            f"{sensor}_lag2",
            f"{sensor}_diff1",
        ]
        return df[cols].copy()

    if hyp_name == "H5_physics_basis_xgb":
        x = df[SETTING_COLS + ["degr", "degr2", "degr3", "inv_rul", "exp_neg_degr"]].copy()
        x["sqrt_degr"] = np.sqrt(np.clip(df["degr"].to_numpy(), 0.0, None))
        x["log1p_inv_rul"] = np.log1p(df["inv_rul"].to_numpy())
        for s in SETTING_COLS:
            x[f"{s}_x_degr"] = df[s].to_numpy() * df["degr"].to_numpy()
        return x

    raise ValueError(f"Unknown hypothesis: {hyp_name}")


def fit_predict(model_kind: str, x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame) -> np.ndarray:
    if model_kind == "linear":
        m = LinearRegression()
        m.fit(x_train, y_train)
        return m.predict(x_test)

    if model_kind == "ridge_scaled":
        scaler = StandardScaler()
        xtr = scaler.fit_transform(x_train)
        xte = scaler.transform(x_test)
        m = Ridge(alpha=1.0, random_state=CONFIG["seed"])
        m.fit(xtr, y_train)
        return m.predict(xte)

    if model_kind == "xgb":
        m = XGBRegressor(
            n_estimators=900,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=CONFIG["seed"],
            n_jobs=4,
        )
        m.fit(x_train, y_train)
        return m.predict(x_test)

    raise ValueError(f"Unknown model kind: {model_kind}")


def evaluate_all(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    catalog = hypotheses_catalog()
    rows: List[Dict[str, float]] = []

    for h in catalog:
        for s in SENSOR_COLS:
            tr = train_df.copy()
            te = test_df.copy()

            xtr = build_features_for_hypothesis(tr, s, h["name"])
            xte = build_features_for_hypothesis(te, s, h["name"])
            ytr = tr[s].to_numpy()
            yte = te[s].to_numpy()

            valid_train = ~xtr.isna().any(axis=1)
            valid_test = ~xte.isna().any(axis=1)

            xtr = xtr.loc[valid_train]
            ytr = ytr[valid_train.to_numpy()]
            xte = xte.loc[valid_test]
            yte = yte[valid_test.to_numpy()]

            if len(xtr) < 50 or len(xte) < 20:
                continue

            yhat = fit_predict(h["model"], xtr, ytr, xte)
            rows.append(
                {
                    "hypothesis": h["name"],
                    "model": h["model"],
                    "equation": h["equation"],
                    "notes": h["notes"],
                    "sensor": s,
                    "r2": float(r2_score(yte, yhat)),
                    "mae": float(mean_absolute_error(yte, yhat)),
                    "y_test_std": float(np.std(yte)),
                    "n_train": int(len(xtr)),
                    "n_test": int(len(xte)),
                    "n_features": int(xtr.shape[1]),
                }
            )

    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    best = scores.sort_values(["sensor", "r2"], ascending=[True, False]).groupby("sensor", as_index=False).first()

    nonstationary = best[best["y_test_std"] > CONFIG["stationary_std_threshold"]].copy()
    nonstationary["target_met"] = nonstationary["r2"] >= CONFIG["target_r2"]

    stats = {
        "num_nonstationary": int(len(nonstationary)),
        "num_target_met": int(nonstationary["target_met"].sum()),
        "num_target_not_met": int((~nonstationary["target_met"]).sum()),
    }
    return best, stats


def save_plot(best: pd.DataFrame) -> None:
    p = best.sort_values("r2", ascending=False)
    plt.figure(figsize=(10, 7), dpi=300)
    plt.barh(np.arange(len(p)), p["r2"], color="black", alpha=0.9)
    plt.yticks(np.arange(len(p)), p["sensor"] + "|" + p["hypothesis"])
    plt.gca().invert_yaxis()
    plt.axvline(CONFIG["target_r2"], color="red", linestyle="--", linewidth=1)
    plt.xlabel("Best R2 per sensor")
    plt.title("FD001 Best Sensor Models (Target R2=0.9)")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG["top_plot"])
    plt.close()


def write_equation_log_md(scores: pd.DataFrame, best: pd.DataFrame, stats: Dict[str, int]) -> None:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    hypothesis_summary = (
        scores.groupby(["hypothesis", "equation", "model", "notes"], as_index=False)
        .agg(mean_r2=("r2", "mean"), max_r2=("r2", "max"), sensors_ge_09=("r2", lambda x: int((x >= 0.9).sum())))
        .sort_values("mean_r2", ascending=False)
    )

    lines: List[str] = []
    lines.append("# FD001 Physics Hypothesis Equation Log")
    lines.append("")
    lines.append(f"Updated: {now}")
    lines.append("")
    lines.append("## Target")
    lines.append("")
    lines.append("- Goal: R2 > 0.9 for all non-stationary sensors")
    lines.append(f"- Non-stationary sensors: {stats['num_nonstationary']}")
    lines.append(f"- Sensors meeting target: {stats['num_target_met']}")
    lines.append(f"- Sensors below target: {stats['num_target_not_met']}")
    lines.append("")
    lines.append("## Equations Tested")
    lines.append("")
    lines.append("| Hypothesis | Equation | Model | Mean R2 | Max R2 | #Sensors R2>=0.9 | Notes |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for r in hypothesis_summary.itertuples():
        lines.append(
            f"| {r.hypothesis} | {r.equation} | {r.model} | {r.mean_r2:.4f} | {r.max_r2:.4f} | {r.sensors_ge_09} | {r.notes} |"
        )

    lines.append("")
    lines.append("## Best Model Per Sensor")
    lines.append("")
    lines.append("| Sensor | Best Hypothesis | R2 | MAE | y_std |")
    lines.append("|---|---|---:|---:|---:|")
    for r in best.sort_values("sensor").itertuples():
        lines.append(f"| {r.sensor} | {r.hypothesis} | {r.r2:.4f} | {r.mae:.6f} | {r.y_test_std:.6f} |")

    with open(CONFIG["equation_log_md"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run() -> None:
    ensure_dirs()
    df = load_fd001()
    df = add_health_state_features(df)
    df = add_lag_features(df)
    train_df, test_df = split_by_unit(df)

    scores = evaluate_all(train_df, test_df)
    scores.to_csv(CONFIG["scores_csv"], index=False)

    best, stats = summarize(scores)
    best.to_csv(CONFIG["best_csv"], index=False)

    payload = {
        "dataset": "FD001",
        "target_r2": CONFIG["target_r2"],
        "stationary_std_threshold": CONFIG["stationary_std_threshold"],
        "stats": stats,
        "best_per_sensor": best[["sensor", "hypothesis", "r2", "mae", "y_test_std"]].to_dict(orient="records"),
    }
    with open(CONFIG["summary_json"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    save_plot(best)
    write_equation_log_md(scores, best, stats)

    print(f"Saved scores: {CONFIG['scores_csv']}")
    print(f"Saved best: {CONFIG['best_csv']}")
    print(f"Saved summary: {CONFIG['summary_json']}")
    print(f"Saved equation log: {CONFIG['equation_log_md']}")
    print(f"Saved plot: {CONFIG['top_plot']}")
    print(f"Non-stationary sensors: {stats['num_nonstationary']}")
    print(f"Target met (R2>=0.9): {stats['num_target_met']}")


if __name__ == "__main__":
    run()
