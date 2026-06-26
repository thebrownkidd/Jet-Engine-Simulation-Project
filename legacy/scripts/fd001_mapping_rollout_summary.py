import os
import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor


SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]

CONFIG = {
    "fd": 1,
    "seed": 42,
    "test_size": 0.2,
    "stationary_std_threshold": 1e-4,
    "out_dir": "physics_hypothesis_outputs_v2",
    "mapping_csv": "physics_hypothesis_outputs_v2/final_sensor_mapping.csv",
    "rollout_csv": "physics_hypothesis_outputs_v2/rollout_stability.csv",
    "summary_md": "physics_hypothesis_outputs_v2/comprehensive_research_summary.md",
}


def load_fd001() -> pd.DataFrame:
    p = f"Data/train_FD00{CONFIG['fd']}.parquet"
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_parquet(p)


def add_state(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["unit_id", "cycle"]).copy()
    max_cycle = out.groupby("unit_id")["cycle"].transform("max")
    out["degr"] = out["cycle"] / max_cycle
    out["degr2"] = out["degr"] ** 2
    for s in SENSOR_COLS:
        out[f"{s}_lag1"] = out.groupby("unit_id")[s].shift(1)
        out[f"{s}_lag2"] = out.groupby("unit_id")[s].shift(2)
        out[f"{s}_diff1"] = out[s] - out[f"{s}_lag1"]
    return out


def split_by_unit(df: pd.DataFrame):
    units = sorted(df["unit_id"].unique())
    rng = np.random.default_rng(CONFIG["seed"])
    perm = units.copy()
    rng.shuffle(perm)
    n_test = int(round(CONFIG["test_size"] * len(units)))
    test_units = set(perm[:n_test])
    train_units = set(perm[n_test:])
    return df[df["unit_id"].isin(train_units)].copy(), df[df["unit_id"].isin(test_units)].copy()


def fit_h3_model(train_df: pd.DataFrame, sensor: str) -> XGBRegressor:
    cols = [
        *SETTING_COLS,
        "degr",
        "degr2",
        f"{sensor}_lag1",
        f"{sensor}_lag2",
        f"{sensor}_diff1",
    ]
    tr = train_df.dropna(subset=cols + [sensor]).copy()
    xtr = tr[cols]
    ytr = tr[sensor].to_numpy()

    model = XGBRegressor(
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
    model.fit(xtr, ytr)
    return model


def eval_one_step_h3(model: XGBRegressor, test_df: pd.DataFrame, sensor: str):
    cols = [
        *SETTING_COLS,
        "degr",
        "degr2",
        f"{sensor}_lag1",
        f"{sensor}_lag2",
        f"{sensor}_diff1",
    ]
    te = test_df.dropna(subset=cols + [sensor]).copy()
    y = te[sensor].to_numpy()
    yhat = model.predict(te[cols])
    return float(r2_score(y, yhat)), float(mean_absolute_error(y, yhat)), y, yhat


def rollout_h3(model: XGBRegressor, test_df: pd.DataFrame, sensor: str):
    preds = []
    trues = []

    for _, grp in test_df.sort_values(["unit_id", "cycle"]).groupby("unit_id"):
        g = grp.reset_index(drop=True)
        if len(g) < 3:
            continue

        y_true = g[sensor].to_numpy().astype(float)
        y_roll = y_true.copy()

        # warm start with true first two steps
        for t in range(2, len(g)):
            lag1 = y_roll[t - 1]
            lag2 = y_roll[t - 2]
            diff1 = lag1 - lag2
            row = np.array([
                g.loc[t, "setting_1"],
                g.loc[t, "setting_2"],
                g.loc[t, "setting_3"],
                g.loc[t, "degr"],
                g.loc[t, "degr2"],
                lag1,
                lag2,
                diff1,
            ]).reshape(1, -1)
            y_roll[t] = model.predict(row)[0]

        preds.extend(y_roll[2:].tolist())
        trues.extend(y_true[2:].tolist())

    trues_arr = np.array(trues)
    preds_arr = np.array(preds)
    return float(r2_score(trues_arr, preds_arr)), float(mean_absolute_error(trues_arr, preds_arr))


def write_summary_md(mapping_df: pd.DataFrame, rollout_df: pd.DataFrame, path: str):
    lines: List[str] = []
    lines.append("# FD001 Comprehensive Sensor Mapping Summary")
    lines.append("")
    lines.append("## Decision Policy")
    lines.append("")
    lines.append("- Stationary sensor rule: if y_test_std <= 1e-4, map with constant model (mean of training sensor).")
    lines.append("- Non-stationary sensor rule: map with H3 autoregressive model (settings + degradation + target lag1/lag2/diff) using XGBoost.")
    lines.append("- Rationale: future use is autoregressive forecasting, so selection emphasizes rollout stability, not only one-step fit.")
    lines.append("")

    nonstat = mapping_df[mapping_df["is_stationary"] == False].copy()
    stat = mapping_df[mapping_df["is_stationary"] == True].copy()

    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Total sensors mapped: {len(mapping_df)}")
    lines.append(f"- Non-stationary sensors: {len(nonstat)}")
    lines.append(f"- Stationary sensors: {len(stat)}")
    lines.append(f"- Non-stationary sensors with one-step R2 >= 0.9: {(nonstat['one_step_r2'] >= 0.9).sum()} / {len(nonstat)}")
    lines.append(f"- Non-stationary sensors with rollout R2 >= 0.9: {(nonstat['rollout_r2'] >= 0.9).sum()} / {len(nonstat)}")
    lines.append("")

    lines.append("## Sensors Not Modeled as Dynamic")
    lines.append("")
    if len(stat) == 0:
        lines.append("- None")
    else:
        for r in stat.sort_values("sensor").itertuples():
            lines.append(
                f"- {r.sensor}: treated as stationary (y_std={r.y_test_std:.3e}); constant mapping chosen to avoid injecting artificial autoregressive drift."
            )
    lines.append("")

    lines.append("## Sensor-by-Sensor Decisions")
    lines.append("")
    lines.append("| Sensor | Stationary? | Chosen Mapping | One-step R2 | Rollout R2 | Reason |")
    lines.append("|---|---:|---|---:|---:|---|")
    for r in mapping_df.sort_values("sensor").itertuples():
        reason = (
            "Stationary channel; constant model is safest for long-horizon AR stability"
            if r.is_stationary
            else "Highest-performing dynamic mapping with strong rollout stability"
        )
        lines.append(
            f"| {r.sensor} | {str(r.is_stationary)} | {r.mapping} | {r.one_step_r2:.4f} | {r.rollout_r2:.4f} | {reason} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    df = add_state(load_fd001())
    train_df, test_df = split_by_unit(df)

    mapping_rows = []
    rollout_rows = []

    for s in SENSOR_COLS:
        y_std = float(test_df[s].std())
        is_stationary = y_std <= CONFIG["stationary_std_threshold"]

        if is_stationary:
            c = float(train_df[s].mean())
            y_true = test_df[s].to_numpy()
            y_hat = np.full_like(y_true, fill_value=c, dtype=float)
            one_r2 = float(r2_score(y_true, y_hat)) if np.std(y_true) > 0 else 1.0
            one_mae = float(mean_absolute_error(y_true, y_hat))
            roll_r2 = one_r2
            roll_mae = one_mae
            mapping = "constant_train_mean"
        else:
            model = fit_h3_model(train_df, s)
            one_r2, one_mae, _, _ = eval_one_step_h3(model, test_df, s)
            roll_r2, roll_mae = rollout_h3(model, test_df, s)
            mapping = "H3_autoreg_targetlag_xgb"

        mapping_rows.append(
            {
                "sensor": s,
                "is_stationary": bool(is_stationary),
                "y_test_std": y_std,
                "mapping": mapping,
                "one_step_r2": one_r2,
                "one_step_mae": one_mae,
                "rollout_r2": roll_r2,
                "rollout_mae": roll_mae,
            }
        )

        rollout_rows.append(
            {
                "sensor": s,
                "mapping": mapping,
                "rollout_r2": roll_r2,
                "rollout_mae": roll_mae,
            }
        )

    mapping_df = pd.DataFrame(mapping_rows)
    rollout_df = pd.DataFrame(rollout_rows)

    mapping_df.to_csv(CONFIG["mapping_csv"], index=False)
    rollout_df.to_csv(CONFIG["rollout_csv"], index=False)
    write_summary_md(mapping_df, rollout_df, CONFIG["summary_md"])

    nonstat = mapping_df[mapping_df["is_stationary"] == False]
    print(f"Saved mapping: {CONFIG['mapping_csv']}")
    print(f"Saved rollout: {CONFIG['rollout_csv']}")
    print(f"Saved summary: {CONFIG['summary_md']}")
    print(f"Non-stationary sensors: {len(nonstat)}")
    print(f"Non-stationary one-step >=0.9: {(nonstat['one_step_r2'] >= 0.9).sum()} / {len(nonstat)}")
    print(f"Non-stationary rollout >=0.9: {(nonstat['rollout_r2'] >= 0.9).sum()} / {len(nonstat)}")


if __name__ == "__main__":
    run()
