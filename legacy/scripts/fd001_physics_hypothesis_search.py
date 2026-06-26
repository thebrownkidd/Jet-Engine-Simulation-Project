import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]


CONFIG = {
    "fd": 1,
    "seed": 42,
    "test_size": 0.2,
    "ridge_alpha": 1.0,
    "success": {
        "min_r2": 0.20,
        "min_delta_r2": 0.05,
    },
    "out_dir": "physics_hypothesis_outputs",
    "summary_csv": "physics_hypothesis_outputs/hypothesis_sensor_scores.csv",
    "success_csv": "physics_hypothesis_outputs/successes.csv",
    "best_json": "physics_hypothesis_outputs/best_hypotheses.json",
    "top_plot": "plotting/physics_hypothesis/top_improvements.png",
}


def ensure_dirs():
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["top_plot"]), exist_ok=True)


def load_fd001() -> pd.DataFrame:
    path = f"Data/train_FD00{CONFIG['fd']}.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing parquet: {path}")
    return pd.read_parquet(path)


def add_degradation_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    max_cycle = out.groupby("unit_id")["cycle"].transform("max")
    out["life_frac"] = out["cycle"] / max_cycle
    out["degr"] = out["life_frac"]
    out["rul_train"] = (max_cycle - out["cycle"]).astype(float)
    out["rul_norm"] = out["rul_train"] / max_cycle
    out["degr2"] = out["degr"] ** 2
    out["degr3"] = out["degr"] ** 3
    out["inv_remaining"] = 1.0 / (out["rul_norm"] + 1e-3)
    out["log_rul"] = np.log1p(out["rul_train"])
    return out


def split_by_unit(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    units = sorted(df["unit_id"].unique())
    train_units, test_units = train_test_split(
        units, test_size=CONFIG["test_size"], random_state=CONFIG["seed"]
    )
    train_df = df[df["unit_id"].isin(train_units)].copy()
    test_df = df[df["unit_id"].isin(test_units)].copy()
    return train_df, test_df


@dataclass
class Hypothesis:
    name: str
    description: str
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame]
    model_kind: str


def fb_settings_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[SETTING_COLS].copy()


def fb_degr_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[["degr", "degr2", "degr3"]].copy()


def fb_settings_plus_degr(df: pd.DataFrame) -> pd.DataFrame:
    return df[SETTING_COLS + ["degr", "degr2"]].copy()


def fb_settings_degr_interactions(df: pd.DataFrame) -> pd.DataFrame:
    x = df[SETTING_COLS + ["degr", "degr2"]].copy()
    for s in SETTING_COLS:
        x[f"{s}_x_degr"] = df[s] * df["degr"]
        x[f"{s}_x_degr2"] = df[s] * df["degr2"]
    return x


def fb_reciprocal_remaining(df: pd.DataFrame) -> pd.DataFrame:
    x = df[SETTING_COLS + ["inv_remaining", "rul_norm"]].copy()
    for s in SETTING_COLS:
        x[f"{s}_x_inv_remaining"] = df[s] * df["inv_remaining"]
    return x


def fb_nonlinear_basis(df: pd.DataFrame) -> pd.DataFrame:
    x = df[SETTING_COLS + ["degr", "degr2", "rul_norm"]].copy()
    x["sqrt_degr"] = np.sqrt(np.clip(df["degr"], 0.0, None))
    x["log1p_inv_remaining"] = np.log1p(df["inv_remaining"])
    x["exp_neg_degr"] = np.exp(-3.0 * df["degr"])
    for s in SETTING_COLS:
        x[f"{s}_x_sqrt_degr"] = df[s] * x["sqrt_degr"]
    return x


def fb_arrhenius_like(df: pd.DataFrame) -> pd.DataFrame:
    x = df[SETTING_COLS + ["degr"]].copy()
    # Arrhenius-like surrogate: 1/(T) term proxied by remaining-life normalized state.
    x["inv_temp_proxy"] = 1.0 / (df["rul_norm"] + 0.05)
    return x


def build_hypotheses() -> List[Hypothesis]:
    return [
        Hypothesis(
            name="H0_settings_linear",
            description="Sensor output is primarily controlled by operating settings.",
            feature_builder=fb_settings_only,
            model_kind="linear",
        ),
        Hypothesis(
            name="H1_degradation_poly",
            description="Sensor output follows monotonic/polynomial degradation trajectory.",
            feature_builder=fb_degr_only,
            model_kind="ridge",
        ),
        Hypothesis(
            name="H2_settings_plus_degr",
            description="Sensors depend on settings and degradation state additively.",
            feature_builder=fb_settings_plus_degr,
            model_kind="ridge",
        ),
        Hypothesis(
            name="H3_interaction_settings_degr",
            description="Operating-condition effects are degradation-dependent (interaction terms).",
            feature_builder=fb_settings_degr_interactions,
            model_kind="ridge",
        ),
        Hypothesis(
            name="H4_reciprocal_remaining",
            description="Near-failure nonlinearity rises as inverse remaining life.",
            feature_builder=fb_reciprocal_remaining,
            model_kind="ridge",
        ),
        Hypothesis(
            name="H5_nonlinear_basis",
            description="Mixed physics-inspired basis (sqrt, reciprocal, exponential degradation terms).",
            feature_builder=fb_nonlinear_basis,
            model_kind="ridge",
        ),
        Hypothesis(
            name="H6_arrhenius_loglinear",
            description="Arrhenius-like exponential dependence linearized in log(sensor).",
            feature_builder=fb_arrhenius_like,
            model_kind="log_linear_nonneg",
        ),
    ]


def make_model(kind: str):
    if kind == "linear":
        return LinearRegression()
    if kind == "ridge":
        return Ridge(alpha=CONFIG["ridge_alpha"], random_state=CONFIG["seed"])
    if kind == "log_linear_nonneg":
        return LinearRegression()
    raise ValueError(f"Unknown model kind: {kind}")


def evaluate_hypothesis(
    h: Hypothesis, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    x_train_raw = h.feature_builder(train_df)
    x_test_raw = h.feature_builder(test_df)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_test = scaler.transform(x_test_raw)

    rows = []
    for s in SENSOR_COLS:
        y_train = train_df[s].to_numpy()
        y_test = test_df[s].to_numpy()

        model = make_model(h.model_kind)
        if h.model_kind == "log_linear_nonneg":
            shift = max(1e-6, -(y_train.min()) + 1e-6)
            y_train_log = np.log(y_train + shift)
            model.fit(x_train, y_train_log)
            y_pred = np.exp(model.predict(x_test)) - shift
        else:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
        rows.append(
            {
                "hypothesis": h.name,
                "description": h.description,
                "sensor": s,
                "r2": float(r2_score(y_test, y_pred)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "y_test_std": float(np.std(y_test)),
                "n_features": x_train.shape[1],
                "model_kind": h.model_kind,
            }
        )
    return pd.DataFrame(rows)


def mark_successes(scores: pd.DataFrame) -> pd.DataFrame:
    baseline = scores[scores["hypothesis"] == "H0_settings_linear"][["sensor", "r2"]].rename(
        columns={"r2": "baseline_r2"}
    )
    merged = scores.merge(baseline, on="sensor", how="left")
    sensor_std = (
        scores.groupby("sensor", as_index=False)["y_test_std"]
        .mean()
        .rename(columns={"y_test_std": "sensor_target_std"})
    )
    merged = merged.merge(sensor_std, on="sensor", how="left")
    merged["delta_r2_vs_settings"] = merged["r2"] - merged["baseline_r2"]
    merged["is_success"] = (
        (merged["r2"] >= CONFIG["success"]["min_r2"])
        & (merged["delta_r2_vs_settings"] >= CONFIG["success"]["min_delta_r2"])
        & (merged["sensor_target_std"] > 1e-4)
        & (merged["hypothesis"] != "H0_settings_linear")
    )
    return merged


def save_top_plot(merged: pd.DataFrame):
    top = merged[merged["hypothesis"] != "H0_settings_linear"].copy()
    top = top.sort_values("delta_r2_vs_settings", ascending=False).head(20)

    plt.figure(figsize=(12, 7), dpi=300)
    labels = [f"{r.sensor}|{r.hypothesis}" for r in top.itertuples()]
    plt.barh(np.arange(len(top)), top["delta_r2_vs_settings"], color="black", alpha=0.9)
    plt.yticks(np.arange(len(top)), labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Delta R2 vs settings-only baseline")
    plt.title("Top Physics-Hypothesis Improvements (FD001)")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG["top_plot"])
    plt.close()


def run():
    ensure_dirs()
    df = add_degradation_features(load_fd001())
    train_df, test_df = split_by_unit(df)

    all_scores = []
    hypotheses = build_hypotheses()
    for h in hypotheses:
        scores = evaluate_hypothesis(h, train_df, test_df)
        all_scores.append(scores)

    score_df = pd.concat(all_scores, ignore_index=True)
    merged = mark_successes(score_df)

    merged.sort_values(["sensor", "r2"], ascending=[True, False]).to_csv(
        CONFIG["summary_csv"], index=False
    )

    successes = merged[merged["is_success"]].sort_values("delta_r2_vs_settings", ascending=False)
    successes.to_csv(CONFIG["success_csv"], index=False)

    best = (
        merged.sort_values(["sensor", "r2"], ascending=[True, False])
        .groupby("sensor", as_index=False)
        .first()
    )

    payload = {
        "dataset": "FD001",
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_successes": int(len(successes)),
        "success_criteria": CONFIG["success"],
        "best_per_sensor": best[
            ["sensor", "hypothesis", "r2", "mae", "delta_r2_vs_settings"]
        ].to_dict(orient="records"),
        "top_successes": successes.head(25)[
            ["sensor", "hypothesis", "r2", "mae", "delta_r2_vs_settings", "description"]
        ].to_dict(orient="records"),
    }

    with open(CONFIG["best_json"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    save_top_plot(merged)

    print(f"Saved summary: {CONFIG['summary_csv']}")
    print(f"Saved successes: {CONFIG['success_csv']}")
    print(f"Saved best json: {CONFIG['best_json']}")
    print(f"Saved top plot: {CONFIG['top_plot']}")
    print(f"Total successful results: {len(successes)}")


if __name__ == "__main__":
    run()
