import os
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from utils import (
    load_parquet,
    SENSOR_COLS,
    SENSOR_LABELS,
    min_max_scale_columns,
    apply_min_max_scale_columns,
)


def _sensor_columns_in_df(df: pd.DataFrame):
    """Return sensor columns present in dataframe, supporting s1 and s_1 naming."""
    s_no_underscore = [f"s{i}" for i in range(1, 22)]
    s_with_underscore = [f"s_{i}" for i in range(1, 22)]

    if all(c in df.columns for c in s_no_underscore):
        return s_no_underscore
    if all(c in df.columns for c in s_with_underscore):
        return s_with_underscore

    # Fallback: use any non-setting sensor-like columns that exist.
    fallback = [c for c in (s_no_underscore + s_with_underscore) if c in df.columns]
    if len(fallback) == 0:
        raise ValueError("No sensor columns found. Expected s1..s21 or s_1..s_21.")
    return fallback


CONFIG = {
    "dataset_id": 1,
    "split": {
        "train_series": 80,
        "test_series": 20,
        "random_seed": 42,
    },
    "eventing": {
        "quantile": 0.95,
        "eps": 1e-12,
    },
    "plot": {
        "nrows": 7,
        "ncols": 3,
        "figsize": (36, 28),
        "dpi": 300,
        "rolling_window": 10,
    },
    "output": {
        "root_dir": "hawkes_outputs",
        "per_sensor_parquet_dir": "hawkes_outputs/per_sensor_data",
        "per_sensor_plot_dir": "plotting/hawkes/per_sensor_pred_vs_truth",
        "summary_plot_path": "plotting/hawkes/pred_vs_truth_multipanel.png",
        "metrics_csv": "hawkes_outputs/hawkes_metrics.csv",
        "config_json": "hawkes_outputs/config.json",
    },
}


@dataclass
class HawkesParams:
    mu: float
    alpha: float
    beta: float


def _ensure_dirs():
    os.makedirs(CONFIG["output"]["root_dir"], exist_ok=True)
    os.makedirs(CONFIG["output"]["per_sensor_parquet_dir"], exist_ok=True)
    os.makedirs(CONFIG["output"]["per_sensor_plot_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["output"]["summary_plot_path"]), exist_ok=True)


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    if arr.size == 0:
        return arr
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="same")


def split_units(df: pd.DataFrame, train_series: int, test_series: int, seed: int):
    unit_ids = sorted(df["unit_id"].unique().tolist())
    needed = train_series + test_series
    if len(unit_ids) < needed:
        raise ValueError(
            f"Not enough units for requested split: need {needed}, found {len(unit_ids)}"
        )

    rng = np.random.default_rng(seed)
    shuffled = unit_ids.copy()
    rng.shuffle(shuffled)

    train_ids = sorted(shuffled[:train_series])
    test_ids = sorted(shuffled[train_series : train_series + test_series])
    return train_ids, test_ids


def build_diff_rows_for_sensor(df: pd.DataFrame, sensor_col: str, split_map: dict):
    rows = []
    unit_groups = df.sort_values(["unit_id", "cycle"]).groupby("unit_id", sort=True)

    for unit_id, grp in unit_groups:
        values = grp[sensor_col].to_numpy(dtype=np.float64)
        cycles = grp["cycle"].to_numpy(dtype=np.int64)
        if len(values) < 2:
            continue

        diffs = np.diff(values)
        diff_cycles = cycles[1:]
        split = split_map.get(unit_id)
        if split is None:
            continue

        for t, d in zip(diff_cycles, diffs):
            rows.append(
                {
                    "unit_id": int(unit_id),
                    "split": split,
                    "cycle": int(t),
                    "delta_raw": float(d),
                }
            )

    return pd.DataFrame(rows)


def min_max_train_apply_all(sensor_df: pd.DataFrame):
    """Use shared utils min-max scaler: fit on train rows, apply to all rows."""
    train_mask = sensor_df["split"] == "train"

    train_df = sensor_df.loc[train_mask, ["delta_raw"]].copy()
    _, stats = min_max_scale_columns(
        train_df,
        columns=["delta_raw"],
        return_stats=True,
    )

    transformed = apply_min_max_scale_columns(
        sensor_df[["delta_raw"]],
        stats=stats,
        columns=["delta_raw"],
    )

    sensor_df = sensor_df.copy()
    sensor_df["delta_minmax"] = transformed["delta_raw"].to_numpy(dtype=np.float64)
    sensor_df["abs_delta_minmax"] = np.abs(sensor_df["delta_minmax"])

    train_min = float(stats["min"]["delta_raw"])
    train_max = float(stats["max"]["delta_raw"])
    return sensor_df, train_min, train_max


def make_events(sensor_df: pd.DataFrame, quantile: float):
    train_mask = sensor_df["split"] == "train"
    train_abs = sensor_df.loc[train_mask, "abs_delta_minmax"].to_numpy(dtype=np.float64)

    if train_abs.size == 0:
        tau = np.inf
    else:
        tau = float(np.quantile(train_abs, quantile))

    sensor_df = sensor_df.copy()
    sensor_df["event"] = (sensor_df["abs_delta_minmax"] >= tau).astype(int)
    return sensor_df, tau


def _sequence_event_times_and_horizon(sensor_df: pd.DataFrame, split: str):
    event_times = []
    horizons = []

    for _, grp in sensor_df[sensor_df["split"] == split].groupby("unit_id", sort=True):
        grp_sorted = grp.sort_values("cycle")
        cycles = grp_sorted["cycle"].to_numpy(dtype=np.float64)
        event_flags = grp_sorted["event"].to_numpy(dtype=np.int64)

        if cycles.size == 0:
            continue

        start_cycle = cycles[0]
        rel_time = cycles - start_cycle + 1.0
        T = float(rel_time[-1])

        t_events = rel_time[event_flags == 1]
        event_times.append(t_events.astype(np.float64))
        horizons.append(T)

    return event_times, horizons


def _neg_loglik_exp_hawkes(theta, seq_events, seq_horizons):
    mu, alpha, beta = theta
    if mu <= 0 or alpha < 0 or beta <= 0:
        return np.inf

    nll = 0.0
    for events, T in zip(seq_events, seq_horizons):
        if T <= 0:
            continue

        # Integral term
        integral = mu * T
        if events.size > 0 and alpha > 0:
            integral += (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - events)))

        # Event log-intensity term
        log_sum = 0.0
        A_prev = 0.0
        t_prev = None
        for i, t_i in enumerate(events):
            if i == 0:
                A_i = 0.0
            else:
                dt = t_i - t_prev
                A_i = np.exp(-beta * dt) * (1.0 + A_prev)
            lam = mu + alpha * A_i
            if lam <= 0:
                return np.inf
            log_sum += np.log(lam)
            A_prev = A_i
            t_prev = t_i

        nll += (integral - log_sum)

    return float(nll)


def fit_exp_hawkes(seq_events, seq_horizons):
    x0 = np.array([0.05, 0.2, 1.0], dtype=np.float64)
    bounds = [(1e-8, 50.0), (0.0, 50.0), (1e-6, 50.0)]

    res = minimize(
        _neg_loglik_exp_hawkes,
        x0=x0,
        args=(seq_events, seq_horizons),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000},
    )

    if not res.success:
        # Fall back to a conservative parameter set if optimizer struggles.
        mu, alpha, beta = x0.tolist()
    else:
        mu, alpha, beta = res.x.tolist()

    return HawkesParams(mu=float(mu), alpha=float(alpha), beta=float(beta)), res


def loglik_per_event(params: HawkesParams, seq_events, seq_horizons):
    n_events = int(sum(len(x) for x in seq_events))
    if n_events == 0:
        return np.nan
    nll = _neg_loglik_exp_hawkes(
        np.array([params.mu, params.alpha, params.beta], dtype=np.float64),
        seq_events,
        seq_horizons,
    )
    return float(-nll / n_events)


def predict_intensity_path_for_unit(unit_df: pd.DataFrame, params: HawkesParams):
    grp = unit_df.sort_values("cycle")
    cycles = grp["cycle"].to_numpy(dtype=np.int64)
    event = grp["event"].to_numpy(dtype=np.float64)

    if cycles.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    intensity = np.zeros_like(event, dtype=np.float64)
    event_times = []
    t0 = float(cycles[0])

    for i, cyc in enumerate(cycles):
        t = float(cyc - t0 + 1.0)

        if len(event_times) == 0:
            excitation = 0.0
        else:
            prev = np.array(event_times, dtype=np.float64)
            excitation = np.sum(np.exp(-params.beta * (t - prev)))

        lam = params.mu + params.alpha * excitation
        intensity[i] = lam

        if event[i] > 0.5:
            event_times.append(t)

    return event, intensity


def plot_per_sensor_pred_vs_truth(sensor_name: str, label: str, event: np.ndarray, pred: np.ndarray, tau: float):
    window = CONFIG["plot"]["rolling_window"]
    event_smooth = _rolling_mean(event, window)
    pred_smooth = _rolling_mean(pred, window)

    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(event_smooth, color="black", linewidth=1.5, label="Truth (event rate)")
    plt.plot(pred_smooth, color="red", linestyle="--", linewidth=1.5, label="Pred (intensity)")
    plt.title(f"{sensor_name} | {label} | rolling window={window} | tau={tau:.4f}")
    plt.xlabel("Cycle index")
    plt.ylabel("Rate / Intensity")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(CONFIG["output"]["per_sensor_plot_dir"], f"{sensor_name}_pred_vs_truth.png")
    plt.savefig(out_path)
    plt.close()


def plot_multipanel_pred_vs_truth(series_payload):
    nrows = CONFIG["plot"]["nrows"]
    ncols = CONFIG["plot"]["ncols"]
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=CONFIG["plot"]["figsize"], dpi=CONFIG["plot"]["dpi"])
    axes = axes.flatten()

    window = CONFIG["plot"]["rolling_window"]

    for i in range(len(axes)):
        ax = axes[i]
        if i >= len(series_payload):
            ax.axis("off")
            continue

        payload = series_payload[i]
        event = _rolling_mean(payload["event"], window)
        pred = _rolling_mean(payload["pred"], window)

        ax.plot(event, color="black", linewidth=1.5, label="Truth")
        ax.plot(pred, color="red", linestyle="--", linewidth=1.5, label="Pred")
        ax.set_title(payload["title"])
        ax.grid(True, alpha=0.3)

    if len(axes) > 0:
        axes[0].legend(loc="upper left")

    fig.suptitle("Hawkes on Sensor Diffs: Pred vs Truth (Test Units)", fontsize=16)
    plt.tight_layout()
    plt.savefig(CONFIG["output"]["summary_plot_path"])
    plt.close(fig)


def run():
    _ensure_dirs()

    with open(CONFIG["output"]["config_json"], "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)

    df = load_parquet(fd=CONFIG["dataset_id"], split="train")

    train_ids, test_ids = split_units(
        df,
        train_series=CONFIG["split"]["train_series"],
        test_series=CONFIG["split"]["test_series"],
        seed=CONFIG["split"]["random_seed"],
    )

    split_map = {u: "train" for u in train_ids}
    split_map.update({u: "test" for u in test_ids})

    metrics_rows = []
    summary_payload = []

    sensor_cols = _sensor_columns_in_df(df)
    label_map = {}
    for i in range(1, 22):
        label = SENSOR_LABELS[i - 1] if i - 1 < len(SENSOR_LABELS) else f"Sensor {i}"
        label_map[f"s{i}"] = label
        label_map[f"s_{i}"] = label

    for sensor in sensor_cols:
        sensor_label = label_map.get(sensor, sensor)
        sensor_df = build_diff_rows_for_sensor(df, sensor, split_map)

        if sensor_df.empty:
            continue

        sensor_df, train_min, train_max = min_max_train_apply_all(sensor_df)
        sensor_df, tau = make_events(sensor_df, quantile=CONFIG["eventing"]["quantile"])

        out_parquet = os.path.join(CONFIG["output"]["per_sensor_parquet_dir"], f"{sensor}.parquet")
        sensor_df.to_parquet(out_parquet, index=False)

        train_events, train_horizons = _sequence_event_times_and_horizon(sensor_df, "train")
        test_events, test_horizons = _sequence_event_times_and_horizon(sensor_df, "test")

        params, fit_res = fit_exp_hawkes(train_events, train_horizons)

        train_ll = loglik_per_event(params, train_events, train_horizons)
        test_ll = loglik_per_event(params, test_events, test_horizons)

        n_train_events = int(sum(len(x) for x in train_events))
        n_test_events = int(sum(len(x) for x in test_events))

        metrics_rows.append(
            {
                "sensor": sensor,
                "sensor_label": sensor_label,
                "min_train_only": train_min,
                "max_train_only": train_max,
                "event_threshold_tau": tau,
                "mu": params.mu,
                "alpha": params.alpha,
                "beta": params.beta,
                "train_event_count": n_train_events,
                "test_event_count": n_test_events,
                "train_loglik_per_event": train_ll,
                "test_loglik_per_event": test_ll,
                "optimizer_success": bool(getattr(fit_res, "success", False)),
                "optimizer_message": str(getattr(fit_res, "message", "")),
            }
        )

        # Representative test-unit pred-v-truth for plotting
        test_unit_ids = sorted(sensor_df[sensor_df["split"] == "test"]["unit_id"].unique().tolist())
        if len(test_unit_ids) > 0:
            unit_id = test_unit_ids[0]
            unit_df = sensor_df[(sensor_df["split"] == "test") & (sensor_df["unit_id"] == unit_id)].copy()
            y_true, y_pred = predict_intensity_path_for_unit(unit_df, params)
            plot_per_sensor_pred_vs_truth(sensor, sensor_label, y_true, y_pred, tau)

            summary_payload.append(
                {
                    "title": f"{sensor} | unit {unit_id}",
                    "event": y_true,
                    "pred": y_pred,
                }
            )

        print(
            f"[{sensor}] events train/test={n_train_events}/{n_test_events} | "
            f"mu={params.mu:.4f}, alpha={params.alpha:.4f}, beta={params.beta:.4f}"
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("sensor")
    metrics_df.to_csv(CONFIG["output"]["metrics_csv"], index=False)

    plot_multipanel_pred_vs_truth(summary_payload)

    print("\nDone.")
    print(f"Saved metrics: {CONFIG['output']['metrics_csv']}")
    print(f"Saved per-sensor parquet dir: {CONFIG['output']['per_sensor_parquet_dir']}")
    print(f"Saved summary plot: {CONFIG['output']['summary_plot_path']}")


if __name__ == "__main__":
    run()
