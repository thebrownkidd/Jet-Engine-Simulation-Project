# Settings-to-Sensors RNN Experiment
# Input:  settings_t  →  Output: sensors_t
# Predict sensor values conditioned on operating settings.

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler


CONFIG = {
    "dataset_id": 1,
    "split": {
        "train_series": 80,
        "test_series": 20,
        "random_seed": 42,
    },
    "model": {
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.2,
        "head_hidden_dims": [256, 128, 64],
        "bidirectional": False,
    },
    "training": {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "device": "cpu",
    },
    "output": {
        "root_dir": "settings_effect_outputs",
        "metrics_csv": "settings_effect_outputs/training_metrics.csv",
        "model_checkpoint": "settings_effect_outputs/model.pt",
        "config_json": "settings_effect_outputs/config.json",
        "convergence_plot": "plotting/settings_effect/convergence.png",
        "feature_importance_plot": "plotting/settings_effect/feature_importance.png",
        "pred_vs_truth_plot": "plotting/settings_effect/pred_vs_truth.png",
    },
}

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]


def _ensure_dirs():
    os.makedirs(CONFIG["output"]["root_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["output"]["convergence_plot"]), exist_ok=True)


def load_parquet(fd: int, split: str = "train"):
    path = f"Data/{split}_FD00{fd}.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}")
    return pd.read_parquet(path)


def split_units(df: pd.DataFrame, train_series: int, test_series: int, seed: int):
    unit_ids = sorted(df["unit_id"].unique().tolist())
    needed = train_series + test_series
    if len(unit_ids) < needed:
        raise ValueError(f"Not enough units: need {needed}, found {len(unit_ids)}")

    rng = np.random.default_rng(seed)
    shuffled = unit_ids.copy()
    rng.shuffle(shuffled)

    train_ids = sorted(shuffled[:train_series])
    test_ids = sorted(shuffled[train_series : train_series + test_series])
    return train_ids, test_ids


def build_sequences(df: pd.DataFrame, split_ids: list):
    """Build settings_t -> sensors_t sequences."""
    sequences_x = []
    sequences_y = []

    unit_groups = df.sort_values(["unit_id", "cycle"]).groupby("unit_id", sort=True)

    for unit_id, grp in unit_groups:
        if unit_id not in split_ids:
            continue

        grp = grp.sort_values("cycle").reset_index(drop=True)
        settings = grp[SETTING_COLS].to_numpy(dtype=np.float32)
        sensors = grp[SENSOR_COLS].to_numpy(dtype=np.float32)

        if len(settings) < 2:
            continue

        for t in range(1, len(settings)):
            x = settings[t]
            y = sensors[t]
            sequences_x.append(x)
            sequences_y.append(y)

    return np.array(sequences_x, dtype=np.float32), np.array(sequences_y, dtype=np.float32)


class SettingsEffectRNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        head_hidden_dims: list,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        layers = []
        prev_dim = hidden_dim
        for h in head_hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        out, _ = self.gru(x.unsqueeze(1))
        out = out[:, -1, :]
        y = self.head(out)
        return y


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    count = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = nn.MSELoss()(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x_batch)
        count += len(x_batch)

    return total_loss / count if count > 0 else np.nan


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    count = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(x_batch)
        loss = nn.MSELoss()(y_pred, y_batch)

        total_loss += loss.item() * len(x_batch)
        count += len(x_batch)

    return total_loss / count if count > 0 else np.nan


def plot_convergence(train_losses, test_losses):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(train_losses, color="blue", linewidth=1.5, label="Train Loss")
    plt.plot(test_losses, color="red", linestyle="--", linewidth=1.5, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Settings -> Sensors GRU Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(CONFIG["output"]["convergence_plot"])
    plt.close()
    print(f"Saved convergence plot: {CONFIG['output']['convergence_plot']}")


def plot_pred_vs_truth(model, x_test, y_test, device):
    model.eval()
    x_test_t = torch.tensor(x_test[:500], dtype=torch.float32, device=device)

    with torch.no_grad():
        y_pred = model(x_test_t).cpu().numpy()

    y_true = y_test[:500]

    fig, axes = plt.subplots(7, 3, figsize=(18, 16), dpi=300)
    axes = axes.flatten()

    for i in range(min(21, len(axes))):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=10)
        ax.plot([y_true[:, i].min(), y_true[:, i].max()], 
                [y_true[:, i].min(), y_true[:, i].max()], 
                "r--", linewidth=1, label="Perfect prediction")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Sensor {i+1}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(CONFIG["output"]["pred_vs_truth_plot"])
    plt.close()
    print(f"Saved pred vs truth plot: {CONFIG['output']['pred_vs_truth_plot']}")


def plot_feature_importance_map(model: nn.Module, epoch: int):
    """Save a feature-importance map from GRU input weights, replacing the same file each epoch."""
    if not hasattr(model, "gru"):
        return

    # GRU input-hidden weights have shape (3 * hidden_dim, input_dim)
    weight_ih = model.gru.weight_ih_l0.detach().cpu().numpy()
    importance = np.mean(np.abs(weight_ih), axis=0)
    feature_names = SETTING_COLS

    plt.figure(figsize=(12, 4), dpi=300)
    plt.bar(np.arange(len(feature_names)), importance, color="black", alpha=0.85)
    plt.xticks(np.arange(len(feature_names)), feature_names, rotation=75)
    plt.ylabel("Mean |weight_ih_l0|")
    plt.title(f"GRU Input Feature Importance (epoch {epoch})")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG["output"]["feature_importance_plot"])
    plt.close()


def print_model_parameters(model: nn.Module):
    total_params = 0
    trainable_params = 0

    print("Model parameter report:")
    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            trainable_params += numel
        print(
            f"  {name:<35} shape={tuple(param.shape)} numel={numel} trainable={param.requires_grad}"
        )

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


def run():
    _ensure_dirs()
    device = torch.device(CONFIG["training"]["device"])

    with open(CONFIG["output"]["config_json"], "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Load data
    print(f"Loading dataset FD00{CONFIG['dataset_id']}...")
    df = load_parquet(fd=CONFIG["dataset_id"], split="train")
    train_ids, test_ids = split_units(
        df,
        train_series=CONFIG["split"]["train_series"],
        test_series=CONFIG["split"]["test_series"],
        seed=CONFIG["split"]["random_seed"],
    )

    print(f"Train units: {len(train_ids)}, Test units: {len(test_ids)}")

    # Build sequences
    x_train, y_train = build_sequences(df, train_ids)
    x_test_seq, y_test_seq = build_sequences(df, test_ids)
    print(f"Train sequences: {x_train.shape}, Test sequences: {x_test_seq.shape}")

    # Standardize
    scaler_x = StandardScaler()
    x_train_scaled = scaler_x.fit_transform(x_train)
    x_test_seq_scaled = scaler_x.transform(x_test_seq)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_seq_scaled = scaler_y.transform(y_test_seq)

    # DataLoaders
    train_ds = TensorDataset(torch.tensor(x_train_scaled, dtype=torch.float32), torch.tensor(y_train_scaled, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(x_test_seq_scaled, dtype=torch.float32), torch.tensor(y_test_seq_scaled, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=CONFIG["training"]["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["training"]["batch_size"], shuffle=False)

    # Model
    input_dim = len(SETTING_COLS)
    output_dim = len(SENSOR_COLS)
    base_model = SettingsEffectRNN(
        input_dim=input_dim,
        hidden_dim=CONFIG["model"]["hidden_dim"],
        output_dim=output_dim,
        num_layers=CONFIG["model"]["num_layers"],
        dropout=CONFIG["model"]["dropout"],
        head_hidden_dims=CONFIG["model"]["head_hidden_dims"],
    ).to(device)

    print_model_parameters(base_model)
    model = torch.compile(base_model)
    print("Model compiled with torch.compile")

    optimizer = Adam(model.parameters(), lr=CONFIG["training"]["learning_rate"])

    # Training loop
    train_losses = []
    test_losses = []

    for epoch in range(CONFIG["training"]["epochs"]):
        epoch_num = epoch + 1
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_loss = eval_epoch(model, test_loader, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        plot_feature_importance_map(base_model, epoch_num)
        print(
            f"Epoch {epoch_num}/{CONFIG['training']['epochs']}: "
            f"train_loss={train_loss:.6f}, test_loss={test_loss:.6f}"
        )

    # Save metrics
    metrics_df = pd.DataFrame(
        {
            "epoch": list(range(1, CONFIG["training"]["epochs"] + 1)),
            "train_loss": train_losses,
            "test_loss": test_losses,
        }
    )
    metrics_df.to_csv(CONFIG["output"]["metrics_csv"], index=False)
    print(f"Saved metrics: {CONFIG['output']['metrics_csv']}")

    # Save model
    torch.save(base_model.state_dict(), CONFIG["output"]["model_checkpoint"])
    print(f"Saved model: {CONFIG['output']['model_checkpoint']}")

    # Plot convergence
    plot_convergence(train_losses, test_losses)

    # Save final prediction-vs-truth plot separately from feature importance.
    plot_pred_vs_truth(model, x_test_seq_scaled, y_test_seq_scaled, device)

    print("\nDone.")


if __name__ == "__main__":
    run()
