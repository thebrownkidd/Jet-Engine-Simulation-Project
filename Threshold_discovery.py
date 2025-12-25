# Threshold Optimization for Jet Engine RUL Prediction
# Author: thebrownkidd
# Date: 2025-09-13

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import itertools

device = torch.device("cpu")  # force CPU

# ======== CONFIG ========
config = {
    "run": {
        "root_dir": "Threshold_Optimization",
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "results_filename": "threshold_optimization.csv"
    },
    "data": {
        "input_files": ["Data/train_FD001in.json"],
        "output_files": ["Data/train_FD001in.json"],
        "batch_size": 1,
        "flipping": {
            "enabled": True,
            "n_start": 5,
            "n_end": 5,
            "method": "mirror_start"
        }
    },
    "model": {
        "input_dim": 15,
        "hidden_dim": 512,
        "sim_hidden_dim": 128,
        "output_dim": 15,
        "checkpoint_path": "runs_simulator/20250913_151757/final_model.pth"
    },
    "optimization": {
        "n_ar": 20,  # Fixed autoregressive starting point
        "signal_indices": [1, 4],  # Signal indices to check for stopping
        "threshold_ranges": {
            1: np.arange(0.70, 0.85, 0.01),  # Test thresholds from 0.70 to 0.84 for signal 1
            4: np.arange(0.95, 1.10, 0.01)   # Test thresholds from 0.95 to 1.09 for signal 4
        },
        "extra_steps": 30  # Continue prediction beyond sequence end
    }
}

# Create output directory
os.makedirs(config["run"]["root_dir"], exist_ok=True)

# === Dataset and Model Classes (same as test script) ===
def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs_raw = inputs
        self.targets_raw = targets
        self.lengths = [len(seq) for seq in inputs]
        self.max_len = max(self.lengths)
        self.indices = list(range(len(inputs)))
        
        # Flipping parameters from config
        self.n_start = config["data"]["flipping"]["n_start"]
        self.n_end = config["data"]["flipping"]["n_end"]
        self.method = config["data"]["flipping"]["method"]
        
        # Track flipped sensors for each unit
        self.flipped_sensors = self._detect_flipped_sensors()
        print(f"Initialized dataset with {len(inputs)} sequences, flipping enabled")
        
    def _detect_flipped_sensors(self):
        """Detect which sensors need flipping for each unit"""
        flipped_sensors = {}
        
        for i, unit in enumerate(self.inputs_raw):
            if len(unit) == 0:
                flipped_sensors[i] = []
                continue
                
            unit_array = np.array(unit)
            seq_len, n_features = unit_array.shape
            sensors_to_flip = []
            start_values = []
            
            for f in range(n_features):
                n_start_actual = min(self.n_start, seq_len)
                n_end_actual = min(self.n_end, seq_len)
                
                start_avg = np.mean(unit_array[:n_start_actual, f])
                end_avg = np.mean(unit_array[-n_end_actual:, f])
                
                if end_avg < start_avg:
                    sensors_to_flip.append(f)
                    start_values.append(float(start_avg))
            
            flipped_sensors[i] = {
                "sensors": sensors_to_flip,
                "start_values": start_values
            }
            
        return flipped_sensors
    
    def __len__(self):
        return len(self.inputs_raw)
    
    def __getitem__(self, idx):
        # Get raw data
        raw_input = self.inputs_raw[idx]
        raw_target = self.targets_raw[idx]
        
        # Convert to tensors
        input_tensor = torch.tensor(raw_input[:-1], dtype=torch.float32)
        target_tensor = torch.tensor(raw_target[1:], dtype=torch.float32)
        
        # Apply flipping
        flipped_info = self.flipped_sensors[idx]
        flipped_sensors = flipped_info["sensors"] if isinstance(flipped_info, dict) else flipped_info
        start_values = flipped_info.get("start_values", []) if isinstance(flipped_info, dict) else []
        
        if flipped_sensors:
            # Flip input signals
            for i, f in enumerate(flipped_sensors):
                if f < input_tensor.size(1):
                    if self.method == "mirror_start":
                        # Mirror around start value approach
                        start_avg = start_values[i] if i < len(start_values) else input_tensor[:self.n_start, f].mean().item()
                        input_tensor[:, f] = 2 * start_avg - input_tensor[:, f]
                    elif self.method == "invert":
                        # Standard inversion
                        input_tensor[:, f] = 1.0 - input_tensor[:, f]
                    
            # Flip target signals (same features need flipping)
            for i, f in enumerate(flipped_sensors):
                if f < target_tensor.size(1):
                    if self.method == "mirror_start":
                        start_avg = start_values[i] if i < len(start_values) else target_tensor[:self.n_start, f].mean().item()
                        target_tensor[:, f] = 2 * start_avg - target_tensor[:, f]
                    elif self.method == "invert":
                        target_tensor[:, f] = 1.0 - target_tensor[:, f]
        
        return input_tensor, target_tensor, self.lengths[idx], idx, flipped_info

def collate_fn(batch):
    sequences, targets, lengths, indices, flipped_info = zip(*batch)
    lengths = torch.tensor(lengths)
    sorted_indices = torch.argsort(lengths, descending=True)
    sequences = [sequences[i] for i in sorted_indices]
    targets = [targets[i] for i in sorted_indices]
    lengths = lengths[sorted_indices]
    indices = [indices[i] for i in sorted_indices]
    flipped_info = [flipped_info[i] for i in sorted_indices]
    padded_inputs = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0.0)
    return padded_inputs, lengths, padded_targets, indices, flipped_info

class LstmRegressor(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=512, sim_hidden_dim=128, output_dim=15):
        super(LstmRegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.sim_hidden_dim = sim_hidden_dim
        self.fc1 = nn.Linear(input_dim, sim_hidden_dim, bias=False)
        self.act1 = nn.GELU()
        self.lstm_cell = nn.LSTMCell(sim_hidden_dim, sim_hidden_dim)
        self.fc = nn.Linear(sim_hidden_dim, output_dim, bias=False)

    def forward(self, x, autoregressive_steps=0):
        batch_size, seq_len, _ = x.size()
        device = x.device
        x_processed = self.act1(self.fc1(x))
        h = torch.zeros(batch_size, self.sim_hidden_dim, device=device)
        c = torch.zeros(batch_size, self.sim_hidden_dim, device=device)
        out_list = []
        transition_point = max(0, seq_len - autoregressive_steps)
        for t in range(seq_len):
            if t == 0 or t < transition_point:
                x_t = x_processed[:, t, :]
            else:
                prev_output = out_list[-1].squeeze(1)
                x_t = self.act1(self.fc1(prev_output))
            h, c = self.lstm_cell(x_t, (h, c))
            next_state = self.fc(h)
            out_list.append(next_state.unsqueeze(1))
        outputs = torch.cat(out_list, dim=1)
        return outputs

def run_autoregressive(model, x, n_ar, signal_indices, signal_thresholds, extra_steps=30):
    """Run model in autoregressive mode starting from n_ar ground truth inputs"""
    seq_len = x.shape[1]
    max_steps = seq_len + extra_steps  # Allow predictions to extend beyond ground truth
    
    with torch.no_grad():
        # Run the first n_ar steps to get a state
        rollout = [x[:, t, :].unsqueeze(1) for t in range(n_ar)]
        current_input = x[:, :n_ar, :]
        h = torch.zeros(1, model.sim_hidden_dim, device=device)
        c = torch.zeros(1, model.sim_hidden_dim, device=device)
        x_processed = model.act1(model.fc1(current_input))
        
        for t in range(n_ar):
            h, c = model.lstm_cell(x_processed[:, t, :], (h, c))
            next_state = model.fc(h)
            rollout[t] = next_state.unsqueeze(1)
        
        # Continue autoregressive prediction
        t_ptr = n_ar
        stop = False
        stopping_signal = -1  # Which signal caused stopping
        stopping_value = 0.0  # What was its value
        threshold_value = 0.0  # What was the threshold
        reached_end = False  # Flag to track if we reached sequence end
        
        while t_ptr < max_steps and not stop:
            prev_output = rollout[-1].squeeze(1)
            x_t = model.act1(model.fc1(prev_output))
            h, c = model.lstm_cell(x_t, (h, c))
            next_state = model.fc(h)
            rollout.append(next_state.unsqueeze(1))
            
            # Check if we should stop based on thresholds (UPPER LIMIT)
            for idx in signal_indices:
                if next_state[0, idx] >= signal_thresholds[idx]:
                    stop = True
                    stopping_signal = idx
                    stopping_value = float(next_state[0, idx])
                    threshold_value = signal_thresholds[idx]
                    break
            
            # Mark if we've reached the end of the original sequence
            if t_ptr == seq_len - 1:
                reached_end = True
                
            t_ptr += 1
    
    # Return the full rollout, stop index, and which signal caused stopping
    outputs = torch.cat(rollout, dim=1)
    stop_idx = len(rollout) - 1
    return outputs, stop_idx, stopping_signal, stopping_value, threshold_value, reached_end, seq_len - 1

def optimize_thresholds(model, test_loader, config):
    """Test different threshold combinations to find the optimal values"""
    n_ar = config["optimization"]["n_ar"]
    signal_indices = config["optimization"]["signal_indices"]
    threshold_ranges = config["optimization"]["threshold_ranges"]
    extra_steps = config["optimization"]["extra_steps"]
    
    # Generate all threshold combinations to test
    threshold_combinations = []
    for thresholds in itertools.product(*[threshold_ranges[idx] for idx in signal_indices]):
        threshold_dict = {signal_indices[i]: thresholds[i] for i in range(len(signal_indices))}
        threshold_combinations.append(threshold_dict)
    
    print(f"Testing {len(threshold_combinations)} threshold combinations")
    print(f"Signal {signal_indices[0]} range: {min(threshold_ranges[signal_indices[0]])} to {max(threshold_ranges[signal_indices[0]])}")
    print(f"Signal {signal_indices[1]} range: {min(threshold_ranges[signal_indices[1]])} to {max(threshold_ranges[signal_indices[1]])}")
    
    # Store results for each threshold combination
    all_results = []
    
    # Count total engines for progress reporting
    total_engines = len(test_loader)
    print(f"Evaluating on {total_engines} engines")
    
    for threshold_idx, thresholds in enumerate(threshold_combinations):
        if threshold_idx % 10 == 0:
            print(f"Testing threshold combination {threshold_idx+1}/{len(threshold_combinations)}: {thresholds}")
        
        # Results for this threshold combination
        combination_results = []
        
        # Test on all engines
        for i, batch in enumerate(test_loader):
            tensors = []
            for b in batch[:-1]:
                if isinstance(b, torch.Tensor):
                    tensors.append(b.to(device))
                else:
                    tensors.append(b)
            x, l, y, indices = tensors
            
            # Get ground truth
            y_np = y[0].cpu().numpy()
            
            # Run autoregressive prediction with current thresholds
            y_hat_ar, stop_idx, stopping_signal, stopping_value, threshold_value, reached_end, seq_end_idx = run_autoregressive(
                model, x, n_ar, signal_indices, thresholds, extra_steps)
            
            # Calculate true and predicted RUL
            true_rul = y_np.shape[0] - n_ar
            pred_rul = stop_idx - n_ar + 1  # +1 because stop_idx is 0-indexed
            
            # Calculate RUL error and accuracy
            rul_error = abs(pred_rul - true_rul)
            rul_accuracy = 1.0 - (rul_error / (true_rul + 1e-6))
            rul_accuracy = max(0.0, min(1.0, rul_accuracy))  # Clip to [0,1]
            
            combination_results.append({
                "engine_idx": i,
                "true_rul": true_rul,
                "pred_rul": pred_rul,
                "rul_error": rul_error,
                "rul_accuracy": rul_accuracy,
                "stopping_signal": stopping_signal,
                "reached_end": reached_end
            })
        
        # Calculate aggregate metrics for this threshold combination
        avg_rul_error = np.mean([r["rul_error"] for r in combination_results])
        avg_rul_accuracy = np.mean([r["rul_accuracy"] for r in combination_results])
        rul_rmse = np.sqrt(np.mean([r["rul_error"]**2 for r in combination_results]))
        
        # Count engines that reached end without threshold
        no_threshold_count = sum(1 for r in combination_results if r["stopping_signal"] < 0)
        
        # Count which signals triggered stopping
        signal_counts = {}
        for signal in signal_indices:
            signal_counts[signal] = sum(1 for r in combination_results if r["stopping_signal"] == signal)
        
        result = {
            "threshold_idx": threshold_idx,
        }
        
        # Add thresholds
        for idx in signal_indices:
            result[f"threshold_{idx}"] = thresholds[idx]
        
        # Add metrics
        result.update({
            "avg_rul_error": avg_rul_error,
            "avg_rul_accuracy": avg_rul_accuracy,
            "rul_rmse": rul_rmse,
            "no_threshold_count": no_threshold_count
        })
        
        # Add signal counts
        for idx, count in signal_counts.items():
            result[f"signal_{idx}_count"] = count
        
        all_results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_path = os.path.join(config["run"]["root_dir"], config["run"]["results_filename"])
    results_df.to_csv(results_path, index=False)
    print(f"Saved threshold optimization results to {results_path}")
    
    # Find best thresholds
    best_by_accuracy = results_df.loc[results_df["avg_rul_accuracy"].idxmax()]
    best_by_error = results_df.loc[results_df["avg_rul_error"].idxmin()]
    best_by_rmse = results_df.loc[results_df["rul_rmse"].idxmin()]
    
    print("\n===== BEST THRESHOLD COMBINATIONS =====")
    print("\nBest by Accuracy:")
    for idx in signal_indices:
        print(f"  Signal {idx}: {best_by_accuracy[f'threshold_{idx}']:.4f}")
    print(f"  Avg RUL Accuracy: {best_by_accuracy['avg_rul_accuracy']:.4f}")
    print(f"  Avg RUL Error: {best_by_accuracy['avg_rul_error']:.2f}")
    print(f"  RUL RMSE: {best_by_accuracy['rul_rmse']:.4f}")
    
    print("\nBest by Error:")
    for idx in signal_indices:
        print(f"  Signal {idx}: {best_by_error[f'threshold_{idx}']:.4f}")
    print(f"  Avg RUL Error: {best_by_error['avg_rul_error']:.2f}")
    print(f"  Avg RUL Accuracy: {best_by_error['avg_rul_accuracy']:.4f}")
    print(f"  RUL RMSE: {best_by_error['rul_rmse']:.4f}")
    
    print("\nBest by RMSE:")
    for idx in signal_indices:
        print(f"  Signal {idx}: {best_by_rmse[f'threshold_{idx}']:.4f}")
    print(f"  RUL RMSE: {best_by_rmse['rul_rmse']:.4f}")
    print(f"  Avg RUL Error: {best_by_rmse['avg_rul_error']:.2f}")
    print(f"  Avg RUL Accuracy: {best_by_rmse['avg_rul_accuracy']:.4f}")
    
    # Create visualization of results
    create_heatmap(results_df, signal_indices, config["run"]["root_dir"])
    
    return results_df, best_by_accuracy, best_by_error, best_by_rmse

def create_heatmap(results_df, signal_indices, output_dir):
    """Create heatmap of RUL accuracy for different threshold combinations"""
    if len(signal_indices) != 2:
        print("Heatmap visualization requires exactly 2 signals")
        return
    
    # Prepare data for heatmap
    pivot_df = results_df.pivot(
        index=f"threshold_{signal_indices[0]}", 
        columns=f"threshold_{signal_indices[1]}", 
        values="avg_rul_accuracy"
    )
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_df, cmap="viridis", annot=False, fmt=".3f")
    plt.title(f"RUL Accuracy by Threshold Combination (n_ar={config['optimization']['n_ar']})")
    plt.xlabel(f"Signal {signal_indices[1]} Threshold")
    plt.ylabel(f"Signal {signal_indices[0]} Threshold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "threshold_heatmap.png"), dpi=300)
    
    # Create error heatmap
    pivot_error_df = results_df.pivot(
        index=f"threshold_{signal_indices[0]}", 
        columns=f"threshold_{signal_indices[1]}", 
        values="avg_rul_error"
    )
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_error_df, cmap="viridis_r", annot=False, fmt=".1f")
    plt.title(f"RUL Error by Threshold Combination (n_ar={config['optimization']['n_ar']})")
    plt.xlabel(f"Signal {signal_indices[1]} Threshold")
    plt.ylabel(f"Signal {signal_indices[0]} Threshold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "threshold_error_heatmap.png"), dpi=300)
    
    print(f"Saved threshold heatmaps to {output_dir}")

def main():
    print(f"Starting threshold optimization on: {device}")
    
    # Load data
    input_files = config["data"]["input_files"]
    output_files = config["data"]["output_files"]
    inputs, targets = [], []
    for in_file in input_files:
        inputs += load_json(in_file)
    for out_file in output_files:
        targets += load_json(out_file)
    
    # Create dataset and dataloader
    test_dataset = TimeSeriesDataset(inputs, targets)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Load model
    model = LstmRegressor(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        sim_hidden_dim=config["model"]["sim_hidden_dim"],
        output_dim=config["model"]["output_dim"]
    ).to(device)
    model.load_state_dict(torch.load(config["model"]["checkpoint_path"], map_location=device))
    model.eval()
    
    # Run optimization
    results_df, best_acc, best_err, best_rmse = optimize_thresholds(model, test_loader, config)
    
    # Create configuration using the best thresholds
    best_thresholds = {idx: best_acc[f"threshold_{idx}"] for idx in config["optimization"]["signal_indices"]}
    
    print("\n===== RECOMMENDED CONFIGURATION =====")
    print("Add this to your test script config:")
    print(f"""
    \"test\": {{
        \"autoregressive_n_list\": [{config["optimization"]["n_ar"]}],
        \"signal_indices\": {config["optimization"]["signal_indices"]},
        \"signal_thresholds\": {{{", ".join([f"{k}: {v:.4f}" for k, v in best_thresholds.items()])}}},
        \"extra_steps\": {config["optimization"]["extra_steps"]},
        \"plot_engine_idx\": 0
    }}
    """)

if __name__ == "__main__":
    main()