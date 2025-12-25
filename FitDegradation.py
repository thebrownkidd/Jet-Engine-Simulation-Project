# Jet Engine Simulator Testing Script - Enhanced with Confidence Intervals
# Author: thebrownkidd
# Signal Prediction Accuracy Test with Autoregressive Steps and Confidence Bounds

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
import multiprocessing as mp
from functools import partial
from scipy import stats
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== CONFIG - UPDATE AS NEEDED ========
config = {
    "run": {
        "root_dir": "Test_with_CI",
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "results_filename": "signal_accuracy_with_ci.csv",
        "plot_filename": "signal_prediction_with_ci.png",
        "num_processes": max(1, mp.cpu_count() - 1)  # Use all but one CPU core
    },
    "data": {
        "cache_path": "data_cache.pt",
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
        "name": "LstmRegressor",
        "input_dim": 15,
        "hidden_dim": 512,
        "sim_hidden_dim": 128,
        "output_dim": 15,
        "const_threshold": 0.00005,
        "checkpoint_path": "runs_simulator/20250913_151757/final_model.pth"
    },
    "test": {
        "autoregressive_n_list": [20, 25, 30, 35, 40, 50],  # multiple autoregressive starting points
        "signal_indices": [1, 4],  # 0 indices for signals to check for stopping
        "signal_thresholds": {1: 0.79, 4: 1.09},  # threshold for early stopping (UPPER LIMIT)
        "extra_steps": 30,  # continue prediction for 30 steps past sequence end
        "plot_engine_idx": 0,  # which engine to plot
        "ci_confidence": 0.95,  # confidence level for intervals (95%)
        "use_ci_upper_bound": True  # use upper bound of CI instead of mean prediction
    },
    "plot": {
        "nrows": 5,
        "ncols": 3,
        "figsize": (40, 30),
        "dpi": 300
    }
}

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def plot_signal_with_ci(Y, Yhat_full, Yhat_ar, ci_upper, ci_lower, stop_idx, n_ar, stopping_signal, 
                      seq_end_idx, config, name="SignalTest.png"):
    """Plot signal with both full prediction, autoregressive prediction, and confidence intervals"""
    plt.ioff()
    print(f"New Plot at: {name}")
    nrows, ncols, figsize, dpi = config['plot']['nrows'], config['plot']['ncols'], config['plot']['figsize'], config['plot']['dpi']
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    axes = axes.flatten()
    
    for i in range(min(nrows*ncols, Y.shape[1])):
        # Ground truth
        axes[i].plot(Y[:, i], color='black', label='Ground Truth')
        
        # Full model prediction (no autoregressive)
        axes[i].plot(Yhat_full[:, i], color='blue', linestyle='-', label='Full Prediction')
        
        # Autoregressive prediction - plot full length including extended prediction
        axes[i].plot(range(len(Yhat_ar[:, i])), Yhat_ar[:, i], color='red', linestyle='--', label='Autoregressive')
        
        # Confidence Intervals
        ci_length = min(len(ci_upper), len(Yhat_ar))
        axes[i].fill_between(range(ci_length), 
                           ci_lower[:ci_length, i], 
                           ci_upper[:ci_length, i], 
                           color='red', alpha=0.2, label='95% CI')
        
        # Mark autoregressive start
        axes[i].axvline(n_ar, color='green', linestyle=':', label='AR Start' if i == 0 else None)
        
        # Mark where original sequence ends
        axes[i].axvline(seq_end_idx, color='orange', linestyle='-.', label='Seq End' if i == 0 else None)
        
        # Mark where model stops
        if stop_idx < len(Yhat_ar):
            axes[i].axvline(stop_idx, color='purple', linestyle='-.', 
                           label=f'Stop (Signal {stopping_signal+1})' if i == 0 else None)
        
        # Highlight the stopping signal
        if i == stopping_signal:
            axes[i].set_title(f"Signal {i + 1} (STOPPING SIGNAL)")
            axes[i].set_facecolor('#fff0f0')  # Light red background
        else:
            axes[i].set_title(f"Signal {i + 1}")
        
        # Add range info
        min_y, max_y = np.min(Y[:, i]), np.max(Y[:, i])
        min_yhat, max_yhat = np.min(Yhat_ar[:, i]), np.max(Yhat_ar[:, i])
        min_ci, max_ci = np.min(ci_lower[:ci_length, i]), np.max(ci_upper[:ci_length, i])
        axes[i].text(0.02, 0.98, f"Target: [{min_y:.3f}, {max_y:.3f}]\n"
                               f"Pred: [{min_yhat:.3f}, {max_yhat:.3f}]\n"
                               f"CI: [{min_ci:.3f}, {max_ci:.3f}]", 
                     transform=axes[i].transAxes, verticalalignment='top', fontsize=8)
        
        # Add threshold line for stopping signals
        if i in config["test"]["signal_indices"]:
            threshold = config["test"]["signal_thresholds"][i]
            axes[i].axhline(threshold, color='red', linestyle='-.')
            axes[i].text(0.02, 0.85, f"Upper Threshold: {threshold}", transform=axes[i].transAxes, 
                        color='red', fontsize=8)
    
    plt.tight_layout()
    fig.suptitle(f"Signal Prediction with {int(config['test']['ci_confidence']*100)}% CI (n_ar={n_ar}, stopping at {stopping_signal+1})", 
                fontsize=16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.savefig(name)
    plt.close(fig)

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

def calculate_confidence_intervals(y_true, y_pred, ci_confidence=0.95):
    """
    Calculate confidence intervals based on prediction errors.
    Args:
        y_true: Ground truth data
        y_pred: Prediction data
        ci_confidence: Confidence level (default 0.95 for 95% confidence)
    Returns:
        lower_bound, upper_bound: Numpy arrays with confidence bounds
    """
    # Calculate error distribution for each sensor
    errors = y_true - y_pred
    n_sensors = y_true.shape[1]
    
    # Initialize bounds
    lower_bounds = np.zeros_like(y_pred)
    upper_bounds = np.zeros_like(y_pred)
    
    # Calculate confidence intervals for each sensor
    for s in range(n_sensors):
        # Get errors for this sensor
        sensor_errors = errors[:, s]
        
        # Calculate statistics
        mean_error = np.mean(sensor_errors)
        std_error = np.std(sensor_errors)
        
        # Calculate t-value for the given confidence interval
        t_value = stats.t.ppf((1 + ci_confidence) / 2, len(sensor_errors) - 1)
        
        # Calculate margin of error
        margin = t_value * std_error
        
        # Set bounds for this sensor
        lower_bounds[:, s] = y_pred[:, s] + mean_error - margin
        upper_bounds[:, s] = y_pred[:, s] + mean_error + margin
    
    return lower_bounds, upper_bounds

def run_autoregressive_with_ci(model, x, n_ar, signal_indices, signal_thresholds, extra_steps=30, 
                             ci_confidence=0.95, use_ci_upper_bound=True):
    """Run model in autoregressive mode with confidence intervals"""
    seq_len = x.shape[1]
    max_steps = seq_len + extra_steps  # Allow predictions to extend beyond ground truth
    
    with torch.no_grad():
        # First run the model on full input to get regular predictions
        full_pred = model(x)
        full_pred_np = full_pred[0].cpu().numpy()
        
        # Get ground truth data
        truth_np = x[0, 1:, :].cpu().numpy()
        
        # Calculate confidence intervals on the initial predictions (non-autoregressive)
        # We'll use these CI statistics throughout the autoregressive rollout
        initial_pred_np = full_pred_np[:n_ar]  # Only use the first n_ar steps for CI calculation
        initial_truth_np = truth_np[:n_ar]
        
        # Calculate errors and their statistics for each sensor
        errors = initial_truth_np - initial_pred_np
        error_means = np.mean(errors, axis=0)
        error_stds = np.std(errors, axis=0)
        
        # Get t-value for CI
        t_value = stats.t.ppf((1 + ci_confidence) / 2, n_ar - 1)
        
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
        
        # Initialize confidence intervals array
        ci_upper = np.zeros((max_steps, x.shape[2]))
        ci_lower = np.zeros((max_steps, x.shape[2]))
        
        # Set initial confidence intervals for the first n_ar steps
        for t in range(n_ar):
            pred_t = rollout[t][0].cpu().numpy()
            ci_upper[t] = pred_t + error_means + t_value * error_stds
            ci_lower[t] = pred_t + error_means - t_value * error_stds
        
        # Continue autoregressive prediction with CI propagation
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
            
            # Calculate confidence intervals for this step
            pred_t = next_state[0].cpu().numpy()
            ci_upper[t_ptr] = pred_t + error_means + t_value * error_stds
            ci_lower[t_ptr] = pred_t + error_means - t_value * error_stds
            
            # Check if we should stop based on thresholds (UPPER LIMIT)
            for idx in signal_indices:
                # Either use the upper bound of CI (more conservative) or the mean prediction
                check_value = ci_upper[t_ptr, idx] if use_ci_upper_bound else pred_t[idx]
                
                if check_value >= signal_thresholds[idx]:
                    stop = True
                    stopping_signal = idx
                    stopping_value = float(check_value)
                    threshold_value = signal_thresholds[idx]
                    break
            
            # Mark if we've reached the end of the original sequence
            if t_ptr == seq_len - 1:
                reached_end = True
                
            t_ptr += 1
    
    # Return the full rollout, stop index, and which signal caused stopping
    outputs = torch.cat(rollout, dim=1)
    stop_idx = len(rollout) - 1
    return outputs, stop_idx, stopping_signal, stopping_value, threshold_value, reached_end, seq_len - 1, ci_upper, ci_lower

def process_engine(i, batch, model, config):
    """Process a single engine (for parallel processing)"""
    start_time = time.time()
    
    tensors = []
    for b in batch[:-1]:
        if isinstance(b, torch.Tensor):
            tensors.append(b.to(device))
        else:
            tensors.append(b)
    x, l, y, indices = tensors
    flipped_info = batch[-1]
    
    # Get full sequence prediction (no autoregressive)
    with torch.no_grad():
        y_hat_full = model(x)
    
    # Convert to numpy
    y_np = y[0].cpu().numpy()
    y_hat_full_np = y_hat_full[0].cpu().numpy()
    
    engine_results = []
    
    # Test with different autoregressive starting points
    for n_ar in config["test"]["autoregressive_n_list"]:
        n_ar = min(n_ar, x.shape[1]-1)  # Make sure n_ar isn't larger than sequence
        
        # Run autoregressive prediction with confidence intervals
        y_hat_ar, stop_idx, stopping_signal, stopping_value, threshold_value, reached_end, seq_end_idx, ci_upper, ci_lower = \
            run_autoregressive_with_ci(
                model, x, n_ar, 
                config["test"]["signal_indices"], 
                config["test"]["signal_thresholds"],
                config["test"]["extra_steps"],
                config["test"]["ci_confidence"],
                config["test"]["use_ci_upper_bound"]
            )
        y_hat_ar_np = y_hat_ar[0].cpu().numpy()
        
        # Calculate true and predicted RUL
        true_rul = y_np.shape[0] - n_ar
        pred_rul = stop_idx - n_ar + 1  # +1 because stop_idx is 0-indexed
        
        # Calculate accuracy metrics for autoregressive part (only for overlap with ground truth)
        ar_slice_len = min(y_hat_ar_np.shape[0], y_np.shape[0])
        ar_y = y_np[:ar_slice_len]
        ar_y_hat = y_hat_ar_np[:ar_slice_len]
        
        rmse_ar = np.sqrt(np.mean((ar_y_hat - ar_y) ** 2))
        mae_ar = np.mean(np.abs(ar_y_hat - ar_y))
        acc_ar = 1.0 - (np.abs(ar_y_hat - ar_y).mean(axis=0) / np.maximum(ar_y.mean(axis=0), 1e-6)).mean()
        
        # Calculate CI width statistics (as percentage of prediction)
        ci_width = ci_upper[:ar_slice_len] - ci_lower[:ar_slice_len]
        mean_ci_width = np.mean(ci_width)
        rel_ci_width = np.mean(ci_width / np.maximum(np.abs(ar_y_hat), 1e-6)) * 100  # as percentage
        
        # RUL error calculations
        rul_error = abs(pred_rul - true_rul)
        rul_rmse = np.sqrt(np.mean((pred_rul - true_rul) ** 2))  # For single value, same as abs error
        
        # RUL accuracy (relative error)
        rul_accuracy = 1.0 - (rul_error / (true_rul + 1e-6))
        rul_accuracy = max(0.0, min(1.0, rul_accuracy))  # Clip to [0,1]
        
        # Store results for this n_ar
        engine_results.append({
            "engine_idx": i,
            "n_ar": n_ar,
            "true_rul": true_rul,
            "pred_rul": pred_rul,
            "rul_error": rul_error,
            "rul_rmse": rul_rmse,
            "rul_accuracy": rul_accuracy,
            "rmse_ar": rmse_ar,
            "mae_ar": mae_ar,
            "accuracy_ar": acc_ar,
            "stop_idx": stop_idx,
            "stopping_signal": stopping_signal,
            "stopping_value": stopping_value,
            "threshold_value": threshold_value,
            "reached_end": reached_end,
            "steps_beyond_end": max(0, stop_idx - seq_end_idx),
            "mean_ci_width": mean_ci_width,
            "rel_ci_width": rel_ci_width,
            "plot_data": {
                "y_np": y_np,
                "y_hat_full_np": y_hat_full_np,
                "y_hat_ar_np": y_hat_ar_np,
                "ci_upper": ci_upper,
                "ci_lower": ci_lower
            } if i == config["test"]["plot_engine_idx"] else None
        })
    
    elapsed = time.time() - start_time
    print(f"Engine {i} processed in {elapsed:.2f}s")
    return engine_results

def test_signal_accuracy_parallel(model, test_loader, config):
    """Test signal prediction accuracy with multiple autoregressive steps in parallel"""
    print("\n" + "="*80)
    print(f"RUNNING RUL PREDICTION TEST WITH CI-BASED THRESHOLD DETECTION")
    print("="*80)
    print(f"Signal indices to check: {config['test']['signal_indices']}")
    print(f"Upper limit thresholds: {config['test']['signal_thresholds']}")
    print(f"Autoregressive starting points: {config['test']['autoregressive_n_list']}")
    print(f"Extra steps beyond sequence end: {config['test']['extra_steps']}")
    print(f"Confidence interval level: {config['test']['ci_confidence']*100}%")
    print(f"Using {'upper bound of CI' if config['test']['use_ci_upper_bound'] else 'mean prediction'} for threshold check")
    print(f"Parallelizing with {config['run']['num_processes']} processes")
    print("-"*80)
    
    # Create output directory
    os.makedirs(config["run"]["root_dir"], exist_ok=True)
    
    # Collect all batches
    all_batches = list(test_loader)
    num_engines = len(all_batches)
    
    # Create a partial function with fixed model and config
    process_func = partial(process_engine, model=model, config=config)
    
    # Process engines in parallel
    all_results = []
    if config["run"]["num_processes"] > 1:
        with mp.Pool(config["run"]["num_processes"]) as pool:
            indices_batches = [(i, batch) for i, batch in enumerate(all_batches)]
            results = pool.starmap(process_func, indices_batches)
            for r in results:
                all_results.extend(r)
    else:
        # Sequential processing
        for i, batch in enumerate(all_batches):
            all_results.extend(process_func(i, batch))
    
    # Generate plots for the selected engine
    plot_engine_idx = config["test"]["plot_engine_idx"]
    plot_results = [r for r in all_results if r["engine_idx"] == plot_engine_idx and r["plot_data"] is not None]
    
    for result in plot_results:
        n_ar = result["n_ar"]
        plot_data = result["plot_data"]
        
        plot_signal_with_ci(
            plot_data["y_np"], 
            plot_data["y_hat_full_np"], 
            plot_data["y_hat_ar_np"],
            plot_data["ci_upper"],
            plot_data["ci_lower"],
            result["stop_idx"],
            n_ar,
            result["stopping_signal"],
            result["true_rul"] + n_ar - 1,  # seq_end_idx
            config,
            name=os.path.join(config["run"]["root_dir"], f"engine_{plot_engine_idx}_n{n_ar}_with_ci.png")
        )
    
    # Calculate averages per n_ar
    print("\n" + "="*80)
    print("SUMMARY BY AUTOREGRESSIVE STARTING POINT")
    print("="*80)
    
    n_ar_list = config["test"]["autoregressive_n_list"]
    n_ar_results = {}
    for n_ar in n_ar_list:
        n_results = [r for r in all_results if r["n_ar"] == n_ar]
        if not n_results:
            continue
            
        avg_rul_err = np.mean([r["rul_error"] for r in n_results])
        avg_rul_rmse = np.sqrt(np.mean([r["rul_error"]**2 for r in n_results]))
        avg_rul_acc = np.mean([r["rul_accuracy"] for r in n_results])
        avg_rmse = np.mean([r["rmse_ar"] for r in n_results])
        avg_mae = np.mean([r["mae_ar"] for r in n_results])
        avg_acc = np.mean([r["accuracy_ar"] for r in n_results])
        avg_ci_width = np.mean([r["mean_ci_width"] for r in n_results])
        avg_rel_ci_width = np.mean([r["rel_ci_width"] for r in n_results])
        
        # Count which signals triggered stopping
        stopping_signals = [r["stopping_signal"] for r in n_results if r["stopping_signal"] >= 0]
        signal_counts = {}
        for s in stopping_signals:
            signal_counts[s] = signal_counts.get(s, 0) + 1
            
        # Count how many went beyond sequence end
        extended_count = sum(1 for r in n_results if r["reached_end"])
        extended_percent = (extended_count / len(n_results)) * 100
        
        n_ar_results[n_ar] = {
            "avg_rul_error": avg_rul_err,
            "avg_rul_rmse": avg_rul_rmse,
            "avg_rul_accuracy": avg_rul_acc,
            "avg_rmse": avg_rmse,
            "avg_mae": avg_mae,
            "avg_accuracy": avg_acc,
            "avg_ci_width": avg_ci_width,
            "avg_rel_ci_width": avg_rel_ci_width,
            "signal_counts": signal_counts,
            "extended_count": extended_count,
            "extended_percent": extended_percent
        }
        
        print(f"\n=== Results for n_ar={n_ar} ===")
        print(f"Avg RUL Error: {avg_rul_err:.2f}")
        print(f"Avg RUL RMSE: {avg_rul_rmse:.4f}")
        print(f"Avg RUL Accuracy: {avg_rul_acc:.4f}")
        print(f"Avg Signal RMSE: {avg_rmse:.4f}")
        print(f"Avg Signal MAE: {avg_mae:.4f}")
        print(f"Avg Signal Accuracy: {avg_acc:.4f}")
        print(f"Avg CI Width: {avg_ci_width:.4f} (absolute), {avg_rel_ci_width:.2f}% (relative)")
        print(f"Engines extending beyond sequence end: {extended_count} ({extended_percent:.1f}%)")
        print("Stopping signal distribution:")
        
        total = sum(signal_counts.values())
        for signal, count in sorted(signal_counts.items()):
            percent = (count / len(n_results)) * 100
            print(f"  Signal {signal+1}: {count} engines ({percent:.1f}%)")
            
        no_threshold_count = len(n_results) - total
        if no_threshold_count > 0:
            no_threshold_percent = (no_threshold_count / len(n_results)) * 100
            print(f"  No threshold exceeded: {no_threshold_count} engines ({no_threshold_percent:.1f}%)")
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    # Remove plot data column to avoid large file
    if "plot_data" in results_df.columns:
        results_df = results_df.drop(columns=["plot_data"])
    results_df.to_csv(os.path.join(config["run"]["root_dir"], config["run"]["results_filename"]), index=False)
    
    # Save summary by n_ar
    summary_data = []
    for n_ar, metrics in n_ar_results.items():
        row = {"n_ar": n_ar}
        metrics_copy = metrics.copy()
        signal_counts = metrics_copy.pop("signal_counts")
        row.update(metrics_copy)
        
        # Add signal distribution
        for signal, count in signal_counts.items():
            row[f"signal_{signal+1}_count"] = count
            
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(config["run"]["root_dir"], "summary_by_n.csv"), index=False)
    
    # Save thresholds analysis
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS")
    print("="*80)
    
    # For each signal, find ideal threshold based on various metrics
    signals_to_analyze = config["test"]["signal_indices"]
    threshold_analysis = {}
    
    for signal_idx in signals_to_analyze:
        print(f"\nAnalyzing Signal {signal_idx+1}:")
        
        # Get all unique values observed for this signal
        all_values = []
        for r in all_results:
            if r["stopping_signal"] == signal_idx:
                all_values.append(r["stopping_value"])
        
        # If no values, skip this signal
        if not all_values:
            print(f"  No engines stopped on signal {signal_idx+1}")
            continue
            
        # Sort values and print distribution
        all_values = sorted(all_values)
        q1, median, q3 = np.percentile(all_values, [25, 50, 75])
        min_val, max_val = min(all_values), max(all_values)
        
        print(f"  Value distribution: Min={min_val:.4f}, Q1={q1:.4f}, Median={median:.4f}, Q3={q3:.4f}, Max={max_val:.4f}")
        print(f"  Current threshold: {config['test']['signal_thresholds'][signal_idx]:.4f}")
        
        # Test various thresholds
        test_thresholds = np.linspace(min_val * 0.9, max_val * 1.1, 20)
        threshold_results = []
        
        for threshold in test_thresholds:
            # Count engines that would stop with this threshold
            stopped_engines = sum(1 for r in all_results 
                                if r["stopping_signal"] == signal_idx and r["stopping_value"] >= threshold)
            
            # Find engines where this signal's CI upper bound reached this threshold
            matched_results = [r for r in all_results 
                              if r["stopping_signal"] == signal_idx and r["stopping_value"] >= threshold]
            
            if matched_results:
                avg_rul_error = np.mean([r["rul_error"] for r in matched_results])
                avg_rul_rmse = np.sqrt(np.mean([r["rul_error"]**2 for r in matched_results]))
            else:
                avg_rul_error = np.nan
                avg_rul_rmse = np.nan
            
            threshold_results.append({
                "threshold": threshold,
                "stopped_engines": stopped_engines,
                "percent_stopped": stopped_engines / len(all_results) * 100,
                "avg_rul_error": avg_rul_error,
                "avg_rul_rmse": avg_rul_rmse
            })
        
        # Find optimal threshold
        valid_results = [r for r in threshold_results if not np.isnan(r["avg_rul_error"])]
        if valid_results:
            best_threshold = min(valid_results, key=lambda x: x["avg_rul_error"])
            
            print(f"  Recommended threshold: {best_threshold['threshold']:.4f}")
            print(f"    - Would stop {best_threshold['stopped_engines']} engines ({best_threshold['percent_stopped']:.1f}%)")
            print(f"    - Average RUL error: {best_threshold['avg_rul_error']:.2f}")
            print(f"    - Average RUL RMSE: {best_threshold['avg_rul_rmse']:.4f}")
            
            # Create plotting data
            threshold_df = pd.DataFrame(threshold_results)
            threshold_analysis[signal_idx] = {
                "current": config['test']['signal_thresholds'][signal_idx],
                "recommended": best_threshold['threshold'],
                "data": threshold_df
            }
            
            # Plot threshold analysis
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            ax1.plot(threshold_df["threshold"], threshold_df["avg_rul_error"], 'b-o')
            ax1.set_xlabel("Threshold Value")
            ax1.set_ylabel("Average RUL Error")
            ax1.set_title(f"Signal {signal_idx+1}: RUL Error vs Threshold")
            ax1.axvline(best_threshold['threshold'], color='g', linestyle='--', label='Recommended')
            ax1.axvline(config['test']['signal_thresholds'][signal_idx], color='r', linestyle=':', label='Current')
            ax1.legend()
            
            ax2.plot(threshold_df["threshold"], threshold_df["percent_stopped"], 'r-o')
            ax2.set_xlabel("Threshold Value")
            ax2.set_ylabel("% of Engines Stopped")
            ax2.set_title(f"Signal {signal_idx+1}: % Engines Stopped vs Threshold")
            ax2.axvline(best_threshold['threshold'], color='g', linestyle='--', label='Recommended')
            ax2.axvline(config['test']['signal_thresholds'][signal_idx], color='r', linestyle=':', label='Current')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(config["run"]["root_dir"], f"threshold_analysis_signal{signal_idx+1}.png"))
            plt.close()
        else:
            print("  Unable to determine optimal threshold (insufficient data)")
    
    # Create threshold recommendations file
    with open(os.path.join(config["run"]["root_dir"], "threshold_recommendations.txt"), "w") as f:
        f.write("THRESHOLD RECOMMENDATIONS\n")
        f.write("=======================\n\n")
        
        for signal_idx in threshold_analysis:
            analysis = threshold_analysis[signal_idx]
            f.write(f"Signal {signal_idx+1}:\n")
            f.write(f"  Current threshold: {analysis['current']:.6f}\n")
            f.write(f"  Recommended threshold: {analysis['recommended']:.6f}\n")
            f.write(f"  Change: {(analysis['recommended'] - analysis['current']) / analysis['current'] * 100:.2f}%\n\n")
    
    print(f"\nSaved results to {os.path.join(config['run']['root_dir'], config['run']['results_filename'])}")
    print(f"Saved summary to {os.path.join(config['run']['root_dir'], 'summary_by_n.csv')}")
    print(f"Saved threshold recommendations to {os.path.join(config['run']['root_dir'], 'threshold_recommendations.txt')}")
    
    return threshold_analysis

def main(config):
    print(f"Testing on: {device}")
    start_time = time.time()
    
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
    
    # Test signal accuracy with multiple autoregressive starting points
    threshold_analysis = test_signal_accuracy_parallel(model, test_loader, config)
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f}s")
    
    # Print final recommendations
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    
    for signal_idx, analysis in threshold_analysis.items():
        print(f"Signal {signal_idx+1}: Set threshold to {analysis['recommended']:.6f} " +
              f"(was {analysis['current']:.6f}, {'+' if analysis['recommended'] > analysis['current'] else ''}" +
              f"{(analysis['recommended'] - analysis['current']) / analysis['current'] * 100:.2f}%)")

if __name__ == "__main__":
    main(config)