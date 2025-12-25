# Jet Engine Simulator Testing Script with Power Distribution Stopping - 2025-09-14
# Author: thebrownkidd
# Signal Prediction Accuracy Test with Power Distribution Threshold

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
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

device = torch.device("cpu")  # force CPU

# ======== CONFIG - UPDATE AS NEEDED ========
config = {
    "run": {
        "root_dir": "Test_on_power_dist",
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "results_filename": "signal_accuracy.csv",
        "plot_filename": "signal_prediction.png"
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
        "autoregressive_n_list": [15, 20, 50, 65],  # multiple autoregressive starting points
        "signal_indices": [1, 4],  # Signal indices (s2, s5) to check for power dist stopping
        "signal_thresholds": {1: 0.67, 4: 1.0},  # threshold for power dist stopping (UPPER LIMIT)
        "extra_steps": 30,  # continue prediction for 30 steps past sequence end
        "plot_engine_idx": 0,  # which engine to plot
        "power_model": {
            "min_window_size": 10,  # Min window size for power model fit
            "smoothing_window": 7,  # Savgol filter window
            "polyorder": 2,  # Savgol filter poly order
            "fit_interval": 1,  # Fit power model every N steps (1 = every step)
        }
    },
    "plot": {
        "nrows": 5,
        "ncols": 3,
        "figsize": (40, 30),
        "dpi": 300
    }
}

# ----- Power Distribution Model Functions -----
def power_model(t, A, t0, T, alpha):
    """Power model for degradation modeling"""
    out = np.ones_like(t) * A
    mask = t >= t0
    denom = np.maximum(1e-8, T - t0)
    x = (t[mask] - t0) / denom
    x = np.clip(x, 0, 1)
    out[mask] = A * (1.0 - (x ** alpha))
    out[mask & (t >= T)] = 0.0
    return out

def fit_power_model(t, y_raw, config):
    """Fit power model to sensor data"""
    # Apply smoothing
    if len(y_raw) >= config["test"]["power_model"]["smoothing_window"]:
        window_length = min(config["test"]["power_model"]["smoothing_window"], len(y_raw) // 2 * 2 + 1)
        try:
            y = savgol_filter(y_raw, window_length, config["test"]["power_model"]["polyorder"])
        except:
            y = y_raw
    else:
        y = y_raw
        
    # Check if we have enough data points to fit
    if len(y) < config["test"]["power_model"]["min_window_size"]:
        return None
    
    # Normalize y to [0,1] range for better fitting
    y = np.array(y)
    y_min, y_max = np.min(y), np.max(y)
    
    if np.isclose(y_max, y_min):
        return None
        
    # Check if signal is generally increasing or decreasing
    start_avg = np.mean(y[:max(1, int(0.1*len(y)))])
    end_avg = np.mean(y[max(0, int(0.9*len(y))):])
    increasing = end_avg > start_avg
    
    if not increasing:
        # For decreasing signals, flip to fit power model
        y_scaled = (y - y_min) / (y_max - y_min)
    else:
        # For increasing signals, invert so 1.0 is the start (good health) and 0.0 is end (failure)
        y_scaled = 1.0 - (y - y_min) / (y_max - y_min)
    
    # Clip values to prevent numerical issues
    y_scaled = np.clip(y_scaled, 0.001, 0.999)
    
    # Set up initial guesses and bounds for curve fitting
    A0 = 1.0  # Normalized scale is 1.0
    t0_guess = float(t[0])  # Start at beginning
    T_guess = float(t[-1])  # End time
    
    try:
        # Fit power model with appropriate bounds
        p0 = [A0, t0_guess, T_guess, 1.0]
        bounds = ([0.5, t[0], t0_guess+1e-6, 0.1], [1.5, t[-1], t[-1]*2, 5])
        
        # Use curve_fit to find optimal parameters
        popt, _ = curve_fit(
            lambda tt, A, t0, T, alpha: power_model(tt, A, t0, T, alpha),
            t, y_scaled, p0=p0, bounds=bounds, maxfev=1000
        )
        
        # Create fitted values
        y_hat_scaled = power_model(t, *popt)
        
        # Convert back to original scale
        if not increasing:
            y_hat = y_min + y_hat_scaled * (y_max - y_min)
        else:
            y_hat = y_max - y_hat_scaled * (y_max - y_min)
            
        return {
            "params": popt,
            "y_hat": y_hat,
            "y_hat_scaled": y_hat_scaled,
            "rmse": np.sqrt(np.mean((y - y_hat)**2)),
            "increasing": increasing,
            "y_min": y_min,
            "y_max": y_max
        }
    except Exception as e:
        print(f"Curve fitting failed: {e}")
        return None

def extrapolate_power_model(fit_result, t_extended):
    """Extrapolate fitted power model to future time steps"""
    if fit_result is None:
        return None
        
    # Extract parameters and original scaling
    popt = fit_result["params"]
    y_min = fit_result["y_min"]
    y_max = fit_result["y_max"]
    increasing = fit_result["increasing"]
    
    # Generate scaled predictions
    y_hat_scaled = power_model(t_extended, *popt)
    
    # Convert back to original scale
    if not increasing:
        y_hat_extended = y_min + y_hat_scaled * (y_max - y_min)
    else:
        y_hat_extended = y_max - y_hat_scaled * (y_max - y_min)
    
    return y_hat_extended

def compute_power_model_value(power_params, t, original_range, increasing):
    """Compute power model value at specific time t"""
    A, t0, T, alpha = power_params
    y_min, y_max = original_range
    
    # Get scaled value from power model
    if isinstance(t, (list, np.ndarray)):
        t_arr = np.array(t)
        y_scaled = power_model(t_arr, A, t0, T, alpha)
    else:
        t_arr = np.array([t])
        y_scaled = power_model(t_arr, A, t0, T, alpha)[0]
    
    # Convert back to original scale
    if not increasing:
        y_pred = y_min + y_scaled * (y_max - y_min)
    else:
        y_pred = y_max - y_scaled * (y_max - y_min)
        
    return y_pred

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data[:5]  # Only load first 5 for testing

def plot_signal_with_stop(Y, Yhat_full, Yhat_ar, stop_idx, n_ar, stopping_signal, seq_end_idx, power_fits, config, name="SignalTest.png"):
    """Plot signal with both full prediction and autoregressive prediction"""
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
        
        # Mark autoregressive start
        axes[i].axvline(n_ar, color='green', linestyle=':', label='AR Start' if i == 0 else None)
        
        # Mark where original sequence ends
        axes[i].axvline(seq_end_idx, color='orange', linestyle='-.', label='Seq End' if i == 0 else None)
        
        # Mark where model stops
        if stop_idx < len(Yhat_ar):
            axes[i].axvline(stop_idx, color='purple', linestyle='-.', 
                           label=f'Stop (Signal {stopping_signal+1})' if i == 0 else None)
        
        # Plot power distribution if this is a monitored signal
        if i in config["test"]["signal_indices"]:
            # Get the last power fit before stopping
            if power_fits and i in power_fits and stop_idx in power_fits[i]:
                fit = power_fits[i][stop_idx]
                if fit:
                    # Plot fitted power model up to stopping point
                    t = np.arange(len(Yhat_ar[:stop_idx+1]))
                    power_pred = extrapolate_power_model(fit, t)
                    if power_pred is not None:
                        axes[i].plot(t, power_pred, color='green', linestyle='-.',
                                   linewidth=2, label='Power Model Fit')
                    
                    # Add threshold line for power model value
                    threshold = config["test"]["signal_thresholds"][i]
                    axes[i].axhline(threshold, color='red', linestyle='-.')
                    axes[i].text(0.02, 0.90, f"Power Threshold: {threshold}", 
                               transform=axes[i].transAxes, color='red', fontsize=8)
                    
                    # Show power model params
                    A, t0, T, alpha = fit["params"]
                    param_text = f"Power Model:\nA={A:.2f}, t0={t0:.1f}\nT={T:.1f}, α={alpha:.2f}"
                    axes[i].text(0.70, 0.90, param_text, transform=axes[i].transAxes, 
                               bbox=dict(facecolor='white', alpha=0.7), fontsize=8)
            
            axes[i].set_title(f"Signal {i + 1} (STOPPING SIGNAL)")
            axes[i].set_facecolor('#fff0f0')  # Light red background
        else:
            axes[i].set_title(f"Signal {i + 1}")
        
        # Add range info
        min_y, max_y = np.min(Y[:, i]), np.max(Y[:, i])
        min_yhat, max_yhat = np.min(Yhat_ar[:, i]), np.max(Yhat_ar[:, i])
        axes[i].text(0.02, 0.98, f"Target: [{min_y:.3f}, {max_y:.3f}]\nPred: [{min_yhat:.3f}, {max_yhat:.3f}]", 
                     transform=axes[i].transAxes, verticalalignment='top', fontsize=8)
    
    plt.tight_layout()
    fig.suptitle(f"Signal Prediction with Power Distribution Stopping (n_ar={n_ar}, stopping signal={stopping_signal+1})", 
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

def run_autoregressive_with_power_model(model, x, n_ar, signal_indices, signal_thresholds, config):
    """Run model in autoregressive mode with power model monitoring"""
    seq_len = x.shape[1]
    extra_steps = config["test"]["extra_steps"]
    max_steps = seq_len + extra_steps  # Allow predictions to extend beyond ground truth
    fit_interval = config["test"]["power_model"]["fit_interval"]
    
    # Dictionary to store power model fits for each signal at each timestep
    power_fits = {idx: {} for idx in signal_indices}
    
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
        stopping_value = 0.0  # Power model value at stopping point
        threshold_value = 0.0  # Threshold that was crossed
        reached_end = False  # Flag to track if we reached sequence end
        
        while t_ptr < max_steps and not stop:
            # Generate next prediction
            prev_output = rollout[-1].squeeze(1)
            x_t = model.act1(model.fc1(prev_output))
            h, c = model.lstm_cell(x_t, (h, c))
            next_state = model.fc(h)
            rollout.append(next_state.unsqueeze(1))
            
            # Check if we should fit power model (every fit_interval steps or at stop point)
            if t_ptr % fit_interval == 0 or t_ptr == max_steps - 1:
                # Check power model fits for monitored signals
                for idx in signal_indices:
                    # Get time and signal values up to current point
                    t_vals = np.arange(t_ptr + 1)
                    signal_vals = torch.cat(rollout, dim=1)[0, :, idx].cpu().numpy()
                    
                    # Fit power model
                    fit_result = fit_power_model(t_vals, signal_vals, config)
                    
                    # Store fit result
                    power_fits[idx][t_ptr] = fit_result
                    
                    # Check if power model crosses threshold
                    if fit_result is not None:
                        # Get current power model value
                        pwr_val = compute_power_model_value(
                            fit_result["params"], 
                            t_ptr, 
                            (fit_result["y_min"], fit_result["y_max"]),
                            fit_result["increasing"]
                        )
                        
                        # Check if power model value exceeds threshold
                        threshold = signal_thresholds[idx]
                        if isinstance(pwr_val, (list, np.ndarray)):
                            pwr_val = pwr_val[-1]
                            
                        if (fit_result["increasing"] and pwr_val >= threshold) or \
                           (not fit_result["increasing"] and pwr_val >= threshold):
                            stop = True
                            stopping_signal = idx
                            stopping_value = float(pwr_val)
                            threshold_value = threshold
                            break
            
            # Mark if we've reached the end of the original sequence
            if t_ptr == seq_len - 1:
                reached_end = True
                
            t_ptr += 1
    
    # Return the full rollout, stop index, and which signal caused stopping
    outputs = torch.cat(rollout, dim=1)
    stop_idx = len(rollout) - 1 if not stop else t_ptr
    return outputs, stop_idx, stopping_signal, stopping_value, threshold_value, reached_end, seq_len - 1, power_fits

def test_signal_accuracy_multi_n(model, test_loader, config):
    """Test signal prediction accuracy with multiple autoregressive steps"""
    n_ar_list = config["test"]["autoregressive_n_list"]
    signal_indices = config["test"]["signal_indices"]
    signal_thresholds = config["test"]["signal_thresholds"]
    plot_engine_idx = config["test"]["plot_engine_idx"]
    
    all_results = []
    os.makedirs(config["run"]["root_dir"], exist_ok=True)
    
    print("\n" + "="*80)
    print(f"RUNNING RUL PREDICTION TEST WITH POWER DISTRIBUTION THRESHOLD DETECTION")
    print("="*80)
    print(f"Signal indices to monitor: {signal_indices}")
    print(f"Power model thresholds: {signal_thresholds}")
    print(f"Autoregressive starting points: {n_ar_list}")
    print(f"Extra steps beyond sequence end: {config['test']['extra_steps']}")
    print("-"*80)
    
    for i, batch in enumerate(test_loader):
        tensors = []
        for b in batch[:-1]:
            if isinstance(b, torch.Tensor):
                tensors.append(b.to(device))
            else:
                tensors.append(b)
        x, l, y, indices = tensors
        flipped_info = batch[-1]
        
        print(f"\nTesting Engine {i}:")
        
        # Get full sequence prediction (no autoregressive)
        with torch.no_grad():
            y_hat_full = model(x)
        
        # Convert to numpy
        y_np = y[0].cpu().numpy()
        y_hat_full_np = y_hat_full[0].cpu().numpy()
        
        # Test with different autoregressive starting points
        for n_ar in n_ar_list:
            n_ar = min(n_ar, x.shape[1]-1)  # Make sure n_ar isn't larger than sequence
            
            # Run autoregressive prediction with power model monitoring
            y_hat_ar, stop_idx, stopping_signal, stopping_value, threshold_value, reached_end, seq_end_idx, power_fits = \
                run_autoregressive_with_power_model(model, x, n_ar, signal_indices, signal_thresholds, config)
            
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
            
            # RUL error calculations
            rul_error = abs(pred_rul - true_rul)
            rul_rmse = np.sqrt(np.mean((pred_rul - true_rul) ** 2))  # For single value, same as abs error
            
            # RUL accuracy (relative error)
            rul_accuracy = 1.0 - (rul_error / (true_rul + 1e-6))
            rul_accuracy = max(0.0, min(1.0, rul_accuracy))  # Clip to [0,1]
            
            # Print results to terminal
            print(f"  n_ar={n_ar}:")
            print(f"    True RUL: {true_rul}, Predicted RUL: {pred_rul}")
            print(f"    RUL Error: {rul_error}, RUL RMSE: {rul_rmse:.4f}")
            print(f"    RUL Accuracy: {rul_accuracy:.4f}")
            
            if stopping_signal >= 0:
                signal_name = f"Signal {stopping_signal+1}"
                print(f"    Stopping reason: {signal_name} power model value {stopping_value:.4f} (threshold: {threshold_value:.4f})")
                if stop_idx > seq_end_idx:
                    print(f"    Note: Threshold crossed {stop_idx - seq_end_idx} steps after original sequence end")
            elif reached_end:
                print(f"    Prediction continued {stop_idx - seq_end_idx} steps beyond original sequence end")
                print(f"    No power model threshold exceeded within extended range")
                
            print(f"    Signal RMSE: {rmse_ar:.4f}, MAE: {mae_ar:.4f}, Signal Accuracy: {acc_ar:.4f}")
            
            # Store results
            all_results.append({
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
                "stopping_value": stopping_value if stopping_signal >= 0 else None,
                "threshold_value": threshold_value if stopping_signal >= 0 else None,
                "reached_end": reached_end,
                "steps_beyond_end": max(0, stop_idx - seq_end_idx)
            })
            
            # Plot for the selected engine
            if i == plot_engine_idx:
                plot_signal_with_stop(
                    y_np, 
                    y_hat_full_np, 
                    y_hat_ar_np, 
                    stop_idx,
                    n_ar,
                    stopping_signal,
                    seq_end_idx,
                    power_fits,
                    config,
                    name=os.path.join(config["run"]["root_dir"], f"engine_{i}_n{n_ar}.png")
                )
    
    # Calculate averages per n_ar
    print("\n" + "="*80)
    print("SUMMARY BY AUTOREGRESSIVE STARTING POINT")
    print("="*80)
    
    n_ar_results = {}
    for n_ar in n_ar_list:
        n_results = [r for r in all_results if r["n_ar"] == n_ar]
        if not n_results:
            continue
            
        avg_rul_err = np.mean([r["rul_error"] for r in n_results])
        avg_rul_rmse = np.sqrt(np.mean([r["rul_error"]**2 for r in n_results]))  # Calculate RMSE across all engines
        avg_rul_acc = np.mean([r["rul_accuracy"] for r in n_results])
        avg_rmse = np.mean([r["rmse_ar"] for r in n_results])
        avg_mae = np.mean([r["mae_ar"] for r in n_results])
        avg_acc = np.mean([r["accuracy_ar"] for r in n_results])
        
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
    
    print(f"\nSaved results to {os.path.join(config['run']['root_dir'], config['run']['results_filename'])}")
    print(f"Saved summary to {os.path.join(config['run']['root_dir'], 'summary_by_n.csv')}")

def main(config):
    print(f"Testing on: {device}")
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
    test_signal_accuracy_multi_n(model, test_loader, config)

if __name__ == "__main__":
    main(config)