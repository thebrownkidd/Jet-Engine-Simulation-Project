# Simulator Training Script - 2025-08-10
# Author: thebrownkidd
# Training script for Unit Simulator model
import random
import gc
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gc
import torch.nn.functional as F
import os
import numpy as np
import time
import csv
from datetime import datetime
import imageio

device = torch.device("cpu") # force CPU

# ======== CONFIG DICT: ALL PARAMETERS FOR EXPERIMENT ========
config = {
    "run": {
        "root_dir": "runs_simulator",
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "metrics_filename": "metrics.csv",
        "config_filename": "config.json"
    },
    "hardware": {
        "device": str(device),
        "cuda_benchmark": False,
        "gpu": "None",
        "cuda_version": "None",
        "gpu_mem_gb": 0,
        "num_workers": 2,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2
    },
    "data": {
        "cache_path": "data_cache.pt",
        "input_files": ["Data/train_FD001in.json"],
        "output_files": ["Data/train_FD001in.json"],
        "test_size": 0.1,
        "random_state": 42,
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
        "sim_hidden_dim": 512,
        "output_dim": 15,
        "const_threshold": 0.00005
    },
    "training": {
        "epochs": 100,
        "initial_lr": 2e-4,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 1,
        "mixed_precision": False,
        "autoregressive_schedule": {
            10:50,
            30:60,
            50:70,
        },
        "warmup_steps": 20,
        "ar_steps_init": 20,
        "ar_reset_lr": 5e-3,
        "checkpoint_freq": 10,
        "best_acc_threshold": 0.80,
        "best_acc_delta": 0.01,
        "takes": [1,2,3,5,6,7,8,10,11,12,13,14,16,19,20],
        "train_noise": {
            "enabled": True,
            "initial_scale": 1,  # Start with small noise
            "final_scale": 1.5,     # Increase noise gradually
            "noise_schedule": {      # Epochs at which to increase noise
                30: 1.3,
                50: 1.5
            }
        }
    },
    "loss": {
        "mse": "MSELoss",
        "l1": "L1Loss"
    },
    "plot": {
        "nrows": 5,
        "ncols": 3,
        "figsize": (40, 30),
        "dpi": 300
    },
    "mc_eval": {
    "samples": 5,  # Reduced from 10 to 5 samples
    "noise_scale": 1,
    "tf_noise_samples": 20,  # Add this line - number of noise samples for teacher forcing
    "tf_noise_scale": 1, # Add this line - noise scale for teacher forcing 
    "frequency": 10,
    "fps": 5
    }
}

class MetricsMonitor:
    def __init__(self, filepath, flush_freq=10):
        self.filepath = filepath
        self.flush_freq = flush_freq
        self.file = None
        self.writer = None
        self.counter = 0
        self.start_time = time.time()
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'epoch', 'batch', 'train_loss', 'pinns_loss', 
                'train_acc', 'test_loss', 'test_pinns_loss', 'test_acc',
                'learning_rate', 'elapsed_seconds', 'gpu_memory_allocated_gb'
            ])
        self.file = open(filepath, 'a', newline='')
        self.writer = csv.writer(self.file)
        print(f"Metrics will be saved to {filepath}")
    def log_batch(self, epoch, batch, train_loss, pinns_loss, train_acc, lr):
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        gpu_mem = 0 # always 0 on CPU
        self.writer.writerow([
            timestamp, epoch, batch, train_loss, pinns_loss, train_acc, 
            "", "", "", lr, elapsed, gpu_mem
        ])
        self._maybe_flush()
    def log_epoch(self, epoch, train_loss, pinns_loss, train_acc, test_loss, test_pinns_loss, test_acc, lr):
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        gpu_mem = 0
        self.writer.writerow([
            timestamp, epoch, "end", train_loss, pinns_loss, train_acc, 
            test_loss, test_pinns_loss, test_acc, lr, elapsed, gpu_mem
        ])
        self.file.flush()
    def _maybe_flush(self):
        self.counter += 1
        if self.counter % self.flush_freq == 0:
            self.file.flush()
    def close(self):
        if self.file:
            self.file.close()

def plot_signal(Y, Yhat, test_acc, train_acc, config, name="SignalLstm.png", Ystd=None, noisy_samples=None):
    """
    Plot signal with ground truth, predictions, and noise/uncertainty.
    
    Args:
        Y: Ground truth values
        Yhat: Model predictions (mean values)
        test_acc: Test accuracy
        train_acc: Training accuracy
        config: Configuration dictionary
        name: Output filename
        Ystd: Standard deviation of predictions (optional)
        noisy_samples: List of noisy prediction samples (optional)
    """
    plt.ioff()
    print(f"New Plot at: {name}")
    nrows, ncols, figsize, dpi = config['plot']['nrows'], config['plot']['ncols'], config['plot']['figsize'], config['plot']['dpi']
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    axes = axes.flatten()
    
    for i in range(nrows*ncols):
        if i >= len(Y[0][0]):  # Skip if index exceeds number of features
            continue
            
        # Plot ground truth
        axes[i].plot([Y[0][k][i] for k in range(len(Y[0]))], color='black', linewidth=1.5, label='Targets')
        
        # Plot mean prediction
        axes[i].plot([Yhat[0][k][i] for k in range(len(Y[0]))], color='red', linestyle='--', linewidth=1.5, label='Mean Prediction')
        
        # Plot uncertainty bands if standard deviation is provided
        if Ystd is not None:
            upper = [Yhat[0][k][i] + 2*Ystd[0][k][i] for k in range(len(Y[0]))]
            lower = [Yhat[0][k][i] - 2*Ystd[0][k][i] for k in range(len(Y[0]))]
            axes[i].fill_between(range(len(Y[0])), lower, upper, color='red', alpha=0.2, label='Noise (±2σ)')
        
        # Plot individual noisy samples if provided (max 10 samples for clarity)
        if noisy_samples is not None:
            num_samples = min(len(noisy_samples), 10)
            for s in range(num_samples):
                if len(noisy_samples[s][0]) > 0:  # Check there's data to plot
                    try:
                        axes[i].plot([noisy_samples[s][0][k][i] for k in range(min(len(Y[0]), len(noisy_samples[s][0])))], 
                                color='blue', alpha=0.1, linewidth=0.5)
                    except IndexError:
                        # Skip if indices don't match up
                        continue
            
            # Add a single blue line to the legend to represent all samples
            axes[i].plot([], [], color='blue', alpha=0.3, linewidth=1.0, label=f'Noisy Samples ({len(noisy_samples)})')
        
        axes[i].set_title(f"Signal {i + 1}")
        axes[i].grid(alpha=0.3)
    
    # Add legend to the first subplot only
    if axes.size > 0:
        if isinstance(axes, np.ndarray):
            axes[0].legend(loc='upper left')
    
    plt.tight_layout()
    fig.suptitle(f"Test Acc: {test_acc:.4f}, Train Acc: {train_acc:.4f}", fontsize=16)
    plt.savefig(name)
    plt.close(fig)
    plt.clf()
    plt.close('all')
    gc.collect()

def plot_signal_with_uncertainty(Y, Yhat, Ystd, config, name="SignalWithUncertainty.png", all_trajectories=None):
    """
    Plot signal with uncertainty bands and individual trajectories.
    With improved visibility and debugging.
    """
    plt.ioff()
    print(f"New Plot at: {name}")
    print(f"Plotting with {len(all_trajectories) if all_trajectories else 0} trajectories")
    print(f"Y shape: {np.shape(Y)}, Yhat shape: {np.shape(Yhat)}, Ystd shape: {np.shape(Ystd)}")
    
    nrows, ncols = 3, 2  # Limit to 6 plots for better visibility
    figsize = (15, 10)  # Larger figure size
    dpi = 100
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    axes = axes.flatten()
    
    # Determine features to plot
    max_features = min(nrows * ncols, len(Y[0]) if isinstance(Y[0], list) else Y.shape[1])
    time_steps = len(Y) if isinstance(Y, list) else Y.shape[0]
    
    print(f"Plotting {max_features} features over {time_steps} timesteps")
    
    # Process each feature
    for i in range(max_features):
        # 1. Plot individual trajectories FIRST (so they're in the background)
        if all_trajectories is not None and len(all_trajectories) > 0:
            print(f"  Feature {i+1}: Plotting {len(all_trajectories)} trajectories")
            
            # Limit number of trajectories for clearer visualization
            max_to_plot = min(30, len(all_trajectories))
            for t_idx in range(max_to_plot):
                try:
                    traj = all_trajectories[t_idx]
                    
                    # Convert tensor to numpy if needed
                    if isinstance(traj, torch.Tensor):
                        traj = traj.detach().cpu().numpy()
                    
                    # Plot based on data structure
                    if isinstance(traj, np.ndarray):
                        if len(traj.shape) == 3:  # [batch, seq, features]
                            # Plot this trajectory using higher alpha for visibility
                            axes[i].plot(range(time_steps), traj[0, :time_steps, i], 
                                      color='blue', alpha=0.2, linewidth=0.5)
                        else:
                            print(f"    Skipping trajectory with shape {traj.shape}")
                    elif isinstance(traj, list) and len(traj) > 0:
                        # Handle list format
                        if isinstance(traj[0], list):
                            axes[i].plot(range(min(time_steps, len(traj[0]))), 
                                      [traj[0][t][i] for t in range(min(time_steps, len(traj[0])))],
                                      color='blue', alpha=0.2, linewidth=0.5)
                except Exception as e:
                    print(f"    Error plotting trajectory {t_idx}: {str(e)}")
                    continue
            
            # Add representative line to legend
            axes[i].plot([], [], color='blue', alpha=0.5, linewidth=1.5, 
                       label=f'MC Trajectories ({len(all_trajectories)})')
        
        # 2. Plot ground truth as solid black line
        try:
            if isinstance(Y, list):
                axes[i].plot(range(len(Y)), [Y[t][i] for t in range(len(Y))], 
                           color='black', linewidth=2.0, label='Ground Truth')
            else:
                axes[i].plot(range(Y.shape[0]), Y[:, i], 
                           color='black', linewidth=2.0, label='Ground Truth')
        except Exception as e:
            print(f"    Error plotting ground truth: {str(e)}")
        
        # 3. Plot mean prediction as dashed red line
        try:
            if isinstance(Yhat, list):
                axes[i].plot(range(len(Yhat)), [Yhat[t][i] for t in range(len(Yhat))], 
                           color='red', linestyle='--', linewidth=2.0, label='Mean Prediction')
            else:
                axes[i].plot(range(Yhat.shape[0]), Yhat[:, i], 
                           color='red', linestyle='--', linewidth=2.0, label='Mean Prediction')
        except Exception as e:
            print(f"    Error plotting mean prediction: {str(e)}")
        
        # 4. Plot uncertainty bands with higher alpha for visibility
        try:
            upper = []
            lower = []
            
            if isinstance(Yhat, list) and isinstance(Ystd, list):
                for t in range(len(Yhat)):
                    upper.append(Yhat[t][i] + 2*Ystd[t][i])
                    lower.append(Yhat[t][i] - 2*Ystd[t][i])
                axes[i].fill_between(range(len(Yhat)), lower, upper, 
                                  color='red', alpha=0.3, label='95% Confidence')
            elif isinstance(Yhat, np.ndarray) and isinstance(Ystd, np.ndarray):
                upper = Yhat[:, i] + 2*Ystd[:, i]
                lower = Yhat[:, i] - 2*Ystd[:, i]
                axes[i].fill_between(range(Yhat.shape[0]), lower, upper, 
                                  color='red', alpha=0.3, label='95% Confidence')
            
            print(f"    Added confidence bands: min={min(lower):.3f}, max={max(upper):.3f}")
        except Exception as e:
            print(f"    Error plotting confidence bands: {str(e)}")
        
        # Add grid and labels
        axes[i].set_title(f"Signal {i + 1}")
        axes[i].grid(True, alpha=0.3)
        
        # Add legend to each subplot
        axes[i].legend(loc='best', fontsize=8)
    
    # Add overall title and layout
    plt.tight_layout()
    plt.suptitle("Prediction with Uncertainty Bands and MC Trajectories", fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    # Save with higher quality
    plt.savefig(name, dpi=dpi)
    print(f"Plot saved to {name}")
    
    # Clean up
    plt.close(fig)
    plt.close('all')
    gc.collect()
    
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
    def __init__(self, input_dim=15, hidden_dim=2048, sim_hidden_dim=8112, output_dim=15):
        super(LstmRegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.sim_hidden_dim = sim_hidden_dim
        self.fc1 = nn.Linear(input_dim, sim_hidden_dim, bias=False)
        self.act1 = nn.GELU()
        self.lstm_cell = nn.LSTMCell(sim_hidden_dim, sim_hidden_dim)
        self.fc = nn.Linear(sim_hidden_dim, output_dim, bias=False)
        
        # Constants for physics constraints
        self.Mins = [518.67, 641.21, 1571.04, 1382.25, 14.62, 21.6, 549.85, 2387.9, 9021.73, 1.3, 46.85, 518.69, 2387.88, 8099.94, 8.3249, 0.03, 388, 2388, 100.0, 38.14, 22.8942]
        self.Maxes = [518.67, 644.53, 1616.91, 1441.49, 14.62, 21.61, 556.06, 2388.56, 9244.59, 1.3, 48.53, 523.38, 2388.56, 8293.72, 8.5848, 0.03, 400, 2388, 100.0, 39.43, 23.6184]
        

    def project_to_physics_constraints(self, base_pred, noisy_pred, takes):
        """
        Project noisy prediction to respect engine physics constraints.
        Fixed to avoid in-place operations that break autograd.
        
        Args:
            base_pred: Base prediction tensor
            noisy_pred: Prediction with noise added
            takes: List of feature indices to apply constraints to
            
        Returns:
            Projected tensor that satisfies physics constraints
        """
        # Create a new tensor instead of modifying in-place
        projected = noisy_pred.clone()
        batch_size, features = projected.size()
        
        # Scale factors for each feature to ensure realistic noise
        feature_sensitivity = torch.ones_like(projected) * 5.0
        
        # Critical features that need tighter constraints
        critical_features = [2, 6]  # Sensors 3 and 7
        for f in critical_features:
            if f < features:
                feature_sensitivity[:, f] = 10.0  # More sensitive = less noise allowed
        
        # Basic constraint: limit deviation from base prediction based on sensitivity
        max_deviation = torch.abs(base_pred) / feature_sensitivity
        lower_bound = base_pred - max_deviation
        upper_bound = base_pred + max_deviation
        # Use torch.clamp on the whole tensor instead of column-by-column
        projected = torch.max(torch.min(projected, upper_bound), lower_bound)
        
        # Engine-specific constraints
        # Ensure related sensors maintain appropriate ratios
        if torch.rand(1).item() < 0.5 and 4 < features and 7 < features:
            # Get the ratio between related sensors in the base prediction
            ratio_4_7 = base_pred[:, 4] / (base_pred[:, 7] + 1e-6)
            
            # Create a new projected tensor to avoid in-place operations
            new_projected = projected.clone()
            
            # Adjust to maintain similar ratio
            if torch.rand(1).item() < 0.5:
                # Adjust sensor 7 based on sensor 4
                new_projected[:, 7] = projected[:, 4] / (ratio_4_7 + 1e-6)
            else:
                # Adjust sensor 4 based on sensor 7
                new_projected[:, 4] = projected[:, 7] * ratio_4_7
                
            projected = new_projected
        
        # Final check: ensure all values are within valid ranges
        # Create a new tensor for the final result
        result = torch.zeros_like(projected)
        
        for f in range(features):
            min_val = 0.0
            max_val = float('inf')
            
            # Look up min/max for this feature
            for i, t in enumerate(takes):
                if i < len(takes) and f == i:
                    take_idx = takes[i]
                    if take_idx < len(self.Mins):
                        min_val = self.Mins[take_idx]
                    if take_idx < len(self.Maxes):
                        max_val = self.Maxes[take_idx]
            
            # Apply the limits (without in-place operations)
            result[:, f] = torch.clamp(projected[:, f], min=min_val, max=max_val)
        
        return result
    def forward(self, x, autoregressive_steps=0, noise_scale=0.0, noise_config=None):
        """
        Forward pass with option to inject noise during autoregressive steps.
        
        Args:
            x: Input tensor [batch_size, seq_len, features]
            autoregressive_steps: Number of steps to run in autoregressive mode
            noise_scale: Amount of noise to add during autoregressive steps
            noise_config: Additional configuration for noise (e.g., takes list)
            
        Returns:
            outputs: Model outputs [batch_size, seq_len, features]
        """
        batch_size, seq_len, _ = x.size()
        device = x.device

        x_processed = self.act1(self.fc1(x))

        # Initialize states
        h = torch.zeros(batch_size, self.sim_hidden_dim, device=device)
        c = torch.zeros(batch_size, self.sim_hidden_dim, device=device)

        # Avoid in-place assignment: build a list and concat at the end
        out_list = []
        transition_point = max(0, seq_len - autoregressive_steps)

        for t in range(seq_len):
            if t == 0 or t < transition_point:
                # Teacher forcing mode
                x_t = x_processed[:, t, :]
            else:
                # Autoregressive mode - potentially with noise
                prev_output = out_list[-1].squeeze(1)
                
                if noise_scale > 0.0 and noise_config is not None:
                    # Add noise
                    noise = torch.randn_like(prev_output) * noise_scale
                    noisy_output = prev_output + noise
                    
                    # Project to satisfy physics constraints
                    noisy_output = self.project_to_physics_constraints(
                        prev_output, noisy_output, 
                        takes=noise_config.get("takes", [1,2,3,5,6,7,8,10,11,12,13,14,16,19,20])
                    )
                    
                    # Use noisy output
                    x_t = self.act1(self.fc1(noisy_output))
                else:
                    # No noise
                    x_t = self.act1(self.fc1(prev_output))
                    
            h, c = self.lstm_cell(x_t, (h, c))
            next_state = self.fc(h)
            out_list.append(next_state.unsqueeze(1))
        outputs = torch.cat(out_list, dim=1)
        return outputs
    
def increasing_segments_loss(O, device):
    """
    Loss function that enforces that the mean of the last few segments of the signal
    for sensors 3 and 7 (indices 2 and 6) are in increasing order.
    """
    # Indices for sensor 3 and sensor 7 (0-indexed in the tensor)
    sensor_indices = [2, 6]
    batch_size, seq_len = O.size(0), O.size(1)
    
    # Define number of segments to check and their size
    num_segments = 4  # Check last 4 segments
    min_seq_len = 30  # Minimum sequence length required
    
    # Skip if sequence is too short
    if seq_len < min_seq_len:
        return torch.tensor(0.0, device=device)
    
    # Calculate segment size based on sequence length
    segment_size = min(20, seq_len // 5)  # Adaptive segment size
    
    total_loss = 0.0
    
    for idx in sensor_indices:
        # Extract signal for this sensor
        signal = O[:, :, idx]
        
        for b in range(batch_size):
            # Calculate means for the last segments
            segment_means = []
            
            for i in range(num_segments):
                start_idx = max(0, seq_len - (i + 1) * segment_size)
                end_idx = seq_len - i * segment_size
                
                # Calculate mean of this segment
                segment_mean = torch.mean(signal[b, start_idx:end_idx])
                segment_means.append(segment_mean)
            
            # Check if means are in increasing order (from earliest to latest)
            # We need to reverse the list since we collected from the end
            segment_means.reverse()
            
            # Calculate penalty if not increasing
            for i in range(len(segment_means) - 1):
                # If current mean is not less than next mean, penalize
                diff = segment_means[i+1] - segment_means[i]
                # Use ReLU to penalize only if diff < 0 (means are not increasing)
                penalty = torch.nn.functional.relu(-diff) * 10.0  # Scale penalty
                total_loss += penalty
    
    return total_loss / (batch_size * len(sensor_indices))

def gompertz_pinns_loss(O, device):
    # Use pooled Gompertz parameters for Sensor 3 (O[:, :, 2]) and Sensor 7 (O[:, :, 6])
    gompertz_indices = [2, 6]
    params = {
        2: dict(K=1.90642855, c=1.80001917, gamma=0.00165272),
        6: dict(K=1.85183456, c=1.80088216, gamma=0.00196087),
    }
    batch_size, seq_len = O.size(0), O.size(1)
    t = torch.arange(seq_len, device=device).float().unsqueeze(0).expand(batch_size, seq_len)
    autograd_loss = 0
    for idx in gompertz_indices:
        y = O[:,:,idx]
        # Gompertz parameters from pooled fit
        K = params[idx]["K"]
        c = params[idx]["c"]
        gamma = params[idx]["gamma"]
        # Compute dy/dt with finite differences
        dy_dt = torch.zeros_like(y)
        dy_dt[:, 1:] = (y[:, 1:] - y[:, :-1])  # Forward difference
        # Gompertz ODE rhs
        rhs = gamma * c * torch.exp(-gamma * t) * y
        residual = dy_dt - rhs
        autograd_loss += torch.mean(residual ** 2)
    return autograd_loss / len(gompertz_indices)

def optimized_pinns_lossfn(O_normm, l, flipped_info, config):
    # Unflip the signals that were flipped during loading
    O_norm = O_normm.clone()
    method = config["data"]["flipping"]["method"]
    n_start = config["data"]["flipping"]["n_start"]
    
    for batch_idx, info in enumerate(flipped_info):
        sensors = info["sensors"] if isinstance(info, dict) else info
        start_values = info.get("start_values", []) if isinstance(info, dict) else []
        
        for i, f in enumerate(sensors):
            if f < O_norm.size(2):
                if method == "mirror_start":
                    # Use the same mirroring approach as during flipping
                    start_avg = start_values[i] if i < len(start_values) else O_norm[batch_idx, :n_start, f].mean().item()
                    O_norm[batch_idx, :, f] = 2 * start_avg - O_norm[batch_idx, :, f]
                elif method == "invert":
                    # Reverse the inversion
                    O_norm[batch_idx, :, f] = 1.0 - O_norm[batch_idx, :, f]
    
    takes = config["training"]["takes"]
    warmup = config["training"]["warmup_steps"]
    lossfn = nn.L1Loss()
    Mins = [518.67, 641.21, 1571.04, 1382.25, 14.62, 21.6, 549.85, 2387.9, 9021.73, 1.3, 46.85, 518.69, 2387.88, 8099.94, 8.3249, 0.03, 388, 2388, 100.0, 38.14, 22.8942]
    Maxes = [518.67, 644.53, 1616.91, 1441.49, 14.62, 21.61, 556.06, 2388.56, 9244.59, 1.3, 48.53, 523.38, 2388.56, 8293.72, 8.5848, 0.03, 400, 2388, 100.0, 39.43, 23.6184]
    epsilon = 1e-6
    bs, seq_len, num_features = O_norm.size()
    
    O = torch.zeros((bs,seq_len, 21), device=O_norm.device, dtype=O_norm.dtype)
    O[:,:,1] = O_norm[:,:,0].clone()
    O[:,:,2] = O_norm[:,:,1].clone()
    O[:,:,3] = O_norm[:,:,2].clone()
    O[:,:,5] = O_norm[:,:,3].clone()
    O[:,:,6] = O_norm[:,:,4].clone()
    O[:,:,7] = O_norm[:,:,5].clone()
    O[:,:,8] = O_norm[:,:,6].clone()
    O[:,:,10] = O_norm[:,:,7].clone()
    O[:,:,11] = O_norm[:,:,8].clone()
    O[:,:,12] = O_norm[:,:,9].clone()
    O[:,:,13] = O_norm[:,:,10].clone()
    O[:,:,14] = O_norm[:,:,11].clone()
    O[:,:,16] = O_norm[:,:,12].clone()
    O[:,:,19] = O_norm[:,:,13].clone()
    O[:,:,20] = O_norm[:,:,14].clone()
    
    j = 0
    
    for i in range(O.size(-1)):
        if i in takes:
            unnorm = O_norm[:,:,j] * (Maxes[i] - Mins[i] + 1e-10) + Mins[i]
            j += 1
        else:
            unnorm = torch.zeros((bs,seq_len),device=O_norm.device, dtype=O_norm.dtype) + Mins[i]
        O[:,:,i] = torch.clamp(unnorm, min=Mins[i], max=Maxes[i])
    O = O[:,warmup:,:]
    l = l-warmup
    losses = []
    for j in range(O.size(0)):
        mf_in = 0.4374 * O[j,:,4] * O[j,:,7] * 0.2045/(O[j,:,0] * (1 + O[j,:,14]))
        rhs = torch.abs(mf_in - 0.453592*(O[j,:,19] + O[j,:,20]) + 0.453592*O[j,:,11]*O[j,:,10])
        V50 = torch.abs(torch.sqrt(torch.abs(2 * ((mf_in * 27214.9160 * O[j,:,0]) + (O[j,:,11]*O[j,:,10] * 21022.4824) - (rhs * 27214.9160 * O[j,:,3]) - (O[j,:,16] * (O[j,:,19] + O[j,:,20])) / rhs))))
        lhs = O[j,:,12] * O[j,:,7] * 27209.7227 * V50/(O[j,:,6] * 287)
        loss = lossfn(lhs, rhs)
        losses.append(loss)
    mass_conservation = sum(losses) / len(losses)

    # Replace Gompertz loss with increasing segments loss
    segment_loss = increasing_segments_loss(O_norm, O_norm.device)
    return torch.log(mass_conservation)*1e-3 + gompertz_pinns_loss(O_norm, O_norm.device)

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def apply_mask(y_hat, lengths):
    batch_size, seq_len, features = y_hat.size()
    mask = torch.zeros((batch_size, seq_len, 1), device=y_hat.device)
    for i, length in enumerate(lengths):
        mask[i, :length, 0] = 1
    return y_hat * mask

def create_video_from_images(image_paths, output_path, fps=5):
    """
    Creates a video from a list of image paths with memory optimization
    """
    print(f"Creating video from {len(image_paths)} images...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if we have images to work with
    if not image_paths:
        print("No images to create video from!")
        return
    
    try:
        # Memory-optimized video creation
        writer = imageio.get_writer(output_path, fps=fps, quality=7)  # Lower quality
        
        for img_path in image_paths:
            # Read image at reduced size if needed
            img = imageio.v3.imread(img_path)
            writer.append_data(img)
            
            # Optional: remove each image file after adding to video to save disk space
            # os.remove(img_path)
        
        writer.close()
        print(f"Video created successfully: {output_path}")
    
    except Exception as e:
        print(f"Error creating video with imageio: {str(e)}")
        try:
            import cv2
            # Use more memory-efficient OpenCV approach
            first_img = cv2.imread(image_paths[0])
            # Reduce size if needed
            scale = 0.5  # Scale down by 50%
            width = int(first_img.shape[1] * scale)
            height = int(first_img.shape[0] * scale)
            dim = (width, height)
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video = cv2.VideoWriter(output_path, fourcc, fps, dim)
            
            for img_path in image_paths:
                img = cv2.imread(img_path)
                if scale != 1.0:
                    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                video.write(img)
                
                # Optional: remove each image file after adding to video
                # os.remove(img_path)
            
            video.release()
            print(f"Video created with OpenCV: {output_path}")
        except Exception as e2:
            print(f"Failed to create video with OpenCV as well: {str(e2)}")
def mc_physics_rollout(model, x, lengths, flipped_info, config, n_samples=5, noise_scale=0.01, ar_steps=None, 
                      save_steps=False, save_dir=None, sample_idx=0, max_trajectories=1000):
    """
    Monte Carlo rollout with:
    - Teacher forcing phase: Multiple noisy samples for each step, averaged into a single trajectory
    - Autoregressive phase: Exponential branching with noise and intelligent trajectory management
    
    Args:
        model: The trained LSTM model
        x: Input tensor [batch_size, seq_len, features]
        lengths: Sequence lengths
        flipped_info: Information about flipped signals
        config: Configuration dictionary
        n_samples: Maximum number of new trajectories to generate from each existing trajectory
        noise_scale: Scale of the noise to add during AR phase
        ar_steps: Number of autoregressive steps (None = full sequence)
        save_steps: Whether to save visualization of each step
        save_dir: Directory to save step visualizations
        sample_idx: Which sample in the batch to visualize
        max_trajectories: Maximum number of trajectories to prevent memory overflow
        
    Returns:
        mean_preds: Mean predictions [batch_size, seq_len, features]
        std_preds: Standard deviation of predictions [batch_size, seq_len, features]
        step_images: List of image paths if save_steps=True, otherwise empty list
        noisy_samples: List of noisy prediction samples for visualization
        all_trajectories: List of all generated trajectories for visualization
    """
    device = x.device
    batch_size, seq_len, features = x.size()
    step_images = []
    
    # Create save directory if needed
    if save_steps and save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Number of autoregressive steps (default to full sequence if None)
    if ar_steps is None:
        ar_steps = min(10, seq_len)  # Limit to 10 steps by default to avoid explosion
    else:
        ar_steps = min(ar_steps, seq_len)
    
    # For teacher forcing phase, we'll use a single trajectory
    transition_point = max(0, seq_len - ar_steps)
    
    # Initialize a single state for teacher forcing
    h_initial = torch.zeros(batch_size, model.sim_hidden_dim, device=device)
    c_initial = torch.zeros(batch_size, model.sim_hidden_dim, device=device)
    
    # Get noise parameters for teacher forcing
    tf_noise_samples = config["mc_eval"].get("tf_noise_samples", 20)  # Use 20 samples by default
    tf_noise_scale = config["mc_eval"].get("tf_noise_scale", 0.008)   # Use 0.008 scale by default
    
    # Keep track of outputs for visualization and statistics
    tf_outputs = []
    
    # Also keep track of all noisy samples for visualization
    tf_noisy_outputs_collection = []
    
    # Function to save visualization of step
    def save_step_visualization(step, all_trajectories, true_values=None):
        if not save_steps or sample_idx >= batch_size:
            return None
            
        plt.ioff()
        print(f"Creating step visualization {step} with {len(all_trajectories)} trajectories")
        
        # Use smaller figure size to save memory when visualizing
        figsize = (10, 7)
        dpi = 80
        
        # Only show a subset of features for visualization
        max_features_to_plot = 6
        
        fig, axes = plt.subplots(nrows=max_features_to_plot, ncols=1, figsize=figsize, dpi=dpi)
        
        # Fix: Handle axes correctly based on dimensionality
        if max_features_to_plot == 1:
            axes = np.array([axes])  # Ensure axes is always an array
        
        # Extract true values if available
        truth_data = []
        if true_values is not None:
            for t in range(min(step+1, len(true_values))):
                truth_data.append(true_values[t][sample_idx, :max_features_to_plot].detach().cpu().numpy())
        
        # Sample trajectories only for visualization (to prevent plot overload)
        plot_trajectories = all_trajectories
        max_plot_trajectories = 100  # Maximum number to actually plot
        if len(all_trajectories) > max_plot_trajectories:
            indices = np.random.choice(len(all_trajectories), max_plot_trajectories, replace=False)
            plot_trajectories = [all_trajectories[i] for i in indices]
        
        # Calculate mean and std of ALL trajectories
        all_values = []
        for traj in all_trajectories:
            traj_values = []
            for t, output in enumerate(traj):
                if t <= step and output is not None:
                    traj_values.append(output[sample_idx, :max_features_to_plot].detach().cpu().numpy())
            all_values.append(traj_values)
        
        # Calculate mean prediction at each timestep
        mean_data = []
        std_data = []
        for t in range(step+1):
            timestep_data = []
            for traj in all_values:
                if t < len(traj):
                    timestep_data.append(traj[t])
            
            if timestep_data:
                timestep_data = np.array(timestep_data)
                mean_data.append(np.mean(timestep_data, axis=0))
                std_data.append(np.std(timestep_data, axis=0))
        
        # Plot each feature
        for i in range(min(max_features_to_plot, features)):
            # Plot sampled trajectories (with very low alpha)
            for traj in plot_trajectories:
                traj_feature_data = []
                traj_steps = []
                for t, output in enumerate(traj):
                    if t <= step and output is not None:
                        traj_feature_data.append(output[sample_idx, i].item())
                        traj_steps.append(t)
                
                if traj_feature_data:  # Only plot if we have data
                    axes[i].plot(traj_steps, traj_feature_data, color='blue', alpha=0.05, linewidth=0.3)
            
            # Plot mean prediction
            if mean_data:
                mean_feature_data = [data[i] if t < len(mean_data) else None for t, data in enumerate(mean_data)]
                valid_indices = [t for t, val in enumerate(mean_feature_data) if val is not None]
                valid_values = [mean_feature_data[t] for t in valid_indices]
                
                if valid_values:
                    axes[i].plot(valid_indices, valid_values, color='red', linestyle='--', 
                                linewidth=1.5, label='Mean Prediction')
                    
                    # Plot uncertainty bands if we have std data
                    if std_data:
                        std_feature_data = [data[i] if t < len(std_data) else None for t, data in enumerate(std_data)]
                        upper = [mean_feature_data[t] + 2*std_feature_data[t] if t < len(mean_feature_data) and 
                                mean_feature_data[t] is not None else None for t in range(step+1)]
                        lower = [mean_feature_data[t] - 2*std_feature_data[t] if t < len(mean_feature_data) and 
                                mean_feature_data[t] is not None else None for t in range(step+1)]
                        axes[i].fill_between(valid_indices, 
                                        [lower[t] for t in valid_indices], 
                                        [upper[t] for t in valid_indices], 
                                        color='red', alpha=0.2, label='95% Confidence')
            
            # Plot ground truth
            if truth_data:
                truth_feature_data = [data[i] if t < len(truth_data) else None for t, data in enumerate(truth_data)]
                valid_indices = [t for t, val in enumerate(truth_feature_data) if val is not None]
                valid_values = [truth_feature_data[t] for t in valid_indices]
                
                if valid_values:
                    axes[i].plot(valid_indices, valid_values, color='black', linewidth=1.0, label='Ground Truth')
            
            axes[i].set_title(f"Signal {i + 1}", fontsize=9)
            axes[i].tick_params(labelsize=8)
        
        if isinstance(axes, np.ndarray) and axes.size > 0:
            axes[0].legend(fontsize=8)
                
        plt.tight_layout()
        plt.suptitle(f"Monte Carlo Simulation - Step {step} - {len(all_trajectories)} trajectories", fontsize=10)
        
        # Save figure without unsupported parameters
        file_path = os.path.join(save_dir, f'step_{step:04d}.jpg')
        plt.savefig(file_path, dpi=dpi)
        plt.close(fig)
        plt.close('all')
        gc.collect()
        
        return file_path

    # ===== TEACHER FORCING PHASE WITH NOISE =====
    with torch.no_grad():
        h, c = h_initial, c_initial
        # Store true values for plotting
        true_values = [x[:, t, :] for t in range(seq_len)]
        
        # Single teacher forcing trajectory with multiple noise samples per step
        teacher_trajectory = []
        
        # Store all teacher forcing noisy samples
        all_teacher_samples = []
        
        print(f"Teacher forcing phase with {tf_noise_samples} noise samples per step")
        
        for t in range(transition_point):
            # Process input through model's first layers
            x_t = model.act1(model.fc1(x[:, t, :]))
            
            # Create multiple noisy samples for this step
            noisy_outputs = []
            
            for i in range(tf_noise_samples):
                # Clone h and c for each noise sample
                h_sample = h.clone()
                c_sample = c.clone()
                
                # Add noise to the LSTM input
                noise = torch.randn_like(x_t) * tf_noise_scale
                x_t_noisy = x_t + noise
                
                # Process through LSTM and output layer
                h_sample, c_sample = model.lstm_cell(x_t_noisy, (h_sample, c_sample))
                next_state_sample = model.fc(h_sample)
                noisy_outputs.append(next_state_sample)
                
                # Add to teacher forcing samples collection
                if i < 10:  # Store only a subset of samples for memory reasons
                    if t == 0:
                        all_teacher_samples.append([])
                    all_teacher_samples[i].append(next_state_sample)
            
            # Store all noisy outputs for visualization
            if t < 30:  # Only store a subset to save memory
                tf_noisy_outputs_collection.append(torch.stack(noisy_outputs))
            
            # Average the noisy outputs
            next_state_avg = torch.mean(torch.stack(noisy_outputs), dim=0)
            
            # Update the main h and c states with the non-noisy version
            h, c = model.lstm_cell(x_t, (h, c))
            
            # Store the averaged output
            teacher_trajectory.append(next_state_avg)
            tf_outputs.append(next_state_avg.unsqueeze(1))
            
            # Save visualization occasionally
            if save_steps and t % 10 == 0:
                img_path = save_step_visualization(t, [teacher_trajectory], true_values)
                if img_path:
                    step_images.append(img_path)
    
    # ===== AUTOREGRESSIVE PHASE WITH INTELLIGENT BRANCHING =====
    # Initialize with the last teacher-forced state
    if transition_point > 0:
        initial_state = teacher_trajectory[-1]
    else:
        # If no teacher forcing, use the first input
        initial_state = model.fc(model.lstm_cell(model.act1(model.fc1(x[:, 0, :])), (h_initial, c_initial))[0])
    
    # Start with a single trajectory
    active_trajectories = [{
        'outputs': teacher_trajectory.copy() if teacher_trajectory else [],
        'h': h.clone(),
        'c': c.clone(),
        'prev_output': initial_state
    }]
    
    # Process autoregressive steps with branching - IMPROVED VERSION
    with torch.no_grad():
        for t in range(transition_point, seq_len):
            # Aggressively clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Adjust sample count dynamically to control growth
            current_traj_count = len(active_trajectories)
            
            # Dynamically reduce samples as trajectory count grows
            # This ensures we stay within memory limits while maximizing diversity
            if current_traj_count >= max_trajectories // 2:
                dynamic_samples = 2  # Minimum branching factor
            elif current_traj_count >= max_trajectories // 3:
                dynamic_samples = 3
            else:
                dynamic_samples = min(n_samples, max_trajectories // current_traj_count)
            
            if dynamic_samples < n_samples:
                print(f"Step {t}: Reducing samples from {n_samples} to {dynamic_samples} to manage trajectory count")
            
            # If still too many trajectories, sample randomly but keep diversity
            if current_traj_count * dynamic_samples > max_trajectories:
                # Stratified sampling - ensure diversity by grouping similar trajectories
                if current_traj_count > 50:
                    target_count = max(50, max_trajectories // dynamic_samples)
                    print(f"Step {t}: Sampling {target_count} from {current_traj_count} trajectories")
                    
                    # Intelligent sampling strategy to maintain diversity
                    # Simple approach: random sampling (more sophisticated approaches possible)
                    selected_indices = torch.randperm(current_traj_count)[:target_count]
                    active_trajectories = [active_trajectories[i.item()] for i in selected_indices]
            
            new_trajectories = []
            
            # For each active trajectory, create dynamic_samples new trajectories
            for traj in active_trajectories:
                prev_output = traj['prev_output']
                h_state = traj['h']
                c_state = traj['c']
                traj_outputs = traj['outputs']
                
                # Generate dynamic_samples new trajectories
                for i in range(dynamic_samples):
                    # Add noise to each new trajectory
                    noise = torch.randn_like(prev_output) * noise_scale
                    noisy_output = prev_output + noise
                    
                    # Project to respect physics constraints
                    noisy_output = model.project_to_physics_constraints(
                        prev_output, noisy_output, 
                        takes=config["training"]["takes"]
                    )
                    
                    # Process through model
                    x_t = model.act1(model.fc1(noisy_output))
                    h_new, c_new = model.lstm_cell(x_t, (h_state.clone(), c_state.clone()))
                    next_state = model.fc(h_new)
                    
                    # Create new trajectory
                    new_traj = {
                        'outputs': traj_outputs.copy() + [next_state],
                        'h': h_new,
                        'c': c_new,
                        'prev_output': next_state
                    }
                    new_trajectories.append(new_traj)
            
            # Replace active trajectories with new ones
            active_trajectories = new_trajectories
            
            # Save visualization occasionally
            if save_steps and t % 10 == 0:
                # Extract just the outputs for visualization
                all_outputs = [traj['outputs'] for traj in active_trajectories]
                img_path = save_step_visualization(t, all_outputs, true_values)
                if img_path:
                    step_images.append(img_path)
                    
            print(f"Step {t}: {len(active_trajectories)} active trajectories")
    
    # Calculate statistics across all trajectories
    all_aligned_outputs = torch.zeros((len(active_trajectories), batch_size, seq_len, features), device=device)
    
    # Prepare a list to collect all trajectories in tensor form
    all_trajectory_tensors = []
    
    for i, traj in enumerate(active_trajectories):
        traj_outputs = traj['outputs']
        
        # Create a full sequence tensor for this trajectory
        traj_tensor = torch.zeros((batch_size, seq_len, features), device=device)
        
        # Fill in teacher forcing outputs (same for all trajectories)
        for t in range(transition_point):
            if t < len(tf_outputs):
                all_aligned_outputs[i, :, t, :] = tf_outputs[t].squeeze(1)
                traj_tensor[:, t, :] = tf_outputs[t].squeeze(1)
        
        # Fill in trajectory-specific outputs
        for t, output in enumerate(traj_outputs[transition_point:], start=transition_point):
            if t < seq_len:  # Ensure we don't go beyond sequence length
                all_aligned_outputs[i, :, t, :] = output
                traj_tensor[:, t, :] = output
        
        # Add trajectory tensor to collection (limit to save memory)
        if i < 200:  # Store at most 200 trajectories for visualization
            all_trajectory_tensors.append(traj_tensor)
    
    # Calculate mean and std across trajectories
    mean_preds = torch.mean(all_aligned_outputs, dim=0)  # [batch_size, seq_len, features]
    std_preds = torch.std(all_aligned_outputs, dim=0)    # [batch_size, seq_len, features]
    
    # Apply length masks
    mask = torch.zeros((batch_size, seq_len, 1), device=device)
    for i, length in enumerate(lengths):
        mask[i, :length, 0] = 1
    
    mean_preds = mean_preds * mask
    std_preds = std_preds * mask
    
    # Also prepare noisy samples for visualization
    noisy_samples = []
    if len(all_teacher_samples) > 0:
        # Format teacher forcing samples for visualization
        for sample_idx, sample_list in enumerate(all_teacher_samples):
            if sample_idx >= 10:  # Limit to 10 samples
                break
            # Concatenate this sample's outputs into a tensor
            sample_tensor = torch.cat([s.unsqueeze(1) for s in sample_list], dim=1)
            noisy_samples.append([sample_tensor])
    
    # Add trajectory tensors for visualization
    # Return mean, std, step images, noisy samples from teacher forcing, and all trajectories
    return mean_preds, std_preds, step_images, noisy_samples, all_trajectory_tensors

def evaluate_with_monte_carlo(model, test_loader, config, run_dir="./", epoch=0):
    """
    Evaluate model using Monte Carlo rollout and save results with visualization video.
    Updated with better memory management and visualization of noise.
    """
    model.eval()
    
    # Monte Carlo parameters from config
    n_samples = config["mc_eval"]["samples"]
    noise_scale = config["mc_eval"]["noise_scale"]
    fps = config["mc_eval"]["fps"]
    
    # Add defaults for new parameters
    if "tf_noise_samples" not in config["mc_eval"]:
        config["mc_eval"]["tf_noise_samples"] = 20
    if "tf_noise_scale" not in config["mc_eval"]:
        config["mc_eval"]["tf_noise_scale"] = 0.008
    
    # Create directory for this evaluation
    eval_dir = os.path.join(run_dir, f"mc_eval_epoch_{epoch}")
    os.makedirs(eval_dir, exist_ok=True)
    
    all_true_values = []
    all_predictions = []
    all_std_devs = []
    
    # Process only first batch to save memory
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx > 0:
            break
            
        tensors = []
        for b in batch[:-1]:
            if isinstance(b, torch.Tensor):
                tensors.append(b.to(config["hardware"]["device"]))
            else:
                tensors.append(b)
        
        x, l, y, indices = tensors
        flipped_info = batch[-1]
        
        # Set max_trajectories based on memory constraints
        max_trajectories = 1000  # Limit to prevent OOM
        
        # Generate Monte Carlo predictions with exponential growth
        save_steps = (batch_idx == 0)
        steps_dir = os.path.join(eval_dir, "steps") if save_steps else None
        
        print(f"\nStarting Monte Carlo evaluation with noise in teacher forcing ({config['mc_eval']['tf_noise_samples']} samples) and branching in AR phase")
        
        # Use shorter AR steps to prevent memory issues
        ar_steps_to_use = min(10, l[0].item() // 4)
        print(f"Using {ar_steps_to_use} autoregressive steps")
        
        # Call mc_physics_rollout with the updated function signature
        outputs = mc_physics_rollout(
            model, x, l, flipped_info, config, 
            n_samples=n_samples,
            noise_scale=noise_scale,
            ar_steps=ar_steps_to_use,
            save_steps=save_steps,
            save_dir=steps_dir,
            sample_idx=0,
            max_trajectories=max_trajectories
        )
        
        # Unpack all outputs - handle both 4 and 5 return values
        if len(outputs) == 5:
            mean_preds, std_preds, step_images, noisy_samples, all_trajectories = outputs
        else:
            mean_preds, std_preds, step_images, noisy_samples = outputs
            all_trajectories = []
        
        # Create video from step images
        if save_steps and step_images:
            video_path = os.path.join(eval_dir, f"rollout_visualization.mp4")
            create_video_from_images(step_images, video_path, fps=fps)
        
        # Save results
        all_true_values.append(y.cpu())
        all_predictions.append(mean_preds.cpu())
        all_std_devs.append(std_preds.cpu())
        
        # Create plots with noise visualization
        if batch_idx == 0:
            y_cpu = y.cpu().tolist()
            mean_cpu = mean_preds.cpu().tolist()
            std_cpu = std_preds.cpu().tolist()
            
            # Create plot showing noise samples
            plot_signal(
                y_cpu, 
                mean_cpu,
                test_acc=0.0,  # Not relevant for MC evaluation
                train_acc=0.0, # Not relevant for MC evaluation
                config=config,
                name=os.path.join(eval_dir, f"noisy_predictions.png"),
                Ystd=std_cpu,
                noisy_samples=noisy_samples
            )
            
            # Also generate the standard uncertainty plot
            plot_signal_with_uncertainty(
                y[0].cpu().numpy(), 
                mean_preds[0].cpu().numpy(),
                std_preds[0].cpu().numpy(),
                config,
                os.path.join(eval_dir, f"uncertainty_plot.png"),
                all_trajectories=all_trajectories
            )
    
    # Combine results
    all_true = torch.cat(all_true_values, dim=0)
    all_preds = torch.cat(all_predictions, dim=0)
    all_stds = torch.cat(all_std_devs, dim=0)
    
    # Calculate overall metrics
    mse = F.mse_loss(all_preds, all_true).item()
    mae = F.l1_loss(all_preds, all_true).item()
    
    # Save metrics
    results = {
        "epoch": epoch,
        "mse": mse,
        "mae": mae,
        "n_samples": n_samples,
        "noise_scale": noise_scale,
        "tf_noise_samples": config["mc_eval"].get("tf_noise_samples", 20),
        "tf_noise_scale": config["mc_eval"].get("tf_noise_scale", 0.008),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exponential_growth": True
    }
    
    with open(os.path.join(eval_dir, "mc_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return mse, mae, all_true, all_preds, all_stds

def get_current_noise_scale(epoch, config):
    """
    Determine the current noise scale based on training schedule
    
    Args:
        epoch: Current epoch
        config: Configuration dictionary
        
    Returns:
        noise_scale: Current noise scale value
    """
    noise_config = config["training"]["train_noise"]
    
    if not noise_config.get("enabled", False):
        return 0.0
    
    # Start with initial scale
    noise_scale = noise_config.get("initial_scale", 0.001)
    
    # Check if we need to update based on schedule
    schedule = noise_config.get("noise_schedule", {})
    for schedule_epoch, scale in schedule.items():
        schedule_epoch = int(schedule_epoch)
        if epoch >= schedule_epoch:
            noise_scale = scale
    
    return noise_scale

def main(config):
    torch.autograd.set_detect_anomaly(True)
    run_id = config["run"]["run_id"]
    run_dir = os.path.join(config["run"]["root_dir"], run_id)
    os.makedirs(run_dir, exist_ok=True)
    metrics_file = os.path.join(run_dir, config["run"]["metrics_filename"])
    monitor = MetricsMonitor(metrics_file)
    print(f"Training on: {config['hardware']['device']}")
    cache_path = config["data"]["cache_path"]
    if os.path.exists(cache_path) and False:  # Disable cache for now to ensure flipping is applied
        print("Loading cached data...")
        with torch.serialization.safe_globals([TimeSeriesDataset]):
            data = torch.load(cache_path, weights_only=True)
            train_dataset = data['train_dataset']
            test_dataset = data['test_dataset']
    else:
        print("Processing data from JSON...")
        input_files = config["data"]["input_files"]
        output_files = config["data"]["output_files"]
        inputs, targets = [], []
        for in_file in input_files:
            inputs += load_json(in_file)
        for out_file in output_files:
            targets += load_json(out_file)
        train_inputs, test_inputs, train_targets, test_targets = train_test_split(
            inputs, targets, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"]
        )
        train_dataset = TimeSeriesDataset(train_inputs, train_targets)
        test_dataset = TimeSeriesDataset(test_inputs, test_targets)
        torch.save({
            'train_dataset': train_dataset,
            'test_dataset': test_dataset
        }, cache_path)
        
        # Print flipping statistics
        print("Signal flipping statistics:")
        total_units = len(train_dataset.flipped_sensors)
        units_with_flips = sum(1 for unit_id, info in train_dataset.flipped_sensors.items() 
                              if (isinstance(info, dict) and len(info["sensors"]) > 0) or 
                                 (isinstance(info, list) and len(info) > 0))
        print(f"  Units with flipped signals: {units_with_flips}/{total_units} ({units_with_flips/total_units*100:.1f}%)")
        
        # Count flips per sensor
        flips_per_sensor = {}
        for unit_id, info in train_dataset.flipped_sensors.items():
            sensors = info["sensors"] if isinstance(info, dict) else info
            for s in sensors:
                flips_per_sensor[s] = flips_per_sensor.get(s, 0) + 1
        
        if flips_per_sensor:
            print("  Most commonly flipped sensors:")
            for sensor, count in sorted(flips_per_sensor.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    Sensor {sensor}: {count} units ({count/total_units*100:.1f}%)")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
        persistent_workers=config["hardware"]["persistent_workers"],
        prefetch_factor=config["hardware"]["prefetch_factor"]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
        persistent_workers=config["hardware"]["persistent_workers"]
    )

    # Initialize LSTM model
    model = LstmRegressor(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        sim_hidden_dim=config["model"]["sim_hidden_dim"],
        output_dim=config["model"]["output_dim"]
    ).to(config["hardware"]["device"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["initial_lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config["training"]["scheduler_factor"], patience=config["training"]["scheduler_patience"])
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    best_acc = 0
    best_last = 0
    EPOCHS = config["training"]["epochs"]
    AUTOREGRESSIVE_STEPS = config["training"]["ar_steps_init"]
    config_path = os.path.join(run_dir, config["run"]["config_filename"])
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print("\n========================= Starting LSTM Regressor Training =========================\n")
    
    # Log whether training with noise is enabled
    noise_enabled = config["training"]["train_noise"]["enabled"]
    if noise_enabled:
        initial_noise = config["training"]["train_noise"]["initial_scale"]
        print(f"Training with noise ENABLED - Initial scale: {initial_noise}")
    else:
        print("Training with noise DISABLED")
    
    try:
        for epoch in range(EPOCHS):
            model.train()
            epoch_train_loss = 0
            epoch_pinns_loss = 0
            epoch_train_acc = 0
            num_batches = 0
            
            # Get current noise scale for this epoch
            current_noise_scale = get_current_noise_scale(epoch, config)
            
            # Update autoregressive steps if scheduled
            if epoch in config["training"]["autoregressive_schedule"]:
                AUTOREGRESSIVE_STEPS = config["training"]["autoregressive_schedule"][epoch]
                print(f"Switching to {AUTOREGRESSIVE_STEPS} autoregressive steps")
                if epoch == 400:
                    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["ar_reset_lr"])
            
            for i, batch in enumerate(train_loader):
                # Fixed batch processing - correctly handle tensors and non-tensors
                tensors = []
                for b in batch[:-1]:
                    if isinstance(b, torch.Tensor):
                        tensors.append(b.to(config["hardware"]["device"]))
                    else:
                        tensors.append(b)
                
                x, l, y, indices = tensors
                flipped_info = batch[-1]
                
                # Forward pass with noise during training
                noise_config = {"takes": config["training"]["takes"]}
                y_hat = model(x, AUTOREGRESSIVE_STEPS, noise_scale=current_noise_scale, noise_config=noise_config)
                y_hat = apply_mask(y_hat, l)
                
                pinns_loss = optimized_pinns_lossfn(y_hat, l, flipped_info, config)
                y_hat_trimmed = y_hat[:, config["training"]["warmup_steps"]:, :]
                y_trimmed = y[:, config["training"]["warmup_steps"]:, :]
                loss_sim = mse_loss(y_hat_trimmed, y_trimmed)
                loss = loss_sim*100

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                acc = (1 - torch.abs(y_trimmed - y_hat_trimmed).mean() / y_trimmed.mean()).item()
                epoch_train_loss += loss_sim.item()
                epoch_pinns_loss += pinns_loss.item()
                epoch_train_acc += acc
                num_batches += 1
                monitor.log_batch(
                    epoch=epoch,
                    batch=i,
                    train_loss=loss_sim.item(),
                    pinns_loss=pinns_loss.item(),
                    train_acc=acc,
                    lr=optimizer.param_groups[0]['lr']
                )
                if i % 10 == 0:
                    print(f"Train : Epoch {epoch+1}/{EPOCHS} | Loss: {loss_sim.item():.4f} | PINNS: {pinns_loss.item():.4f} | Acc: {acc*100:.2f}% | Noise: {current_noise_scale:.5f}")
            
            epoch_train_loss /= num_batches
            epoch_pinns_loss /= num_batches
            epoch_train_acc /= num_batches
            
            # Evaluation phase
            model.eval()
            epoch_test_loss = 0
            epoch_test_pinns_loss = 0
            epoch_test_acc = 0
            num_test_batches = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    # Fixed batch processing for evaluation loop
                    tensors = []
                    for b in batch[:-1]:
                        if isinstance(b, torch.Tensor):
                            tensors.append(b.to(config["hardware"]["device"]))
                        else:
                            tensors.append(b)
                    
                    x, l, y, indices = tensors
                    flipped_info = batch[-1]
                    
                    # No noise during standard evaluation
                    y_hat = model(x, AUTOREGRESSIVE_STEPS, noise_scale=0.0)
                    y_hat = apply_mask(y_hat, l)
                    
                    pinns_loss = optimized_pinns_lossfn(y_hat, l, flipped_info, config)
                    loss_sim = mse_loss(y_hat, y)
                    acc = (1 - torch.abs(y - y_hat).mean() / y.mean()).item()
                    epoch_test_loss += loss_sim.item()
                    epoch_test_pinns_loss += pinns_loss.item()
                    epoch_test_acc += acc
                    num_test_batches += 1
                
                last_y = y
                last_y_hat = y_hat
                epoch_test_loss /= num_test_batches
                epoch_test_pinns_loss /= num_test_batches
                epoch_test_acc /= num_test_batches
                
                monitor.log_epoch(
                    epoch=epoch,
                    train_loss=epoch_train_loss,
                    pinns_loss=epoch_pinns_loss,
                    train_acc=epoch_train_acc,
                    test_loss=epoch_test_loss,
                    test_pinns_loss=epoch_test_pinns_loss,
                    test_acc=epoch_test_acc,
                    lr=optimizer.param_groups[0]['lr']
                )
                
                if epoch_train_acc > best_acc:
                    model_path = os.path.join(run_dir, "best_model.pth")
                    torch.save(model.state_dict(), model_path)
                    if epoch_train_acc - best_last > config["training"]["best_acc_delta"] and best_acc > config["training"]["best_acc_threshold"]:
                        y_cpu = y.cpu().tolist()
                        y_hat_cpu = y_hat.cpu().tolist()
                        plot_signal(y_cpu, y_hat_cpu, epoch_test_acc, epoch_train_acc, config, os.path.join(run_dir, "best_model.png"))
                        best_last = epoch_train_acc
                    best_acc = epoch_train_acc
                
                print(f"Test  : Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_test_loss:.4f} | PINNS: {epoch_test_pinns_loss:.4f} | Acc: {epoch_test_acc*100:.2f}% | AR Steps: {AUTOREGRESSIVE_STEPS}")
                scheduler.step(epoch_test_loss)
            
            print(f"============== Epoch {epoch} | Best Acc: {best_acc:.4f} | Noise: {current_noise_scale:.5f} | AR Steps: {AUTOREGRESSIVE_STEPS} ==============\n")
            
            if (epoch+1) % config["training"]["checkpoint_freq"] == 0:
                checkpoint_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': epoch_train_loss,
                    'test_loss': epoch_test_loss,
                    'train_acc': epoch_train_acc,
                    'test_acc': epoch_test_acc,
                    'best_acc': best_acc,
                    'autoregressive_steps': AUTOREGRESSIVE_STEPS
                }, checkpoint_path)
                y_cpu = last_y.cpu().tolist()
                y_hat_cpu = last_y_hat.cpu().tolist()
                plot_signal(y_cpu, y_hat_cpu, epoch_test_acc, epoch_train_acc, config, os.path.join(run_dir, f"plot_epoch_{epoch}.png"))
                
            # Run Monte Carlo evaluation periodically
            if epoch % config["mc_eval"]["frequency"] == 0 or epoch == EPOCHS - 1:
                print(f"\nRunning Monte Carlo evaluation with {config['mc_eval']['samples']} samples...\n")
                mse, mae, _, _, _ = evaluate_with_monte_carlo(model, test_loader, config, run_dir, epoch)
                print(f"Monte Carlo Evaluation: MSE: {mse:.6f}, MAE: {mae:.6f}")
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving final state...")
    finally:
        monitor.close()
        final_model_path = os.path.join(run_dir, "final_model.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"Training completed. Final model saved to {final_model_path}")
        
        # Final Monte Carlo evaluation
        print("\nRunning final Monte Carlo evaluation...")
        evaluate_with_monte_carlo(model, test_loader, config, run_dir, "final")

if __name__ == "__main__":
    main(config)