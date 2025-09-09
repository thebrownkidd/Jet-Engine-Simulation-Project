# Simulator Training Script - 2025-08-10
# Author: thebrownkidd
# Training script for Unit Simulator model

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gc
import torch.nn.functional as F
import torch.amp as amp
import torch.backends.cudnn as cudnn
import os
import numpy as np
import time
import csv
from datetime import datetime

cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = amp.GradScaler()

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
        "cuda_benchmark": True,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
        "gpu_mem_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2
    },
    "data": {
        "cache_path": "data_cache.pt",
        "input_files": ["Data/train_FD001in.json"],
        "output_files": ["Data/train_FD001in.json"],
        "test_size": 0.1,
        "random_state": 42,
        "batch_size": 1
    },
    "model": {
        "name": "LstmRegressor",
        "input_dim": 15,
        "hidden_dim": 512,
        "sim_hidden_dim": 128,
        "output_dim": 15,
        "const_threshold": 0.00005
    },
    "training": {
        "epochs": 1000,
        "initial_lr": 2e-5,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 1,
        "mixed_precision": True,
        "autoregressive_schedule": {
            100:60,
            200:70
        },
        "warmup_steps": 20,
        "ar_steps_init": 50,
        "ar_reset_lr": 5e-3,
        "checkpoint_freq": 50,
        "best_acc_threshold": 0.80,
        "best_acc_delta": 0.01,
        "takes": [1,2,3,5,6,7,8,10,11,12,13,14,16,19,20]
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
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        self.writer.writerow([
            timestamp, epoch, batch, train_loss, pinns_loss, train_acc, 
            "", "", "", lr, elapsed, gpu_mem
        ])
        self._maybe_flush()
    def log_epoch(self, epoch, train_loss, pinns_loss, train_acc, test_loss, test_pinns_loss, test_acc, lr):
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
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

def plot_signal(Y, Yhat, test_acc, train_acc, config, name="SignalLstm.png"):
    plt.ioff()
    print(f"New Plot at: {name}")
    nrows, ncols, figsize, dpi = config['plot']['nrows'], config['plot']['ncols'], config['plot']['figsize'], config['plot']['dpi']
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    axes = axes.flatten()
    for i in range(nrows*ncols):
        axes[i].plot([Y[0][k][i] for k in range(len(Y[0]))], color='black', label='Targets')
        axes[i].plot([Yhat[0][k][i] for k in range(len(Y[0]))], color='red', linestyle='--', label='Predictions')
        axes[i].set_title(f"Signal {i + 1}")
    plt.tight_layout()
    fig.suptitle(f"Test Acc: {test_acc:.4f}, Train Acc: {train_acc:.4f}", fontsize=16)
    plt.legend()
    plt.savefig(name)
    plt.close(fig)
    plt.clf()
    plt.close('all')
    gc.collect()

class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = [torch.tensor(seq[:-1], dtype=torch.float32) for seq in inputs]
        self.targets = [torch.tensor(seq[1:], dtype=torch.float32) for seq in targets]
        self.lengths = [len(seq) for seq in self.inputs]
        self.max_len = max(self.lengths)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.lengths[idx]

def collate_fn(batch):
    sequences, targets, lengths = zip(*batch)
    lengths = torch.tensor(lengths)
    sorted_indices = torch.argsort(lengths, descending=True)
    sequences = [sequences[i] for i in sorted_indices]
    targets = [targets[i] for i in sorted_indices]
    lengths = lengths[sorted_indices]
    padded_inputs = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0.0)
    return padded_inputs, lengths, padded_targets

class LstmRegressor(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=2048, sim_hidden_dim=8112, output_dim=15):
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

        # Preprocess all inputs at once for efficiency
        x_processed = self.act1(self.fc1(x))
        
        # Initialize states
        h = torch.zeros(batch_size, self.sim_hidden_dim, device=device)
        c = torch.zeros(batch_size, self.sim_hidden_dim, device=device)
        outputs = torch.zeros(batch_size, seq_len, self.fc.out_features, device=device)
        
        # HI related initialization
        
        # For supporting autoregressive mode similar to the original model
        transition_point = max(0, seq_len - autoregressive_steps)
        
        for t in range(seq_len):
            # Get current input
            if t == 0 or t < transition_point:
                # Teacher forcing
                x_t = x_processed[:, t, :]
            else:
                # Autoregressive - use previous output
                prev_output = outputs[:, t-1, :]
                x_t = self.act1(self.fc1(prev_output))
            
            # LSTM forward
            h, c = self.lstm_cell(x_t, (h, c))
            
            # Generate next state
            next_state = self.fc(h)
            outputs[:, t, :] = next_state
            
        
        return outputs

def optimized_pinns_lossfn(O_norm, l, config):
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
            unnorm = O_norm[:,:,j] * (Maxes[i] - Mins[i]) + Mins[i]
            j += 1
        else:
            unnorm = torch.zeros((bs,seq_len),device=O_norm.device, dtype=O_norm.dtype) + Mins[i]
        O[:,:,i] = torch.clamp(unnorm, min=Mins[i], max=Maxes[i])
    O = O[:,warmup:,:]
    l = l-warmup
    losses = []
    for j in range(O.size(0)):
        mf_in = 0.4374 * O[:,:,4] * O[:,:,7] * 0.2045/(O[:,:,0] * (1 + O[:,:,14]))
        rhs = torch.abs(mf_in - 0.453592*(O[:,:,19] + O[:,:,20]) + 0.453592*O[:,:,11]*O[:,:,10])
        V50 = torch.abs(torch.sqrt(torch.abs(2 * ((mf_in * 27214.9160 * O[:,:,0]) + (O[:,:,11]*O[:,:,10] * 21022.4824) - (rhs * 27214.9160 * O[:,:,3]) - (O[:,:,16] * (O[:,:,19] + O[:,:,20])) / rhs))))
        lhs = O[:,:,12] * O[:,:,7] * 27209.7227 * V50/(O[:,:,6] * 287)
        loss = lossfn(lhs, rhs)
        losses.append(loss)
    mass_conservation = sum(losses) / len(losses)
    return torch.log(mass_conservation)

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

def main(config):
    run_id = config["run"]["run_id"]
    run_dir = os.path.join(config["run"]["root_dir"], run_id)
    os.makedirs(run_dir, exist_ok=True)
    metrics_file = os.path.join(run_dir, config["run"]["metrics_filename"])
    monitor = MetricsMonitor(metrics_file)
    print(f"Training on: {config['hardware']['device']}")
    if torch.cuda.is_available():
        print(f"GPU: {config['hardware']['gpu']}")
        print(f"CUDA Version: {config['hardware']['cuda_version']}")
        print(f"Available GPU memory: {config['hardware']['gpu_mem_gb']:.2f} GB")
    cache_path = config["data"]["cache_path"]
    if os.path.exists(cache_path):
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
    try:
        for epoch in range(EPOCHS):
            model.train()
            epoch_train_loss = 0
            epoch_pinns_loss = 0
            epoch_train_acc = 0
            num_batches = 0
            if epoch in config["training"]["autoregressive_schedule"]:
                AUTOREGRESSIVE_STEPS = config["training"]["autoregressive_schedule"][epoch]
                print(f"Switching to {AUTOREGRESSIVE_STEPS} autoregressive steps")
                if epoch == 400:
                    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["ar_reset_lr"])
            for i, batch in enumerate(train_loader):
                x, l, y = [b.to(config["hardware"]["device"], non_blocking=True) for b in batch]
                with torch.amp.autocast('cuda'):
                    # Get both outputs and health indicators
                    y_hat = model(x, AUTOREGRESSIVE_STEPS)
                    y_hat = apply_mask(y_hat, l)
                    pinns_loss = optimized_pinns_lossfn(y_hat, l, config)
                    y_hat_trimmed = y_hat[:, config["training"]["warmup_steps"]:, :]
                    y_trimmed = y[:, config["training"]["warmup_steps"]:, :]
                    loss_sim = mse_loss(y_hat_trimmed, y_trimmed)
                    # Add HI loss (mean squared error between health indicators)
                    
                    
                    if epoch > 200:
                        loss = loss_sim*100 + pinns_loss
                    else:
                        loss = loss_sim
                
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
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
                    print(f"Train : Epoch {epoch+1}/{EPOCHS} | Loss: {loss_sim.item():.4f} | PINNS: {pinns_loss.item():.4f} | Acc: {acc*100:.2f}%")
            epoch_train_loss /= num_batches
            epoch_pinns_loss /= num_batches
            epoch_train_acc /= num_batches
            model.eval()
            epoch_test_loss = 0
            epoch_test_pinns_loss = 0
            epoch_test_acc = 0
            num_test_batches = 0
            with torch.no_grad():
                for batch in test_loader:
                    x, l, y = [b.to(config["hardware"]["device"], non_blocking=True) for b in batch]
                    with torch.amp.autocast('cuda'):
                        y_hat= model(x, AUTOREGRESSIVE_STEPS)
                        y_hat = apply_mask(y_hat, l)
                        pinns_loss = optimized_pinns_lossfn(y_hat, l, config)
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
            print(f"============== Epoch {epoch} | Best Acc: {best_acc:.4f} | AR Steps: {AUTOREGRESSIVE_STEPS} ==============\n")
            if epoch % config["training"]["checkpoint_freq"] == 0:
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
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving final state...")
    finally:
        monitor.close()
        final_model_path = os.path.join(run_dir, "final_model.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"Training completed. Final model saved to {final_model_path}")

if __name__ == "__main__":
    main(config)