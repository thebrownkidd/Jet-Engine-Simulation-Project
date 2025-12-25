import numpy as np
import torch
import json
import os

class SignalFlipper:
    """
    Utility class for handling the flipping of decreasing signals during preprocessing
    and unflipping them during physics-based calculations.
    
    This class maintains a mapping between engine units and their flipped sensors,
    allowing for per-engine customization of signal processing.
    """
    
    def __init__(self, n_start=5, n_end=5, method="mirror_start", metadata_file="flipped_sensors_metadata.json"):
        """
        Initialize the SignalFlipper.
        
        Args:
            n_start (int): Number of timesteps to use for start average
            n_end (int): Number of timesteps to use for end average
            method (str): Flipping method ('mirror_start' or 'invert')
            metadata_file (str): File to save/load flipped sensors metadata
        """
        self.n_start = n_start
        self.n_end = n_end
        self.method = method
        self.flipped_sensors_by_unit = {}
        self.metadata_file = metadata_file
    
    def flip_data(self, data, save_metadata=True):
        """
        Flip decreasing signals in the data and optionally save metadata.
        
        Args:
            data (list): List of units, where each unit is a list of timesteps
            save_metadata (bool): Whether to save flipped indices to file
            
        Returns:
            list: Flipped data in the same format
        """
        result = []
        
        for unit_idx, unit in enumerate(data):
            if len(unit) == 0:
                result.append([])
                self.flipped_sensors_by_unit[str(unit_idx)] = []
                continue
                
            unit_array = np.array(unit)
            seq_len, n_features = unit_array.shape
            unit_result = unit_array.copy()
            
            flipped_sensors = []
            start_values = []
            
            for f in range(n_features):
                n_start_actual = min(self.n_start, seq_len)
                n_end_actual = min(self.n_end, seq_len)
                
                start_avg = np.mean(unit_array[:n_start_actual, f])
                end_avg = np.mean(unit_array[-n_end_actual:, f])
                
                if end_avg < start_avg:
                    flipped_sensors.append(f)
                    start_values.append(float(start_avg))  # Convert to regular float for JSON
                    
                    if self.method == "mirror_start":
                        # Mirror around start value: 2*start - x
                        unit_result[:, f] = 2 * start_avg - unit_array[:, f]
                    elif self.method == "invert":
                        # Standard inversion: 1 - x
                        unit_result[:, f] = 1.0 - unit_array[:, f]
            
            result.append(unit_result.tolist())
            # Store both flipped sensors and their start values
            self.flipped_sensors_by_unit[str(unit_idx)] = {
                "sensors": flipped_sensors,
                "start_values": start_values
            }
        
        if save_metadata:
            self.save_metadata()
        
        return result
    
    def unflip_tensor(self, tensor, unit_indices):
        """
        Unflip specific sensors in a tensor based on stored metadata.
        
        Args:
            tensor (torch.Tensor): Tensor of shape [batch_size, seq_len, features]
            unit_indices (list): List of unit indices corresponding to each batch item
            
        Returns:
            torch.Tensor: Tensor with appropriate sensors unflipped
        """
        result = tensor.clone()
        
        for batch_idx, unit_idx in enumerate(unit_indices):
            # Get flipped sensors for this unit
            unit_key = str(unit_idx)
            if unit_key not in self.flipped_sensors_by_unit:
                continue
                
            flipped_data = self.flipped_sensors_by_unit[unit_key]
            
            # Handle both old and new format
            if isinstance(flipped_data, list):
                flipped_sensors = flipped_data
                has_start_values = False
            else:
                flipped_sensors = flipped_data["sensors"]
                start_values = flipped_data["start_values"]
                has_start_values = True
            
            for i, f in enumerate(flipped_sensors):
                if f >= tensor.size(2):
                    continue
                    
                if self.method == "mirror_start":
                    if has_start_values:
                        # Use stored start value if available
                        start_avg = start_values[i]
                    else:
                        # Calculate start value from current tensor
                        start_avg = tensor[batch_idx, :self.n_start, f].mean().item()
                    
                    # Apply reverse mirroring: 2*start - x
                    result[batch_idx, :, f] = 2 * start_avg - tensor[batch_idx, :, f]
                elif self.method == "invert":
                    # Apply reverse inversion: 1 - x
                    result[batch_idx, :, f] = 1.0 - tensor[batch_idx, :, f]
        
        return result
    
    def save_metadata(self):
        """Save flipped sensors metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.flipped_sensors_by_unit, f)
    
    def load_metadata(self):
        """Load flipped sensors metadata from file"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.flipped_sensors_by_unit = json.load(f)
            return True
        return False