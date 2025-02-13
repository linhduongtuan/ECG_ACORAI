#!/usr/bin/env python
import numpy as np
import neurokit2 as nk
import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def simulate_ecg_signal(fs: int, duration: int, noise: float):
    """Simulate an ECG signal using NeuroKit2."""
    ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=fs, noise=noise)
    return np.array(ecg_signal)

def load_ecg_signal(data_source: str, data_format: str = "npy"):
    """
    Load ECG signal from a file.
    If data_source=="simulate", then simulation should be used via simulate_ecg_signal.
    Otherwise, load data from the given file.
    """
    if data_source.lower() == "simulate":
        raise ValueError("For simulation, use simulate_ecg_signal function.")
    else:
        if data_format.lower() == "npy":
            return np.load(data_source)
        else:
            raise NotImplementedError("Only npy format is currently supported.")

class ECGEventDataset(Dataset):
    """
    Dataset that creates sliding windows from an ECG signal and builds binary labels.
    The label is 1 if any event is present in the forecast horizon following the window.
    """
    def __init__(self, ecg_signal, event_indicator, window_size, forecast_horizon):
        self.ecg_signal = ecg_signal
        self.event_indicator = event_indicator
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.n_samples = len(ecg_signal) - window_size - forecast_horizon + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        X = self.ecg_signal[idx: idx + self.window_size]
        label = 1.0 if np.any(
            self.event_indicator[idx + self.window_size: idx + self.window_size + self.forecast_horizon]
        ) else 0.0
        return torch.tensor(X, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def split_dataset(dataset: Dataset, train_frac: float, val_frac: float, test_frac: float):
    """
    Split the dataset into train, validation, and test sets.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Train, val, and test fractions must sum to 1.")
    n = len(dataset)
    train_len = int(train_frac * n)
    val_len = int(val_frac * n)
    test_len = n - train_len - val_len
    if test_len > 0:
        return random_split(dataset, [train_len, val_len, test_len])
    else:
        return random_split(dataset, [train_len, n - train_len])

def convert_to_graph(ecg_window: torch.Tensor, label: torch.Tensor) -> Data:
    """
    Converts a 1D ECG sliding window into a PyTorch Geometric Data object.
    Each point in the window becomes a node (with one feature) and nodes are
    connected in a chain (undirected).
    """
    x = ecg_window.unsqueeze(1)  # shape: (window_size, 1)
    num_nodes = x.size(0)
    edge_list = []
    for i in range(num_nodes - 1):
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, y=label.unsqueeze(0))