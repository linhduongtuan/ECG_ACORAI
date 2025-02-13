#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# Import PyTorch Geometric modules for GNN models
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

import typer

app = typer.Typer()

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Choose device: GPU if available, otherwise CPU (or "mps" for macOS)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print("Using device:", device)


##########################################################################
# DATA SIMULATION AND LOADING
##########################################################################
def simulate_ecg_signal(fs: int, duration: int, noise: float):
    """Simulate an ECG signal using NeuroKit2."""
    ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=fs, noise=noise)
    return np.array(ecg_signal)


def load_ecg_signal(data_source: str, data_format: str = "npy"):
    """
    Load ECG signal from a file.
    If data_source=="simulate", then simulate the data.
    Otherwise, load data from the given file according to data_format.
    """
    if data_source.lower() == "simulate":
        # Caller must pass simulation parameters separately.
        raise ValueError("For simulation use the simulate command.")
    else:
        # Here we only support npy as example.
        if data_format.lower() == "npy":
            return np.load(data_source)
        else:
            raise NotImplementedError("Only npy format is currently supported.")


##########################################################################
# DATASET AND DATA SPLIT
##########################################################################
class ECGEventDataset(Dataset):
    """
    Dataset that creates sliding windows from an ECG signal and builds binary labels.
    The label is 1 if any event is present in the forecast horizon (following the window).
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
        X = self.ecg_signal[idx : idx + self.window_size]
        label = (
            1.0
            if np.any(
                self.event_indicator[
                    idx + self.window_size : idx
                    + self.window_size
                    + self.forecast_horizon
                ]
            )
            else 0.0
        )
        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        )


def split_dataset(
    dataset: Dataset, train_frac: float, val_frac: float, test_frac: float
):
    """
    Split the dataset into train, validation, and test sets.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Train, val, and test fractions must sum to 1.")
    n = len(dataset)
    train_len = int(train_frac * n)
    val_len = int(val_frac * n)
    test_len = n - train_len - val_len
    return (
        random_split(dataset, [train_len, val_len, test_len])
        if test_len > 0
        else random_split(dataset, [train_len, n - train_len])
    )


##########################################################################
# GRAPH CONVERSION HELPER
##########################################################################
def convert_to_graph(ecg_window: torch.Tensor, label: torch.Tensor) -> Data:
    """
    Converts a 1D ECG sliding window into a PyTorch Geometric Data object.
    Each point in the window becomes a node (with one feature) and nodes
    are connected in a chain (undirected).
    """
    x = ecg_window.unsqueeze(1)  # (window_size, 1)
    num_nodes = x.size(0)
    edge_list = []
    for i in range(num_nodes - 1):
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, y=label.unsqueeze(0))


##########################################################################
# MODEL DEFINITIONS: DEEP LEARNING
##########################################################################
class LSTMClassifier(nn.Module):
    def __init__(
        self, in_channels: int = 1, lstm_hidden_size: int = 64, lstm_num_layers: int = 2
    ):
        super(LSTMClassifier, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm = nn.LSTM(
            in_channels, lstm_hidden_size, lstm_num_layers, batch_first=True
        )
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, window_size)
        x = x.unsqueeze(-1)  # (batch, window_size, 1)
        h0 = torch.zeros(
            self.lstm_num_layers, x.size(0), self.lstm_hidden_size, device=x.device
        )
        c0 = torch.zeros(
            self.lstm_num_layers, x.size(0), self.lstm_hidden_size, device=x.device
        )
        out, _ = self.lstm(x, (h0, c0))
        last_hidden = out[:, -1, :]
        logit = self.fc(last_hidden)
        return torch.sigmoid(logit).squeeze(-1)


class TCNClassifier(nn.Module):
    def __init__(self, num_channels: int = 32, num_levels: int = 3):
        super(TCNClassifier, self).__init__()
        layers = []
        in_channels = 1
        kernel_size = 3
        for i in range(num_levels):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation
            layers.append(
                nn.Conv1d(
                    in_channels,
                    num_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            )
            layers.append(nn.ReLU())
            in_channels = num_channels
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels, 1)

    def forward(self, x):
        # x shape: (batch, window_size)
        x = x.unsqueeze(1)  # (batch, 1, window_size)
        out = self.network(x)
        out = out[:, :, : x.shape[-1]]
        out = out.mean(dim=2)
        logit = self.fc(out)
        return torch.sigmoid(logit).squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        window_size: int,
        d_model: int = 64,
        nhead: int = 8,
        transformer_layers: int = 2,
    ):
        super(TransformerClassifier, self).__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (batch, window_size)
        x = x.unsqueeze(-1)  # (batch, window_size, 1)
        x = self.input_proj(x)  # (batch, window_size, d_model)
        x = x.transpose(0, 1)  # (window_size, batch, d_model)
        transformer_out = self.transformer_encoder(x)
        pool = transformer_out.mean(dim=0)  # (batch, d_model)
        logit = self.fc(pool)
        return torch.sigmoid(logit).squeeze(-1)


class HybridCNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        window_size: int,
        cnn_channels: int = 16,
        lstm_hidden_size: int = 32,
        lstm_num_layers: int = 1,
    ):
        super(HybridCNNLSTMClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, cnn_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(
            cnn_channels, lstm_hidden_size, lstm_num_layers, batch_first=True
        )
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        # x: (batch, window_size)
        x = x.unsqueeze(1)  # (batch, 1, window_size)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)  # (batch, cnn_channels, window_size//2)
        x = x.transpose(1, 2)  # (batch, window_size//2, cnn_channels)
        lstm_out, (hn, _) = self.lstm(x)
        logit = self.fc(hn[-1])
        return torch.sigmoid(logit).squeeze(-1)


class HybridTransformerLSTMClassifier(nn.Module):
    def __init__(
        self,
        window_size: int,
        d_model: int = 64,
        nhead: int = 8,
        transformer_layers: int = 2,
        lstm_hidden_size: int = 32,
        lstm_num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super(HybridTransformerLSTMClassifier, self).__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.lstm = nn.LSTM(
            d_model, lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True
        )
        self.fc = nn.Linear(lstm_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, window_size)
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        lstm_out, (hn, _) = self.lstm(x)
        last_hidden = hn[-1]
        logits = self.fc(last_hidden)
        return self.sigmoid(logits).squeeze(-1)


##########################################################################
# MODEL DEFINITIONS: GNN-BASED
##########################################################################
class GNNClassifier(nn.Module):
    def __init__(
        self, in_channels: int = 1, hidden_channels: int = 32, num_classes: int = 1
    ):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        logits = self.fc(x)
        return torch.sigmoid(logits).squeeze(-1)


class HybridGNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        gnn_hidden_channels: int = 32,
        lstm_hidden_size: int = 16,
        lstm_layers: int = 1,
        num_classes: int = 1,
    ):
        super(HybridGNNLSTMClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, gnn_hidden_channels)
        self.conv2 = GCNConv(gnn_hidden_channels, gnn_hidden_channels)
        self.lstm = nn.LSTM(
            gnn_hidden_channels,
            lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        batch_size = int(batch.max().item()) + 1
        node_counts = torch.bincount(batch)
        T = node_counts[0].item()
        x = x.view(batch_size, T, -1)
        lstm_out, (hn, _) = self.lstm(x)
        last_hidden = hn[-1]
        logits = self.fc(last_hidden)
        return torch.sigmoid(logits).squeeze(-1)


##########################################################################
# OPTIMIZER AND SCHEDULER HELPERS
##########################################################################
def get_optimizer(model: nn.Module, opt_name: str, lr: float):
    if opt_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif opt_name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(optimizer, sched_name: str, step_size: int, gamma: float):
    if sched_name.lower() == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_name.lower() == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif sched_name.lower() in ["none", ""]:
        return None
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


##########################################################################
# TRAINING HELPER FUNCTIONS
##########################################################################
def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs: int,
    device,
    criterion,
    optimizer,
    scheduler=None,
):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            preds = (outputs >= 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        if scheduler is not None:
            scheduler.step()
        train_loss /= total
        train_acc = train_correct / total

        # Evaluation on validation set
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item() * X_val.size(0)
                preds = (outputs >= 0.5).float()
                val_correct += (preds == y_val).sum().item()
                total_val += y_val.size(0)
        val_loss /= total_val
        val_acc = val_correct / total_val

        typer.echo(
            f"Epoch {epoch + 1}/{num_epochs}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}"
        )
    return model


def train_gnn_model(
    model,
    train_loader,
    val_loader,
    num_epochs: int,
    device,
    criterion,
    optimizer,
    scheduler=None,
):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        for X_batch, y_batch in train_loader:
            batch_graphs = []
            for x, y in zip(X_batch, y_batch):
                data = convert_to_graph(x, y)
                batch_graphs.append(data)
            batch_data = Batch.from_data_list(batch_graphs).to(device)
            optimizer.zero_grad()
            output = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            loss = criterion(output, batch_data.y.float().view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds = (output > 0.5).float()
            acc = (preds == batch_data.y.float().view(-1)).sum().item() / len(y_batch)
            epoch_acc += acc
            num_batches += 1
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        typer.echo(
            f"Epoch {epoch + 1}/{num_epochs}: Train Loss {avg_loss:.4f}, Train Acc {avg_acc:.4f}"
        )

        if scheduler is not None:
            scheduler.step()

        # Final evaluation on validation set
        model.eval()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                batch_graphs = []
                for x, y in zip(X_batch, y_batch):
                    data = convert_to_graph(x, y)
                    batch_graphs.append(data)
                batch_data = Batch.from_data_list(batch_graphs).to(device)
                output = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                loss = criterion(output, batch_data.y.float().view(-1))
                epoch_loss += loss.item()
                preds = (output > 0.5).float()
                acc = (preds == batch_data.y.float().view(-1)).sum().item() / len(
                    y_batch
                )
                epoch_acc += acc
                num_batches += 1
        val_loss = epoch_loss / num_batches
        val_acc = epoch_acc / num_batches
        typer.echo(f"Validation: Loss {val_loss:.4f}, Acc {val_acc:.4f}")


##########################################################################
# TYper COMMANDS
##########################################################################
@app.command()
def simulate(
    fs: int = typer.Option(500, help="Sampling frequency (Hz)"),
    duration: int = typer.Option(60, help="Signal duration (seconds)"),
    noise: float = typer.Option(0.05, help="Noise level"),
    window_size: int = typer.Option(1000, help="Window size (number of samples)"),
    forecast_horizon: int = typer.Option(
        1000, help="Forecast horizon (number of samples)"
    ),
    num_events: int = typer.Option(5, help="Number of events to inject"),
    event_duration: int = typer.Option(200, help="Duration of each event (samples)"),
    visualize: bool = typer.Option(True, help="Display plot?"),
):
    """
    Simulate an ECG signal, inject synthetic events, and (optionally) visualize the result.
    """
    ecg_signal = simulate_ecg_signal(fs, duration, noise)
    signal_length = len(ecg_signal)
    event_indicator = np.zeros(signal_length)
    for i in range(num_events):
        event_start = np.random.randint(
            window_size, signal_length - forecast_horizon - event_duration
        )
        event_indicator[event_start : event_start + event_duration] = 1

    if visualize:
        plt.figure(figsize=(12, 3))
        time_axis = np.arange(signal_length) / fs
        plt.plot(time_axis, ecg_signal, label="ECG signal")
        plt.plot(
            time_axis,
            event_indicator * (np.max(ecg_signal) * 0.8),
            "r.",
            label="Injected event",
        )
        plt.xlabel("Time (s)")
        plt.ylabel("ECG amplitude")
        plt.title("Simulated ECG Signal with Injected Events")
        plt.legend()
        plt.show()

    typer.echo("ECG simulation completed and visualized.")


@app.command()
def train_dl(
    model_name: str = typer.Option(
        "LSTMClassifier",
        help="Choose DL model: LSTMClassifier, TCNClassifier, TransformerClassifier, HybridCNNLSTMClassifier, or HybridTransformerLSTMClassifier",
    ),
    num_epochs: int = typer.Option(10, help="Number of training epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    optimizer_name: str = typer.Option("adam", help="Optimizer: adam or sgd"),
    scheduler_name: str = typer.Option(
        "none", help="Learning scheduler: none, step, or exponential"
    ),
    step_size: int = typer.Option(5, help="Step size for step LR scheduler (if used)"),
    gamma: float = typer.Option(0.5, help="Gamma for learning scheduler (if used)"),
    learning_rate: float = typer.Option(0.001, help="Learning rate"),
    # Dataset parameters:
    fs: int = typer.Option(500, help="Sampling frequency (Hz)"),
    duration: int = typer.Option(60, help="Signal duration (seconds)"),
    window_size: int = typer.Option(1000, help="Window size (samples)"),
    forecast_horizon: int = typer.Option(1000, help="Forecast horizon (samples)"),
    num_events: int = typer.Option(5, help="Number of events to inject"),
    event_duration: int = typer.Option(200, help="Duration of each event (samples)"),
    train_frac: float = typer.Option(0.8, help="Fraction of data for training"),
    val_frac: float = typer.Option(0.1, help="Fraction of data for validation"),
    test_frac: float = typer.Option(0.1, help="Fraction of data for test"),
    # Model hyper-parameters for DL models:
    in_channels: int = typer.Option(1, help="Input channels (default=1)"),
    lstm_hidden_size: int = typer.Option(64, help="LSTM hidden size (for LSTM models)"),
    lstm_num_layers: int = typer.Option(2, help="Number of LSTM layers"),
    num_channels: int = typer.Option(32, help="Number of channels for TCN"),
    num_levels: int = typer.Option(3, help="Number of levels for TCN"),
    d_model: int = typer.Option(64, help="d_model for Transformer models"),
    nhead: int = typer.Option(8, help="nhead for Transformer models"),
    transformer_layers: int = typer.Option(
        2, help="Number of Transformer encoder layers"
    ),
    cnn_channels: int = typer.Option(
        16, help="CNN channels for HybridCNNLSTMClassifier"
    ),
    hybrid_dropout: float = typer.Option(
        0.1, help="Dropout rate for HybridTransformerLSTMClassifier"
    ),
):
    """
    Train a deep learning model for ECG event prediction.
    Model-specific hyperparameters are provided as options.
    Data is simulated (or could be adapted to load from a source).
    """
    # Simulate ECG and inject events.
    ecg_signal = simulate_ecg_signal(fs, duration, noise=0.05)
    signal_length = len(ecg_signal)
    event_indicator = np.zeros(signal_length)
    for i in range(num_events):
        event_start = np.random.randint(
            window_size, signal_length - forecast_horizon - event_duration
        )
        event_indicator[event_start : event_start + event_duration] = 1

    dataset = ECGEventDataset(
        ecg_signal, event_indicator, window_size, forecast_horizon
    )
    splits = split_dataset(dataset, train_frac, val_frac, test_frac)
    if len(splits) == 3:
        train_dataset, val_dataset, test_dataset = splits
    else:
        train_dataset, val_dataset = splits

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Map model names to constructors using the hyperparameter options.
    models_dict = {
        "LSTMClassifier": lambda: LSTMClassifier(
            in_channels=in_channels,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
        ),
        "TCNClassifier": lambda: TCNClassifier(
            num_channels=num_channels, num_levels=num_levels
        ),
        "TransformerClassifier": lambda: TransformerClassifier(
            window_size=window_size,
            d_model=d_model,
            nhead=nhead,
            transformer_layers=transformer_layers,
        ),
        "HybridCNNLSTMClassifier": lambda: HybridCNNLSTMClassifier(
            window_size=window_size,
            cnn_channels=cnn_channels,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
        ),
        "HybridTransformerLSTMClassifier": lambda: HybridTransformerLSTMClassifier(
            window_size=window_size,
            d_model=d_model,
            nhead=nhead,
            transformer_layers=transformer_layers,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            dropout=hybrid_dropout,
        ),
    }
    if model_name not in models_dict:
        typer.echo(f"Unknown model: {model_name}")
        raise typer.Exit()
    model = models_dict[model_name]()
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = get_optimizer(model, optimizer_name, learning_rate)
    scheduler = get_scheduler(optimizer, scheduler_name, step_size, gamma)

    typer.echo(f"Training {model_name} for {num_epochs} epochs...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs,
        device,
        criterion,
        optimizer,
        scheduler,
    )
    typer.echo(f"Training of {model_name} completed.")


@app.command()
def train_gnn(
    model_name: str = typer.Option(
        "GNN_only", help="Choose GNN model: GNN_only or Hybrid_GNN_LSTM"
    ),
    num_epochs: int = typer.Option(10, help="Number of training epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    optimizer_name: str = typer.Option("adam", help="Optimizer: adam or sgd"),
    scheduler_name: str = typer.Option(
        "none", help="Learning scheduler: none, step, or exponential"
    ),
    step_size: int = typer.Option(5, help="Step size for scheduler"),
    gamma: float = typer.Option(0.5, help="Gamma for scheduler"),
    learning_rate: float = typer.Option(0.001, help="Learning rate"),
    # Data/dataset options:
    fs: int = typer.Option(500, help="Sampling frequency (Hz)"),
    duration: int = typer.Option(60, help="Signal duration (seconds)"),
    window_size: int = typer.Option(1000, help="Window size (samples)"),
    forecast_horizon: int = typer.Option(1000, help="Forecast horizon (samples)"),
    num_events: int = typer.Option(5, help="Number of events to inject"),
    event_duration: int = typer.Option(200, help="Event duration (samples)"),
    train_frac: float = typer.Option(0.8, help="Fraction of data for training"),
    val_frac: float = typer.Option(0.1, help="Fraction of data for validation"),
    test_frac: float = typer.Option(0.1, help="Fraction of data for test"),
    # Model hyper-parameters for GNN models:
    in_channels: int = typer.Option(1, help="Input channels"),
    hidden_channels: int = typer.Option(32, help="Hidden channels for GNN"),
    num_classes: int = typer.Option(1, help="Number of output classes for GNN"),
    gnn_hidden_channels: int = typer.Option(
        32, help="GNN hidden channels for hybrid GNN+LSTM"
    ),
    lstm_hidden_size: int = typer.Option(
        16, help="LSTM hidden size for hybrid GNN+LSTM"
    ),
    lstm_layers: int = typer.Option(
        1, help="Number of LSTM layers for hybrid GNN+LSTM"
    ),
):
    """
    Train a GNN-based model for ECG event prediction.
    Options allow customizing both pure GNN and hybrid GNN+LSTM architectures.
    """
    # Simulate ECG and inject events.
    ecg_signal = simulate_ecg_signal(fs, duration, noise=0.05)
    signal_length = len(ecg_signal)
    event_indicator = np.zeros(signal_length)
    for i in range(num_events):
        event_start = np.random.randint(
            window_size, signal_length - forecast_horizon - event_duration
        )
        event_indicator[event_start : event_start + event_duration] = 1

    dataset = ECGEventDataset(
        ecg_signal, event_indicator, window_size, forecast_horizon
    )
    splits = split_dataset(dataset, train_frac, val_frac, test_frac)
    if len(splits) == 3:
        train_dataset, val_dataset, test_dataset = splits
    else:
        train_dataset, val_dataset = splits

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    models_dict = {
        "GNN_only": lambda: GNNClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
        ),
        "Hybrid_GNN_LSTM": lambda: HybridGNNLSTMClassifier(
            in_channels=in_channels,
            gnn_hidden_channels=gnn_hidden_channels,
            lstm_hidden_size=lstm_hidden_size,
            lstm_layers=lstm_layers,
            num_classes=num_classes,
        ),
    }
    if model_name not in models_dict:
        typer.echo(f"Unknown GNN model: {model_name}")
        raise typer.Exit()
    model = models_dict[model_name]()
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = get_optimizer(model, optimizer_name, learning_rate)
    scheduler = get_scheduler(optimizer, scheduler_name, step_size, gamma)

    typer.echo(f"Training {model_name} model for {num_epochs} epochs...")
    train_gnn_model(
        model,
        train_loader,
        val_loader,
        num_epochs,
        device,
        criterion,
        optimizer,
        scheduler,
    )
    typer.echo(f"Training of {model_name} completed.")


if __name__ == "__main__":
    app()
