#!/usr/bin/env python
import typer
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data_loader import simulate_ecg_signal, ECGEventDataset, split_dataset
from models import (
    LSTMClassifier,
    TCNClassifier,
    TransformerClassifier,
    HybridCNNLSTMClassifier,
    HybridTransformerLSTMClassifier,
    GNNClassifier,
    HybridGNNLSTMClassifier,
)
from training import get_optimizer, get_scheduler, train_model, train_gnn_model

app = typer.Typer()

device = torch.device(
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print("Using device:", device)

@app.command()
def simulate(
    fs: int = typer.Option(500, help="Sampling frequency (Hz)"),
    duration: int = typer.Option(60, help="Signal duration (seconds)"),
    noise: float = typer.Option(0.05, help="Noise level"),
    window_size: int = typer.Option(1000, help="Window size (number of samples)"),
    forecast_horizon: int = typer.Option(1000, help="Forecast horizon (number of samples)"),
    num_events: int = typer.Option(5, help="Number of events to inject"),
    event_duration: int = typer.Option(200, help="Duration of each event (samples)"),
    visualize: bool = typer.Option(True, help="Display plot?")
):
    ecg_signal = simulate_ecg_signal(fs, duration, noise)
    signal_length = len(ecg_signal)
    event_indicator = np.zeros(signal_length)
    for i in range(num_events):
        event_start = np.random.randint(window_size, signal_length - forecast_horizon - event_duration)
        event_indicator[event_start:event_start + event_duration] = 1

    if visualize:
        plt.figure(figsize=(12, 3))
        time_axis = np.arange(signal_length) / fs
        plt.plot(time_axis, ecg_signal, label="ECG signal")
        plt.plot(time_axis, event_indicator * (np.max(ecg_signal) * 0.8), "r.", label="Injected events")
        plt.xlabel("Time (s)")
        plt.ylabel("ECG amplitude")
        plt.title("Simulated ECG Signal with Injected Events")
        plt.legend()
        plt.show()
    typer.echo("ECG simulation completed and visualized.")

@app.command()
def train_dl(
    model_name: str = typer.Option("LSTMClassifier", help="Choose DL model"),
    num_epochs: int = typer.Option(10, help="Number of training epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    optimizer_name: str = typer.Option("adam", help="Optimizer: adam or sgd"),
    scheduler_name: str = typer.Option("none", help="Learning scheduler: none, step, or exponential"),
    step_size: int = typer.Option(5, help="Step size for scheduler"),
    gamma: float = typer.Option(0.5, help="Gamma for scheduler"),
    learning_rate: float = typer.Option(0.001, help="Learning rate"),
    # Data parameters:
    fs: int = typer.Option(500, help="Sampling frequency (Hz)"),
    duration: int = typer.Option(60, help="Signal duration (seconds)"),
    window_size: int = typer.Option(1000, help="Window size (samples)"),
    forecast_horizon: int = typer.Option(1000, help="Forecast horizon (samples)"),
    num_events: int = typer.Option(5, help="Number of events to inject"),
    event_duration: int = typer.Option(200, help="Duration of each event (samples)"),
    train_frac: float = typer.Option(0.8, help="Train fraction"),
    val_frac: float = typer.Option(0.1, help="Validation fraction"),
    test_frac: float = typer.Option(0.1, help="Test fraction"),
    # Model hyper-parameters for DL models:
    in_channels: int = typer.Option(1, help="Input channels"),
    lstm_hidden_size: int = typer.Option(64, help="LSTM hidden size"),
    lstm_num_layers: int = typer.Option(2, help="Number of LSTM layers"),
    num_channels: int = typer.Option(32, help="Number of channels for TCN"),
    num_levels: int = typer.Option(3, help="Number of levels for TCN"),
    d_model: int = typer.Option(64, help="d_model for Transformer"),
    nhead: int = typer.Option(8, help="nhead for Transformer"),
    transformer_layers: int = typer.Option(2, help="Number of Transformer layers"),
    cnn_channels: int = typer.Option(16, help="CNN channels for HybridCNNLSTM"),
    hybrid_dropout: float = typer.Option(0.1, help="Dropout for HybridTransformerLSTM")
):
    ecg_signal = simulate_ecg_signal(fs, duration, noise=0.05)
    signal_length = len(ecg_signal)
    event_indicator = np.zeros(signal_length)
    for i in range(num_events):
        event_start = np.random.randint(window_size, signal_length - forecast_horizon - event_duration)
        event_indicator[event_start:event_start + event_duration] = 1

    dataset = ECGEventDataset(ecg_signal, event_indicator, window_size, forecast_horizon)
    splits = split_dataset(dataset, train_frac, val_frac, test_frac)
    if len(splits) == 3:
        train_dataset, val_dataset, test_dataset = splits
    else:
        train_dataset, val_dataset = splits
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    models_dict = {
        "LSTMClassifier": lambda: LSTMClassifier(in_channels=in_channels, lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers),
        "TCNClassifier": lambda: TCNClassifier(num_channels=num_channels, num_levels=num_levels),
        "TransformerClassifier": lambda: TransformerClassifier(window_size=window_size, d_model=d_model, nhead=nhead, transformer_layers=transformer_layers),
        "HybridCNNLSTMClassifier": lambda: HybridCNNLSTMClassifier(window_size=window_size, cnn_channels=cnn_channels, lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers),
        "HybridTransformerLSTMClassifier": lambda: HybridTransformerLSTMClassifier(window_size=window_size, d_model=d_model, nhead=nhead, transformer_layers=transformer_layers, lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers, dropout=hybrid_dropout)
    }

    if model_name not in models_dict:
        typer.echo(f"Unknown model: {model_name}")
        raise typer.Exit()
    model = models_dict[model_name]()
    model = model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = get_optimizer(model, optimizer_name, learning_rate)
    scheduler = get_scheduler(optimizer, scheduler_name, step_size, gamma)

    typer.echo(f"Training {model_name} for {num_epochs} epochs...")
    model = train_model(model, train_loader, val_loader, num_epochs, device, criterion, optimizer, scheduler)
    typer.echo(f"Training of {model_name} completed.")

@app.command()
def train_gnn(
    model_name: str = typer.Option("GNN_only", help="Choose GNN model: GNN_only or Hybrid_GNN_LSTM"),
    num_epochs: int = typer.Option(10, help="Number of training epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    optimizer_name: str = typer.Option("adam", help="Optimizer: adam or sgd"),
    scheduler_name: str = typer.Option("none", help="Learning scheduler: none, step, or exponential"),
    step_size: int = typer.Option(5, help="Step size for scheduler"),
    gamma: float = typer.Option(0.5, help="Gamma for scheduler"),
    learning_rate: float = typer.Option(0.001, help="Learning rate"),
    # Data parameters:
    fs: int = typer.Option(500, help="Sampling frequency (Hz)"),
    duration: int = typer.Option(60, help="Signal duration (seconds)"),
    window_size: int = typer.Option(1000, help="Window size (samples)"),
    forecast_horizon: int = typer.Option(1000, help="Forecast horizon (samples)"),
    num_events: int = typer.Option(5, help="Number of events to inject"),
    event_duration: int = typer.Option(200, help="Event duration (samples)"),
    train_frac: float = typer.Option(0.8, help="Train fraction"),
    val_frac: float = typer.Option(0.1, help="Validation fraction"),
    test_frac: float = typer.Option(0.1, help="Test fraction"),
    # GNN model hyper-parameters:
    in_channels: int = typer.Option(1, help="Input channels"),
    hidden_channels: int = typer.Option(32, help="Hidden channels for GNN"),
    num_classes: int = typer.Option(1, help="Number of output classes"),
    gnn_hidden_channels: int = typer.Option(32, help="GNN hidden channels for hybrid model"),
    lstm_hidden_size: int = typer.Option(16, help="LSTM hidden size for hybrid model"),
    lstm_layers: int = typer.Option(1, help="Number of LSTM layers for hybrid model")
):
    ecg_signal = simulate_ecg_signal(fs, duration, noise=0.05)
    signal_length = len(ecg_signal)
    event_indicator = np.zeros(signal_length)
    for i in range(num_events):
        event_start = np.random.randint(window_size, signal_length - forecast_horizon - event_duration)
        event_indicator[event_start:event_start + event_duration] = 1

    dataset = ECGEventDataset(ecg_signal, event_indicator, window_size, forecast_horizon)
    splits = split_dataset(dataset, train_frac, val_frac, test_frac)
    if len(splits) == 3:
        train_dataset, val_dataset, test_dataset = splits
    else:
        train_dataset, val_dataset = splits
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    models_dict = {
        "GNN_only": lambda: GNNClassifier(in_channels=in_channels, hidden_channels=hidden_channels, num_classes=num_classes),
        "Hybrid_GNN_LSTM": lambda: HybridGNNLSTMClassifier(in_channels=in_channels, gnn_hidden_channels=gnn_hidden_channels, lstm_hidden_size=lstm_hidden_size, lstm_layers=lstm_layers, num_classes=num_classes)
    }
    if model_name not in models_dict:
        typer.echo(f"Unknown GNN model: {model_name}")
        raise typer.Exit()
    model = models_dict[model_name]()
    model = model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = get_optimizer(model, optimizer_name, learning_rate)
    scheduler = get_scheduler(optimizer, scheduler_name, step_size, gamma)

    typer.echo(f"Training {model_name} model for {num_epochs} epochs using GNN approach...")
    train_gnn_model(model, train_loader, val_loader, num_epochs, device, criterion, optimizer, scheduler)
    typer.echo(f"Training of {model_name} completed.")

if __name__ == "__main__":
    app()