#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# ---------- Deep Learning Models ----------

class LSTMClassifier(nn.Module):
    def __init__(self, in_channels: int = 1, lstm_hidden_size: int = 64, lstm_num_layers: int = 2):
        super(LSTMClassifier, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm = nn.LSTM(in_channels, lstm_hidden_size, lstm_num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch, window_size, 1)
        h0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size, device=x.device)
        c0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size, device=x.device)
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
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(nn.Conv1d(in_channels, num_channels, kernel_size, padding=padding, dilation=dilation))
            layers.append(nn.ReLU())
            in_channels = num_channels
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, window_size)
        out = self.network(x)
        out = out[:, :, :x.shape[-1]]
        out = out.mean(dim=2)
        logit = self.fc(out)
        return torch.sigmoid(logit).squeeze(-1)

class TransformerClassifier(nn.Module):
    def __init__(self, window_size: int, d_model: int = 64, nhead: int = 8, transformer_layers: int = 2):
        super(TransformerClassifier, self).__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)       # (batch, window_size, 1)
        x = self.input_proj(x)      # (batch, window_size, d_model)
        # With batch_first=True, we can use the input directly
        transformer_out = self.transformer_encoder(x)  # (batch, window_size, d_model)
        pool = transformer_out.mean(dim=1)             # average pooling over sequence
        logit = self.fc(pool)
        return torch.sigmoid(logit).squeeze(-1)

class HybridCNNLSTMClassifier(nn.Module):
    def __init__(self, window_size: int, cnn_channels: int = 16, lstm_hidden_size: int = 32, lstm_num_layers: int = 1):
        super(HybridCNNLSTMClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, cnn_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden_size, lstm_num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, window_size)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)  # (batch, cnn_channels, window_size//2)
        x = x.transpose(1, 2)  # (batch, window_size//2, cnn_channels)
        lstm_out, (hn, _) = self.lstm(x)
        logit = self.fc(hn[-1])
        return torch.sigmoid(logit).squeeze(-1)

class HybridTransformerLSTMClassifier(nn.Module):
    def __init__(self, window_size: int, d_model: int = 64, nhead: int = 8, transformer_layers: int = 2, lstm_hidden_size: int = 32, lstm_num_layers: int = 1, dropout: float = 0.1):
        super(HybridTransformerLSTMClassifier, self).__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.lstm = nn.LSTM(d_model, lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(-1)           # (batch, window_size, 1)
        x = self.input_proj(x)          # (batch, window_size, d_model)
        x = self.transformer_encoder(x) # (batch, window_size, d_model)
        lstm_out, (hn, _) = self.lstm(x)
        last_hidden = hn[-1]
        logits = self.fc(last_hidden)
        return self.sigmoid(logits).squeeze(-1)

# ---------- GNN-based Models ----------

class GNNClassifier(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_channels: int = 32, num_classes: int = 1):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # pool node features per graph
        logits = self.fc(x)
        return torch.sigmoid(logits).squeeze(-1)

class HybridGNNLSTMClassifier(nn.Module):
    def __init__(self, in_channels: int = 1, gnn_hidden_channels: int = 32, lstm_hidden_size: int = 16, lstm_layers: int = 1, num_classes: int = 1):
        super(HybridGNNLSTMClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, gnn_hidden_channels)
        self.conv2 = GCNConv(gnn_hidden_channels, gnn_hidden_channels)
        self.lstm = nn.LSTM(gnn_hidden_channels, lstm_hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        batch_size = int(batch.max().item()) + 1
        node_counts = torch.bincount(batch)
        T = node_counts[0].item()  # assume uniform node count per graph
        x = x.view(batch_size, T, -1)
        lstm_out, (hn, _) = self.lstm(x)
        last_hidden = hn[-1]
        logits = self.fc(last_hidden)
        return torch.sigmoid(logits).squeeze(-1)