#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch
import typer

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

def train_model(model, train_loader, val_loader, num_epochs: int, device, criterion, optimizer, scheduler=None):
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

        typer.echo(f"Epoch {epoch+1}/{num_epochs}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f} | Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

    return model

def train_gnn_model(model, train_loader, val_loader, num_epochs: int, device, criterion, optimizer, scheduler=None):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        for X_batch, y_batch in train_loader:
            batch_graphs = []
            # Import the graph conversion helper from data_loader
            from data_loader import convert_to_graph
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
        if scheduler is not None:
            scheduler.step()
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        typer.echo(f"Epoch {epoch+1}/{num_epochs}: Train Loss {avg_loss:.4f}, Train Acc {avg_acc:.4f}")

        model.eval()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                batch_graphs = []
                from data_loader import convert_to_graph
                for x, y in zip(X_batch, y_batch):
                    data = convert_to_graph(x, y)
                    batch_graphs.append(data)
                batch_data = Batch.from_data_list(batch_graphs).to(device)
                output = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                loss = criterion(output, batch_data.y.float().view(-1))
                epoch_loss += loss.item()
                preds = (output > 0.5).float()
                acc = (preds == batch_data.y.float().view(-1)).sum().item() / len(y_batch)
                epoch_acc += acc
                num_batches += 1
        val_loss = epoch_loss / num_batches
        val_acc = epoch_acc / num_batches
        typer.echo(f"Validation: Loss {val_loss:.4f}, Acc {val_acc:.4f}")