# ecg_timeseries_classifier_torch.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Dict, Optional, Union, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom exception for the ECG classifier
class ECGClassifierError(Exception):
    pass

class ECGTimeSeriesClassifier(nn.Module):
    def __init__(self, input_length: int, num_classes: int = 2):
        """
        A deep learning model for time-series classification of ECG signals.
        It applies two convolutional layers followed by max pooling, then processes
        the feature sequence with an LSTM and finally predicts the class.

        Parameters:
          input_length: Length of the (preprocessed) ECG time series.
          num_classes : Number of output classes (default is 2, e.g. risk vs. no risk).

        Raises:
          ECGClassifierError: If input parameters are invalid.
        """
        if input_length < 16:  # Minimum length needed for the architecture
            raise ECGClassifierError(
                f"input_length must be at least 16, got {input_length}"
            )
        if num_classes < 2:
            raise ECGClassifierError(
                f"num_classes must be at least 2, got {num_classes}"
            )

        super(ECGTimeSeriesClassifier, self).__init__()

        self.input_length = input_length
        self.num_classes = num_classes

        # Convolutional layers to extract local features
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(2)  # halves the time dimension

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)

        # After two pooling operations, the time dimension becomes: input_length // 4
        lstm_input_size = 32  # number of channels from conv2
        self.lstm = nn.LSTM(
            input_size=lstm_input_size, hidden_size=64, num_layers=1, batch_first=True
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters:
          x : torch.Tensor
              Input tensor of shape (batch_size, 1, input_length)

        Returns:
          torch.Tensor
              Output tensor of shape (batch_size, num_classes)

        Raises:
          ECGClassifierError: If input tensor has incorrect shape.
        """
        if x.dim() != 3:
            raise ECGClassifierError(f"Expected 3D input tensor, got {x.dim()}D")
        if x.size(1) != 1:
            raise ECGClassifierError(f"Expected 1 input channel, got {x.size(1)}")
        if x.size(2) != self.input_length:
            raise ECGClassifierError(
                f"Expected input length {self.input_length}, got {x.size(2)}"
            )

        try:
            # Convolution Block 1
            x = self.conv1(x)  # (batch, 16, input_length)
            x = self.bn1(x)
            x = torch.relu(x)
            x = self.pool(x)   # (batch, 16, input_length/2)
        except Exception as e:
            logger.error(f"Error in convolution block 1: {str(e)}")
            raise ECGClassifierError(f"Forward pass failed in conv block 1: {str(e)}")

        # Convolution Block 2
        x = self.conv2(x)  # (batch, 32, input_length/2)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)   # (batch, 32, input_length/4)

        # Prepare data for LSTM: permute from (batch, channels, seq_length) --> (batch, seq_length, channels)
        x = x.permute(0, 2, 1)  # (batch, input_length/4, 32)

        # Process with LSTM: take the last time step's output as the embedding
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq_length, 64)
        last_output = lstm_out[:, -1, :]  # (batch, 64)

        # Final classification layer
        output = self.fc(last_output)  # (batch, num_classes)
        return output

def train_time_series_classifier(
    model: ECGTimeSeriesClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
) -> Dict[str, list]:
    """
    Train the ECG time-series classifier.

    Parameters:
      model         : Instance of ECGTimeSeriesClassifier.
      x_train       : NumPy array of shape (num_samples, input_length) (or with channel dimension).
      y_train       : NumPy array of integer labels.
      x_val, y_val  : (Optional) Validation data.
      epochs        : Number of training epochs.
      batch_size    : Batch size.
      learning_rate : Learning rate for the optimizer.

    Returns:
      A dictionary containing training history.
    """
    try:
        # Validate inputs
        if not isinstance(x_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise ECGClassifierError("Training data must be numpy arrays")
        if len(x_train) != len(y_train):
            raise ECGClassifierError(
                f"Length mismatch: x_train ({len(x_train)}) != y_train ({len(y_train)})"
            )
        if not np.isfinite(x_train).all():
            raise ECGClassifierError("x_train contains invalid values (inf or nan)")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        model.to(device)

        # Ensure input has shape (N, 1, input_length)
        if x_train.ndim == 2:
            x_train = np.expand_dims(x_train, axis=1)
        elif x_train.ndim != 3:
            raise ECGClassifierError(f"x_train must be 2D or 3D, got {x_train.ndim}D")

        if x_train.shape[1] != 1:
            raise ECGClassifierError(f"Expected 1 channel, got {x_train.shape[1]}")
        if x_train.shape[2] != model.input_length:
            raise ECGClassifierError(
                f"Input length {x_train.shape[2]} doesn't match model input length {model.input_length}"
            )

        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Set up loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        history = {"train_loss": [], "train_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0

        if x_val is not None and y_val is not None:
            if len(x_val) != len(y_val):
                raise ECGClassifierError(
                    f"Length mismatch: x_val ({len(x_val)}) != y_val ({len(y_val)})"
                )
            if x_val.ndim == 2:
                x_val = np.expand_dims(x_val, axis=1)
            elif x_val.ndim != 3:
                raise ECGClassifierError(f"x_val must be 2D or 3D, got {x_val.ndim}D")
            x_val = torch.tensor(x_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.long)
            val_dataset = TensorDataset(x_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            history["val_loss"] = []
            history["val_acc"] = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            epoch_loss /= len(train_dataset)
            epoch_acc = 100 * correct / total
            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(epoch_acc)

            if x_val is not None and y_val is not None:
                model.eval()
                val_loss = 0.0
                correct_val = 0
                total_val = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_x.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        total_val += batch_y.size(0)
                        correct_val += (predicted == batch_y).sum().item()
                val_loss /= len(val_dataset)
                val_acc = 100 * correct_val / total_val
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break

                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%"
                )
        return history

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise ECGClassifierError(f"Training failed: {str(e)}")

def predict(
    model: ECGTimeSeriesClassifier,
    x: np.ndarray,
    return_probabilities: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Predict classes for given ECG time-series data.

    Parameters:
      model : ECGTimeSeriesClassifier
          Trained model instance.
      x : np.ndarray
          Input data of shape (num_samples, input_length) or (num_samples, 1, input_length).
      return_probabilities : bool, optional
          If True, returns both predictions and probabilities.

    Returns:
      Either an array of predicted class labels or a tuple of (predictions, probabilities).
    """
    try:
        if not isinstance(x, np.ndarray):
            raise ECGClassifierError("Input must be a numpy array")
        if not np.isfinite(x).all():
            raise ECGClassifierError("Input contains invalid values (inf or nan)")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        if x.ndim == 2:
            x = np.expand_dims(x, axis=1)
        elif x.ndim != 3:
            raise ECGClassifierError(f"Input must be 2D or 3D, got {x.ndim}D")
        if x.shape[1] != 1:
            raise ECGClassifierError(f"Expected 1 channel, got {x.shape[1]}")
        if x.shape[2] != model.input_length:
            raise ECGClassifierError(
                f"Input length {x.shape[2]} doesn't match model input length {model.input_length}"
            )
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(x_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
        if return_probabilities:
            return predictions.cpu().numpy(), probabilities.cpu().numpy()
        return predictions.cpu().numpy()

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise ECGClassifierError(f"Prediction failed: {str(e)}")