# ecg_transformer_classifier_torch.py

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Dict, Optional, Union, Tuple, List

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Custom exceptions for configuration and input problems
class TransformerConfigError(Exception):
    """Exception raised for errors in the Transformer configuration."""

    pass


class TransformerInputError(Exception):
    """Exception raised for errors in the input data."""

    pass


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Implements positional encoding as described in 'Attention Is All You Need'.

        Parameters:
          d_model : int
              Embedding dimensionality.
          dropout : float, optional
              Dropout probability, by default 0.1.
          max_len : int, optional
              Maximum supported input sequence length, by default 5000.

        Raises:
          TransformerConfigError: If any parameters are invalid.
        """
        if d_model <= 0:
            raise TransformerConfigError(f"d_model must be positive, got {d_model}")
        if not 0 <= dropout < 1:
            raise TransformerConfigError(f"dropout must be in [0,1), got {dropout}")
        if max_len <= 0:
            raise TransformerConfigError(f"max_len must be positive, got {max_len}")

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to the input tensor.

        Parameters:
          x : torch.Tensor
              Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
          torch.Tensor with positional encoding added (and dropout applied).

        Raises:
          TransformerInputError: If input tensor shape is invalid.
        """
        if x.dim() != 3:
            raise TransformerInputError(f"Expected 3D input tensor, got {x.dim()}D")
        if x.size(2) != self.pe.size(2):
            raise TransformerInputError(
                f"Input feature dim {x.size(2)} doesn't match positional encoding dim {self.pe.size(2)}"
            )
        if x.size(1) > self.pe.size(1):
            raise TransformerInputError(
                f"Input sequence length {x.size(1)} exceeds maximum length {self.pe.size(1)}"
            )
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ECGTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_length: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        """
        Transformer-based classifier for ECG time-series.

        Parameters:
          input_length : int
              Length of the input ECG sequence.
          d_model : int, optional
              Dimensionality of the embedding space, default is 64.
          nhead : int, optional
              Number of attention heads, default is 4.
          num_layers : int, optional
              Number of transformer encoder layers, default is 2.
          num_classes : int, optional
              Number of output classes, default is 2.
          dropout : float, optional
              Dropout rate, default is 0.1.

        Raises:
          TransformerConfigError: If any parameters are invalid.
        """
        if input_length <= 0:
            raise TransformerConfigError(
                f"input_length must be positive, got {input_length}"
            )
        if d_model <= 0:
            raise TransformerConfigError(f"d_model must be positive, got {d_model}")
        if d_model % nhead != 0:
            raise TransformerConfigError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        if nhead <= 0:
            raise TransformerConfigError(f"nhead must be positive, got {nhead}")
        if num_layers <= 0:
            raise TransformerConfigError(
                f"num_layers must be positive, got {num_layers}"
            )
        if num_classes <= 1:
            raise TransformerConfigError(
                f"num_classes must be greater than 1, got {num_classes}"
            )
        if not 0 <= dropout < 1:
            raise TransformerConfigError(f"dropout must be in [0,1), got {dropout}")

        super(ECGTransformerClassifier, self).__init__()
        self.input_length = input_length
        self.d_model = d_model

        # Map each scalar input to a d_model-dimensional embedding.
        self.embedding = nn.Linear(1, d_model)
        # Add positional encoding.
        self.positional_encoding = PositionalEncoding(
            d_model, dropout=dropout, max_len=input_length
        )
        # Transformer encoder: using batch_first=True for input shape (batch, seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # Final classification layer.
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Expected input shape:
          x: Tensor of shape (batch_size, seq_len) or (batch_size, seq_len, 1).

        Returns:
          Logits of shape (batch_size, num_classes).
        """
        # If input is (batch, seq_len), add a channel dimension.
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        # Embed the input; shape becomes (batch, seq_len, d_model)
        x = self.embedding(x)
        # Add positional encoding.
        x = self.positional_encoding(x)
        # Process through the transformer encoder.
        x = self.transformer_encoder(x)
        # Aggregate over the sequence dimension (here, via mean pooling).
        x = x.mean(dim=1)
        # Final classification.
        logits = self.fc(x)
        return logits


def train_transformer_classifier(
    model: ECGTransformerClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
    gradient_clip_val: float = 1.0,
) -> Dict[str, List[float]]:
    """
    Train the transformer-based ECG classifier.

    Parameters:
      model: Instance of ECGTransformerClassifier.
      x_train: NumPy array of shape (num_samples, input_length). Inputs should be normalized.
      y_train: NumPy array of integer labels.
      x_val, y_val: (Optional) Validation data.
      epochs: Number of training epochs.
      batch_size: Training batch size.
      learning_rate: Optimizer learning rate.

    Returns:
      History dictionary containing training (and optional validation) loss and accuracy.
    """
    try:
        # Input validation for training data
        if not isinstance(x_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise TransformerInputError("Training data must be numpy arrays")
        if len(x_train) != len(y_train):
            raise TransformerInputError(
                f"Length mismatch: x_train ({len(x_train)}) != y_train ({len(y_train)})"
            )
        if not np.isfinite(x_train).all():
            raise TransformerInputError("x_train contains invalid values (inf or nan)")

        # Hyperparameter validation
        if epochs <= 0:
            raise TransformerConfigError(f"epochs must be positive, got {epochs}")
        if batch_size <= 0:
            raise TransformerConfigError(
                f"batch_size must be positive, got {batch_size}"
            )
        if learning_rate <= 0:
            raise TransformerConfigError(
                f"learning_rate must be positive, got {learning_rate}"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        logger.info(f"Using device: {device}")
        model.to(device)

        # Ensure training data is of shape (num_samples, input_length).
        # If provided as (num_samples, input_length, 1), squeeze the last dimension.
        if x_train.ndim == 3:
            x_train = x_train.squeeze(-1)
        elif x_train.ndim != 2:
            raise TransformerInputError(
                f"x_train must be 2D or 3D, got {x_train.ndim}D"
            )

        if x_train.shape[1] != model.input_length:
            raise TransformerInputError(
                f"Input length {x_train.shape[1]} doesn't match model input length {model.input_length}"
            )

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        history = {"train_loss": [], "train_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0

        if x_val is not None and y_val is not None:
            if len(x_val) != len(y_val):
                raise TransformerInputError(
                    f"Length mismatch: x_val ({len(x_val)}) != y_val ({len(y_val)})"
                )
            if x_val.ndim == 3:
                x_val = x_val.squeeze(-1)
            elif x_val.ndim != 2:
                raise TransformerInputError(
                    f"x_val must be 2D or 3D, got {x_val.ndim}D"
                )

            if x_val.shape[1] != model.input_length:
                raise TransformerInputError(
                    f"Validation input length {x_val.shape[1]} doesn't match model input length {model.input_length}"
                )

            x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
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

                # Gradient clipping to keep training stable
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
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
                        _, predicted = torch.max(outputs, 1)
                        total_val += batch_y.size(0)
                        correct_val += (predicted == batch_y).sum().item()
                val_loss /= len(val_dataset)
                val_acc = 100 * correct_val / total_val
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Optionally save best model state here.
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(
                            f"Early stopping triggered after {epoch + 1} epochs"
                        )
                        break

                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%"
                )
        return history

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise TransformerInputError(f"Training failed: {str(e)}")


def predict_transformer(
    model: ECGTransformerClassifier, x: np.ndarray, return_probabilities: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Predict classes for given ECG time-series data using the trained transformer classifier.

    Parameters:
      model : ECGTransformerClassifier
          Trained transformer model.
      x : np.ndarray
          Input data of shape (num_samples, input_length) or (num_samples, input_length, 1).
      return_probabilities : bool, optional
          If True, returns both predictions and probabilities.

    Returns:
      Either an array of predicted class labels or a tuple (predictions, probabilities).
    """
    try:
        if not isinstance(x, np.ndarray):
            raise TransformerInputError("Input must be a numpy array")
        if not np.isfinite(x).all():
            raise TransformerInputError("Input contains invalid values (inf or nan)")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Ensure input shape is (num_samples, input_length)
        if x.ndim == 3:
            x = x.squeeze(-1)
        elif x.ndim != 2:
            raise TransformerInputError(f"Input must be 2D or 3D, got {x.ndim}D")

        if x.shape[1] != model.input_length:
            raise TransformerInputError(
                f"Input length {x.shape[1]} doesn't match model input length {model.input_length}"
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
        raise TransformerInputError(f"Prediction failed: {str(e)}")


# Example usage:
if __name__ == "__main__":
    # Create dummy training data:
    num_samples = 1000
    input_length = 5000  # Each ECG segment
    x_train = np.random.rand(num_samples, input_length).astype(np.float32)
    y_train = np.random.randint(0, 2, size=(num_samples,))

    # Initialize model
    model = ECGTransformerClassifier(
        input_length=input_length,
        d_model=64,
        nhead=4,
        num_layers=3,
        num_classes=2,
        dropout=0.1,
    )

    # Train the model (for a few epochs for demonstration)
    print("Training the Transformer-based ECG classifier...")
    history = train_transformer_classifier(
        model, x_train, y_train, epochs=10, batch_size=32, learning_rate=0.001
    )

    # Predict on new dummy test data:
    x_test = np.random.rand(10, input_length).astype(np.float32)
    preds = predict_transformer(model, x_test)
    print("Predictions:", preds)
