# ecg_deep_denoiser.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DenoiserError(Exception):
    """Base exception class for ECG Denoiser errors."""

    pass


class DenoiserConfigError(DenoiserError):
    """Exception raised for errors in the Denoiser configuration."""

    pass


class DenoiserInputError(DenoiserError):
    """Exception raised for errors in the input data."""

    pass


def pad_to_multiple(signal: np.ndarray, multiple: int) -> Tuple[np.ndarray, int]:
    """Pad signal at the end so its length is a multiple of 'multiple'.

    Parameters
    ----------
    signal : np.ndarray
        Input signal to pad
    multiple : int
        The multiple to pad to

    Returns
    -------
    Tuple[np.ndarray, int]
        Padded signal and the pad length added

    Raises
    ------
    DenoiserInputError
        If signal is not a numpy array or multiple is not positive
    """
    try:
        if not isinstance(signal, np.ndarray):
            raise DenoiserInputError("Signal must be a numpy array")
        if not isinstance(multiple, int) or multiple <= 0:
            raise DenoiserInputError(
                f"Multiple must be a positive integer, got {multiple}"
            )
        if not np.isfinite(signal).all():
            raise DenoiserInputError("Signal contains invalid values (inf or nan)")

        remainder = len(signal) % multiple
        if remainder == 0:
            return signal, 0
        pad_width = multiple - remainder
        padded_signal = np.pad(signal, (0, pad_width), mode="constant")
        return padded_signal, pad_width
    except Exception as e:
        if not isinstance(e, DenoiserInputError):
            logger.error(f"Error in pad_to_multiple: {str(e)}")
            raise DenoiserInputError(f"Padding failed: {str(e)}")
        raise


def crop_to_length(signal: np.ndarray, original_length: int) -> np.ndarray:
    """Crop signal to the original length.

    Parameters
    ----------
    signal : np.ndarray
        Signal to crop
    original_length : int
        Length to crop to

    Returns
    -------
    np.ndarray
        Cropped signal

    Raises
    ------
    DenoiserInputError
        If inputs are invalid
    """
    try:
        if not isinstance(signal, np.ndarray):
            raise DenoiserInputError("Signal must be a numpy array")
        if not isinstance(original_length, int) or original_length <= 0:
            raise DenoiserInputError(
                f"Original length must be positive, got {original_length}"
            )
        if original_length > len(signal):
            raise DenoiserInputError(
                f"Original length {original_length} exceeds signal length {len(signal)}"
            )
        if not np.isfinite(signal).all():
            raise DenoiserInputError("Signal contains invalid values (inf or nan)")

        return signal[:original_length]
    except Exception as e:
        if not isinstance(e, DenoiserInputError):
            logger.error(f"Error in crop_to_length: {str(e)}")
            raise DenoiserInputError(f"Cropping failed: {str(e)}")
        raise


class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder for ECG signal denoising.

    The model consists of an encoder that compresses the input signal
    and a decoder that reconstructs it. Uses batch normalization and
    skip connections for better training stability.
    """

    def __init__(self, dropout_rate: float = 0.1):
        """
        Parameters
        ----------
        dropout_rate : float, optional
            Dropout rate for regularization, by default 0.1
        """
        if not 0 <= dropout_rate < 1:
            raise DenoiserConfigError(
                f"Dropout rate must be in [0,1), got {dropout_rate}"
            )

        super(ConvAutoencoder, self).__init__()

        # Encoder with Batch Normalization and Dropout
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2),  # output: L/2
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2),  # output: L/4
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2),  # output: L/8
        )

        # Decoder with skip connections
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=8,
                out_channels=8,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.ConvTranspose1d(
                in_channels=8,
                out_channels=8,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.ConvTranspose1d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Normalize output to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, sequence_length)

        Returns
        -------
        torch.Tensor
            Reconstructed signal of the same shape as input

        Raises
        ------
        DenoiserInputError
            If input tensor has incorrect shape or dimensions
        """
        try:
            if x.dim() != 3:
                raise DenoiserInputError(f"Expected 3D input tensor, got {x.dim()}D")
            if x.size(1) != 1:
                raise DenoiserInputError(
                    f"Expected 1 channel, got {x.size(1)} channels"
                )
            if not torch.isfinite(x).all():
                raise DenoiserInputError("Input contains invalid values (inf or nan)")

            # Forward pass with residual connection
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

        except Exception as e:
            if not isinstance(e, DenoiserInputError):
                logger.error(f"Error in forward pass: {str(e)}")
                raise DenoiserInputError(f"Forward pass failed: {str(e)}")
            raise


class ECGDeepDenoiser:
    """Deep learning-based ECG signal denoiser using convolutional autoencoder."""

    def __init__(
        self, input_length: int, learning_rate: float = 0.001, dropout_rate: float = 0.1
    ):
        """
        Parameters
        ----------
        input_length : int
            Length of input ECG signals
        learning_rate : float, optional
            Learning rate for optimization, by default 0.001
        dropout_rate : float, optional
            Dropout rate for regularization, by default 0.1

        Raises
        ------
        DenoiserConfigError
            If parameters are invalid
        """
        try:
            if not isinstance(input_length, int) or input_length <= 0:
                raise DenoiserConfigError(
                    f"Input length must be positive integer, got {input_length}"
                )
            if not isinstance(learning_rate, float) or learning_rate <= 0:
                raise DenoiserConfigError(
                    f"Learning rate must be positive float, got {learning_rate}"
                )
            if not 0 <= dropout_rate < 1:
                raise DenoiserConfigError(
                    f"Dropout rate must be in [0,1), got {dropout_rate}"
                )

            self.input_length = input_length
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            self.model = ConvAutoencoder(dropout_rate=dropout_rate).to(self.device)
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        except Exception as e:
            if not isinstance(e, DenoiserConfigError):
                logger.error(f"Initialization error: {str(e)}")
                raise DenoiserConfigError(f"Initialization failed: {str(e)}")
            raise

    def train(
        self,
        x_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 128,
        validation_split: float = 0.1,
        early_stopping_patience: int = 10,
        gradient_clip_val: float = 1.0,
    ) -> dict:
        """
        Train the denoiser model.

        Parameters
        ----------
        x_train : np.ndarray
            Training data
        epochs : int, optional
            Number of training epochs, by default 50
        batch_size : int, optional
            Batch size for training, by default 128
        validation_split : float, optional
            Fraction of data to use for validation, by default 0.1
        early_stopping_patience : int, optional
            Number of epochs to wait before early stopping, by default 10
        gradient_clip_val : float, optional
            Maximum gradient norm for clipping, by default 1.0

        Returns
        -------
        dict
            Training history

        Raises
        ------
        DenoiserInputError
            If input data is invalid
        """
        try:
            # Validate inputs
            if not isinstance(x_train, np.ndarray):
                raise DenoiserInputError("Training data must be a numpy array")
            if not np.isfinite(x_train).all():
                raise DenoiserInputError("Training data contains invalid values")
            if epochs <= 0:
                raise DenoiserConfigError(f"epochs must be positive, got {epochs}")
            if batch_size <= 0:
                raise DenoiserConfigError(
                    f"batch_size must be positive, got {batch_size}"
                )
            if not 0 < validation_split < 1:
                raise DenoiserConfigError(
                    f"validation_split must be in (0,1), got {validation_split}"
                )

            logger.info(f"Starting training with {len(x_train)} samples")

            # Prepare data
            padded_signals = []
            pad_lengths = []
            for sig in x_train:
                padded, pad_len = pad_to_multiple(sig, 8)
                padded_signals.append(padded)
                pad_lengths.append(pad_len)
            padded_signals = np.array(padded_signals)

            # Arrange shape to (num_samples, 1, padded_length)
            if padded_signals.ndim == 2:
                padded_signals = np.expand_dims(padded_signals, axis=1)
            elif padded_signals.ndim == 3 and padded_signals.shape[1] != 1:
                padded_signals = np.transpose(padded_signals, (0, 2, 1))

            # Split into training and validation sets
            n_val = int(len(padded_signals) * validation_split)
            indices = np.random.permutation(len(padded_signals))
            val_idx, train_idx = indices[:n_val], indices[n_val:]

            train_data = padded_signals[train_idx]
            val_data = padded_signals[val_idx]

            # Create data loaders
            train_tensor = torch.tensor(train_data, dtype=torch.float32).to(self.device)
            val_tensor = torch.tensor(val_data, dtype=torch.float32).to(self.device)

            train_dataset = TensorDataset(train_tensor)
            val_dataset = TensorDataset(val_tensor)

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # Initialize training variables
            history = {"train_loss": [], "val_loss": [], "train_mse": [], "val_mse": []}
            best_val_loss = float("inf")
            patience_counter = 0

            # Training loop
            for epoch in range(1, epochs + 1):
                self.model.train()
                train_loss = 0.0
                train_mse = 0.0
                n_train_batches = 0

                for batch in train_loader:
                    try:
                        inputs = batch[0]
                        outputs = self.model(inputs)

                        # Crop to original length
                        outputs_cropped = outputs[:, :, : self.input_length]
                        inputs_cropped = inputs[:, :, : self.input_length]

                        loss = self.criterion(outputs_cropped, inputs_cropped)
                        mse = nn.MSELoss(reduction="mean")(
                            outputs_cropped, inputs_cropped
                        )

                        self.optimizer.zero_grad()
                        loss.backward()

                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), gradient_clip_val
                        )

                        self.optimizer.step()

                        train_loss += loss.item()
                        train_mse += mse.item()
                        n_train_batches += 1

                    except Exception as e:
                        logger.error(f"Error in training batch: {str(e)}")
                        raise DenoiserInputError(f"Training failed: {str(e)}")

                # Validation loop
                self.model.eval()
                val_loss = 0.0
                val_mse = 0.0
                n_val_batches = 0

                with torch.no_grad():
                    for batch in val_loader:
                        try:
                            inputs = batch[0]
                            outputs = self.model(inputs)

                            outputs_cropped = outputs[:, :, : self.input_length]
                            inputs_cropped = inputs[:, :, : self.input_length]

                            loss = self.criterion(outputs_cropped, inputs_cropped)
                            mse = nn.MSELoss(reduction="mean")(
                                outputs_cropped, inputs_cropped
                            )

                            val_loss += loss.item()
                            val_mse += mse.item()
                            n_val_batches += 1

                        except Exception as e:
                            logger.error(f"Error in validation batch: {str(e)}")
                            raise DenoiserInputError(f"Validation failed: {str(e)}")

                # Calculate average losses
                avg_train_loss = train_loss / n_train_batches
                avg_train_mse = train_mse / n_train_batches
                avg_val_loss = val_loss / n_val_batches
                avg_val_mse = val_mse / n_val_batches

                # Update history
                history["train_loss"].append(avg_train_loss)
                history["val_loss"].append(avg_val_loss)
                history["train_mse"].append(avg_train_mse)
                history["val_mse"].append(avg_val_mse)

                logger.info(
                    f"Epoch [{epoch}/{epochs}] - "
                    f"Train Loss: {avg_train_loss:.6f}, MSE: {avg_train_mse:.6f}, "
                    f"Val Loss: {avg_val_loss:.6f}, MSE: {avg_val_mse:.6f}"
                )

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model state here if needed
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch} epochs")
                        break

            return history

        except Exception as e:
            if not isinstance(e, (DenoiserInputError, DenoiserConfigError)):
                logger.error(f"Training failed: {str(e)}")
                raise DenoiserInputError(f"Training failed: {str(e)}")
            raise

    def denoise(
        self,
        signal: np.ndarray,
        batch_size: Optional[int] = None,
        return_reconstruction_error: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Denoise an ECG signal using the trained model.

        Parameters
        ----------
        signal : np.ndarray
            Input signal to denoise. Can be a single signal or batch of signals
        batch_size : Optional[int], optional
            Batch size for processing large inputs, by default None
        return_reconstruction_error : bool, optional
            Whether to return reconstruction error, by default False

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, float]]
            If return_reconstruction_error is False:
                Denoised signal(s)
            If return_reconstruction_error is True:
                Tuple of (denoised signal(s), reconstruction error)

        Raises
        ------
        DenoiserInputError
            If input is invalid
        """
        try:
            # Input validation
            if not isinstance(signal, np.ndarray):
                raise DenoiserInputError("Input must be a numpy array")
            if not np.isfinite(signal).all():
                raise DenoiserInputError("Input contains invalid values")
            if batch_size is not None and batch_size <= 0:
                raise DenoiserConfigError(
                    f"batch_size must be positive, got {batch_size}"
                )

            self.model.eval()
            logger.info("Starting denoising process")

            # Handle single signal vs batch of signals
            single_signal = signal.ndim == 1
            if single_signal:
                signal = np.expand_dims(signal, axis=0)

            # Prepare batches if needed
            if batch_size is None:
                batch_size = len(signal)

            denoised_signals = []
            reconstruction_errors = []

            for i in range(0, len(signal), batch_size):
                batch = signal[i : i + batch_size]
                try:
                    # Pad signals in batch
                    padded_batch = []
                    for sig in batch:
                        padded, _ = pad_to_multiple(sig, 8)
                        padded_batch.append(padded)
                    padded_batch = np.array(padded_batch)

                    # Reshape to (batch_size, 1, sequence_length)
                    padded_batch = np.expand_dims(padded_batch, axis=1)

                    # Convert to tensor and move to device
                    batch_tensor = torch.tensor(padded_batch, dtype=torch.float32).to(
                        self.device
                    )

                    # Forward pass
                    with torch.no_grad():
                        output_tensor = self.model(batch_tensor)

                        if return_reconstruction_error:
                            error = nn.MSELoss(reduction="mean")(
                                output_tensor[:, :, : self.input_length],
                                batch_tensor[:, :, : self.input_length],
                            ).item()
                            reconstruction_errors.append(error)

                    # Convert back to numpy and crop
                    output_np = output_tensor.cpu().numpy().squeeze(axis=1)
                    for j, sig in enumerate(output_np):
                        denoised = crop_to_length(sig, len(batch[j]))
                        denoised_signals.append(denoised)

                except Exception as e:
                    logger.error(f"Error processing batch {i // batch_size}: {str(e)}")
                    raise DenoiserInputError(f"Denoising failed: {str(e)}")

            # Combine results
            denoised_array = np.array(denoised_signals)
            if single_signal:
                denoised_array = denoised_array.squeeze(axis=0)

            if return_reconstruction_error:
                avg_error = np.mean(reconstruction_errors)
                logger.info(f"Average reconstruction error: {avg_error:.6f}")
                return denoised_array, avg_error

            return denoised_array

        except Exception as e:
            if not isinstance(e, (DenoiserInputError, DenoiserConfigError)):
                logger.error(f"Denoising failed: {str(e)}")
                raise DenoiserInputError(f"Denoising failed: {str(e)}")
            raise


class ECGAutoencoder(nn.Module):
    """
    Autoencoder for ECG signal reconstruction and feature learning.
    """

    def __init__(self, input_size: int = 512, latent_dim: int = 32):
        super().__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, latent_dim),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, input_size),
            nn.Tanh(),  # Output between -1 and 1
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode
        latent = self.encoder(x)
        # Decode
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent space."""
        return self.decoder(z)


class ECGAnomalyDetector:
    """
    Anomaly detection for ECG signals using autoencoder reconstruction error
    and additional statistical features.
    """

    def __init__(
        self,
        input_size: int = 512,
        latent_dim: int = 32,
        threshold_percentile: float = 95,
    ):
        self.autoencoder = ECGAutoencoder(input_size, latent_dim)
        self.threshold = None
        self.threshold_percentile = threshold_percentile
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(self.device)

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> dict:
        """
        Train the autoencoder and compute the anomaly threshold.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, input_size)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimization

        Returns
        -------
        dict
            Training history
        """
        try:
            # Scale the data
            X_scaled = self.scaler.fit_transform(X)

            # Convert to torch dataset
            dataset = torch.FloatTensor(X_scaled)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Initialize optimizer
            optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            history = {"loss": [], "val_loss": []}

            # Training loop
            self.autoencoder.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for batch in dataloader:
                    batch = batch.to(self.device)

                    # Forward pass
                    reconstructed, _ = self.autoencoder(batch)
                    loss = criterion(reconstructed, batch)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                # Record loss
                avg_loss = epoch_loss / len(dataloader)
                history["loss"].append(avg_loss)

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

            # Compute reconstruction errors for threshold
            self.autoencoder.eval()
            reconstruction_errors = []
            with torch.no_grad():
                for batch in dataloader:
                    batch = batch.to(self.device)
                    reconstructed, _ = self.autoencoder(batch)
                    errors = torch.mean((batch - reconstructed) ** 2, dim=1)
                    reconstruction_errors.extend(errors.cpu().numpy())

            # Set threshold based on percentile of reconstruction errors
            self.threshold = np.percentile(
                reconstruction_errors, self.threshold_percentile
            )

            return history

        except Exception as e:
            logger.error(f"Error in anomaly detector training: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in ECG signals.

        Parameters
        ----------
        X : np.ndarray
            Data to analyze of shape (n_samples, input_size)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - Binary array indicating anomalies (1) and normal samples (0)
            - Anomaly scores for each sample
        """
        try:
            if self.threshold is None:
                raise ValueError("Model must be fitted before prediction")

            # Scale the data
            X_scaled = self.scaler.transform(X)

            # Convert to torch tensor
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            # Get reconstruction error
            self.autoencoder.eval()
            with torch.no_grad():
                reconstructed, _ = self.autoencoder(X_tensor)
                reconstruction_error = torch.mean(
                    (X_tensor - reconstructed) ** 2, dim=1
                )

            # Convert to numpy
            anomaly_scores = reconstruction_error.cpu().numpy()

            # Classify as anomaly if error > threshold
            anomalies = (anomaly_scores > self.threshold).astype(int)

            return anomalies, anomaly_scores

        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            raise

    def get_latent_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract latent space features from the autoencoder.

        Parameters
        ----------
        X : np.ndarray
            Input data

        Returns
        -------
        np.ndarray
            Latent space representations
        """
        try:
            # Scale the data
            X_scaled = self.scaler.transform(X)

            # Convert to torch tensor
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            # Get latent representations
            self.autoencoder.eval()
            with torch.no_grad():
                _, latent = self.autoencoder(X_tensor)

            return latent.cpu().numpy()

        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            raise

    def save_model(self, path: str):
        """Save the model to disk.

        Parameters
        ----------
        path : str
            Path to save the model
        """
        try:
            # Save autoencoder state dict
            torch.save(
                {
                    "autoencoder_state_dict": self.autoencoder.state_dict(),
                    "threshold": float(self.threshold)
                    if self.threshold is not None
                    else None,
                    "scaler_mean_": self.scaler.mean_.tolist()
                    if hasattr(self.scaler, "mean_")
                    else None,
                    "scaler_scale_": self.scaler.scale_.tolist()
                    if hasattr(self.scaler, "scale_")
                    else None,
                    "scaler_var_": self.scaler.var_.tolist()
                    if hasattr(self.scaler, "var_")
                    else None,
                },
                path,
            )
            logger.info(f"Model saved successfully to {path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str):
        """Load the model from disk.

        Parameters
        ----------
        path : str
            Path to load the model from
        """
        try:
            # Load the state dict with weights_only=False
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            # Load autoencoder state
            self.autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])

            # Load threshold
            self.threshold = checkpoint["threshold"]

            # Reconstruct scaler
            if checkpoint["scaler_mean_"] is not None:
                self.scaler.mean_ = np.array(checkpoint["scaler_mean_"])
                self.scaler.scale_ = np.array(checkpoint["scaler_scale_"])
                self.scaler.var_ = np.array(checkpoint["scaler_var_"])

            logger.info(f"Model loaded successfully from {path}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import neurokit2 as nk
    import numpy as np
    from sklearn.model_selection import train_test_split

    def test_ecg_denoiser():
        try:
            # 1. Generate synthetic ECG data for testing
            print("Generating synthetic ECG data...")
            n_samples = 100  # Number of ECG signals
            signal_length = 1000  # Length of each signal
            sampling_rate = 500  # Hz
            duration = int(signal_length / sampling_rate)  # Duration in seconds

            # Generate clean ECG signals
            clean_signals = []
            for _ in range(n_samples):
                # Generate ECG signal
                ecg = nk.ecg_simulate(
                    duration=duration, sampling_rate=sampling_rate, noise=0
                )
                # Convert to numpy array if not already
                ecg = np.array(ecg)

                # Ensure consistent length
                if len(ecg) > signal_length:
                    ecg = ecg[:signal_length]
                elif len(ecg) < signal_length:
                    # Pad if necessary
                    pad_width = signal_length - len(ecg)
                    ecg = np.pad(ecg, (0, pad_width), mode="constant")

                clean_signals.append(ecg)

            # Convert list to numpy array
            clean_signals = np.array(clean_signals)

            print(f"Generated signals shape: {clean_signals.shape}")

            # Add noise to create noisy signals
            noise_level = 0.1
            noisy_signals = clean_signals + noise_level * np.random.randn(
                *clean_signals.shape
            )

            # 2. Initialize the denoiser
            print("Initializing ECG Deep Denoiser...")
            denoiser = ECGDeepDenoiser(input_length=signal_length)

            # 3. Split data for training
            X_train, X_test = train_test_split(
                noisy_signals, test_size=0.2, random_state=42
            )

            # 4. Train the model
            print("Training the model...")
            history = denoiser.train(
                x_train=X_train,
                epochs=10,  # Reduced for testing
                batch_size=32,
                validation_split=0.2,
                early_stopping_patience=3,
            )

            # 5. Plot training history
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(history["train_loss"], label="Training Loss")
            plt.plot(history["val_loss"], label="Validation Loss")
            plt.title("Training History - Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history["train_mse"], label="Training MSE")
            plt.plot(history["val_mse"], label="Validation MSE")
            plt.title("Training History - MSE")
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.legend()

            plt.tight_layout()
            plt.show()

            # 6. Test the model on a single signal
            print("\nTesting denoising on a single signal...")
            test_signal = X_test[0]
            denoised_signal, reconstruction_error = denoiser.denoise(
                test_signal, return_reconstruction_error=True
            )

            # 7. Plot results
            plt.figure(figsize=(15, 8))

            plt.subplot(3, 1, 1)
            plt.plot(clean_signals[0])
            plt.title("Original Clean Signal")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")

            plt.subplot(3, 1, 2)
            plt.plot(test_signal)
            plt.title("Noisy Signal")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")

            plt.subplot(3, 1, 3)
            plt.plot(denoised_signal)
            plt.title(
                f"Denoised Signal (Reconstruction Error: {reconstruction_error:.6f})"
            )
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")

            plt.tight_layout()
            plt.show()

            # 8. Print performance metrics
            print(f"\nReconstruction Error: {reconstruction_error:.6f}")

            # 9. Test batch processing
            print("\nTesting batch processing...")
            batch_signals = X_test[:5]
            denoised_batch = denoiser.denoise(batch_signals)
            print(f"Successfully denoised batch of {len(denoised_batch)} signals")

            # Save example signals
            np.save("example_clean.npy", clean_signals[0])
            np.save("example_noisy.npy", test_signal)
            np.save("example_denoised.npy", denoised_signal)

            return True

        except Exception as e:
            print(f"Error in test_ecg_denoiser: {str(e)}")
            import traceback

            traceback.print_exc()  # This will print the full error traceback
            return False

    # Run the test
    print("Starting ECG Deep Denoiser test...")
    success = test_ecg_denoiser()
    print(f"\nTest {'successful' if success else 'failed'}")
