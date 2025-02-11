# ecg_deep_denoiser_torch.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Custom exceptions
class DenoiserError(Exception):
    """Base exception class for ECG Denoiser errors."""
    pass

class DenoiserConfigError(DenoiserError):
    """Exception raised for errors in the Denoiser configuration."""
    pass

class DenoiserInputError(DenoiserError):
    """Exception raised for errors in the input data."""
    pass


def pad_to_multiple(signal: Union[np.ndarray, torch.Tensor], multiple: int) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """
    Pad a 1D signal so that its length is a multiple of 'multiple'.

    If the input is a NumPy array, we use np.pad. If it is a torch.Tensor,
    we use torch.nn.functional.pad.

    Parameters
    ----------
    signal : np.ndarray or torch.Tensor
        Input signal.
    multiple : int
        The target multiple for the length.

    Returns
    -------
    Tuple[signal, int]
        The padded signal and the pad length added.

    Raises
    ------
    DenoiserInputError
        If inputs are invalid.
    """
    try:
        # Check type and validity
        if not (isinstance(signal, np.ndarray) or isinstance(signal, torch.Tensor)):
            raise DenoiserInputError("Signal must be a numpy array or torch.Tensor")
        if not isinstance(multiple, int) or multiple <= 0:
            raise DenoiserInputError(f"Multiple must be a positive integer, got {multiple}")

        # Work with NumPy if needed:
        if isinstance(signal, np.ndarray):
            if not np.isfinite(signal).all():
                raise DenoiserInputError("Signal contains invalid values (inf or nan)")
            length = signal.shape[0]
            remainder = length % multiple
            if remainder == 0:
                return signal, 0
            pad_width = multiple - remainder
            padded_signal = np.pad(signal, (0, pad_width), mode="constant")
            return padded_signal, pad_width
        else:
            # signal is a torch.Tensor
            if not torch.isfinite(signal).all().item():
                raise DenoiserInputError("Signal contains invalid values (inf or nan)")
            length = signal.size(0)
            remainder = length % multiple
            if remainder == 0:
                return signal, 0
            pad_width = multiple - remainder
            # F.pad expects the padding tuple (pad_left, pad_right)
            padded_signal = F.pad(signal.unsqueeze(0), (0, pad_width), mode="constant", value=0).squeeze(0)
            return padded_signal, pad_width

    except Exception as e:
        logger.error(f"Error in pad_to_multiple: {str(e)}")
        raise DenoiserInputError(f"Padding failed: {str(e)}")


def crop_to_length(signal: Union[np.ndarray, torch.Tensor], original_length: int) -> Union[np.ndarray, torch.Tensor]:
    """
    Crop a signal to the given original length.

    Parameters
    ----------
    signal : np.ndarray or torch.Tensor
        Signal to crop.
    original_length : int
        Target length.

    Returns
    -------
    Cropped signal.

    Raises
    ------
    DenoiserInputError
        If inputs are invalid.
    """
    try:
        if not (isinstance(signal, np.ndarray) or isinstance(signal, torch.Tensor)):
            raise DenoiserInputError("Signal must be a numpy array or torch.Tensor")
        if not isinstance(original_length, int) or original_length <= 0:
            raise DenoiserInputError(f"Original length must be positive, got {original_length}")

        length = signal.shape[0] if isinstance(signal, np.ndarray) else signal.size(0)
        if original_length > length:
            raise DenoiserInputError(f"Original length {original_length} exceeds signal length {length}")

        if isinstance(signal, np.ndarray):
            if not np.isfinite(signal).all():
                raise DenoiserInputError("Signal contains invalid values (inf or nan)")
            return signal[:original_length]
        else:
            if not torch.isfinite(signal).all().item():
                raise DenoiserInputError("Signal contains invalid values (inf or nan)")
            return signal[:original_length]

    except Exception as e:
        logger.error(f"Error in crop_to_length: {str(e)}")
        raise DenoiserInputError(f"Cropping failed: {str(e)}")


# **********************
# Model definitions
# **********************

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for ECG signal denoising.

    It compresses the input via an encoder and then reconstructs using a decoder.
    Batch normalization, dropout, and skip connections improve training.
    """
    def __init__(self, dropout_rate: float = 0.1):
        if not 0 <= dropout_rate < 1:
            raise DenoiserConfigError(f"Dropout rate must be in [0,1), got {dropout_rate}")
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2),  # length becomes L/2
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2),  # length becomes L/4
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2),  # length becomes L/8
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=8, out_channels=8,
                kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.ConvTranspose1d(
                in_channels=8, out_channels=8,
                kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.ConvTranspose1d(
                in_channels=8, out_channels=16,
                kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # outputs in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            if x.dim() != 3:
                raise DenoiserInputError(f"Expected 3D input tensor, got {x.dim()}D")
            if x.size(1) != 1:
                raise DenoiserInputError(f"Expected 1 channel, got {x.size(1)} channels")
            if not torch.isfinite(x).all():
                raise DenoiserInputError("Input contains invalid values (inf or nan)")

            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise DenoiserInputError(f"Forward pass failed: {str(e)}")


class ECGDeepDenoiser:
    """
    Deep-learning ECG signal denoiser using a convolutional autoencoder.
    """
    def __init__(self, input_length: int, learning_rate: float = 0.001, dropout_rate: float = 0.1):
        try:
            if not isinstance(input_length, int) or input_length <= 0:
                raise DenoiserConfigError(f"Input length must be positive integer, got {input_length}")
            if not isinstance(learning_rate, float) or learning_rate <= 0:
                raise DenoiserConfigError(f"Learning rate must be positive float, got {learning_rate}")
            if not 0 <= dropout_rate < 1:
                raise DenoiserConfigError(f"Dropout rate must be in [0,1), got {dropout_rate}")

            self.input_length = input_length
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            self.model = ConvAutoencoder(dropout_rate=dropout_rate).to(self.device)
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise DenoiserConfigError(f"Initialization failed: {str(e)}")

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
        Train the autoencoder.

        x_train is expected as a numpy array of signals. We pad each signal so its length
        becomes a multiple (here, 8), then reshape to (samples, 1, length) and convert to torch tensors.
        """
        try:
            if not isinstance(x_train, np.ndarray):
                raise DenoiserInputError("Training data must be a numpy array")
            if not np.isfinite(x_train).all():
                raise DenoiserInputError("Training data contains invalid values")
            if epochs <= 0:
                raise DenoiserConfigError(f"epochs must be positive, got {epochs}")
            if batch_size <= 0:
                raise DenoiserConfigError(f"batch_size must be positive, got {batch_size}")
            if not 0 < validation_split < 1:
                raise DenoiserConfigError(f"validation_split must be in (0,1), got {validation_split}")

            logger.info(f"Starting training with {len(x_train)} samples")

            padded_signals = []
            for sig in x_train:
                padded, _ = pad_to_multiple(sig, 8)
                padded_signals.append(padded)
            padded_signals = np.array(padded_signals)

            # Reshape to (num_samples, 1, padded_length)
            if padded_signals.ndim == 2:
                padded_signals = np.expand_dims(padded_signals, axis=1)
            elif padded_signals.ndim == 3 and padded_signals.shape[1] != 1:
                padded_signals = np.transpose(padded_signals, (0, 2, 1))

            n_val = int(len(padded_signals) * validation_split)
            indices = np.random.permutation(len(padded_signals))
            val_idx, train_idx = indices[:n_val], indices[n_val:]
            train_data = padded_signals[train_idx]
            val_data = padded_signals[val_idx]

            train_tensor = torch.tensor(train_data, dtype=torch.float32).to(self.device)
            val_tensor = torch.tensor(val_data, dtype=torch.float32).to(self.device)

            train_dataset = TensorDataset(train_tensor)
            val_dataset = TensorDataset(val_tensor)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            history = {"train_loss": [], "val_loss": [], "train_mse": [], "val_mse": []}
            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(1, epochs + 1):
                self.model.train()
                train_loss = 0.0
                train_mse = 0.0
                n_train_batches = 0

                for batch in train_loader:
                    inputs = batch[0]
                    outputs = self.model(inputs)

                    # Crop to original length along the time axis
                    outputs_cropped = outputs[:, :, :self.input_length]
                    inputs_cropped = inputs[:, :, :self.input_length]

                    loss = self.criterion(outputs_cropped, inputs_cropped)
                    mse = nn.MSELoss(reduction="mean")(outputs_cropped, inputs_cropped)

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)
                    self.optimizer.step()

                    train_loss += loss.item()
                    train_mse += mse.item()
                    n_train_batches += 1

                # Validation loop
                self.model.eval()
                val_loss = 0.0
                val_mse = 0.0
                n_val_batches = 0

                with torch.no_grad():
                    for batch in val_loader:
                        inputs = batch[0]
                        outputs = self.model(inputs)

                        outputs_cropped = outputs[:, :, :self.input_length]
                        inputs_cropped = inputs[:, :, :self.input_length]

                        loss = self.criterion(outputs_cropped, inputs_cropped)
                        mse = nn.MSELoss(reduction="mean")(outputs_cropped, inputs_cropped)
                        val_loss += loss.item()
                        val_mse += mse.item()
                        n_val_batches += 1

                avg_train_loss = train_loss / n_train_batches
                avg_train_mse = train_mse / n_train_batches
                avg_val_loss = val_loss / n_val_batches
                avg_val_mse = val_mse / n_val_batches

                history["train_loss"].append(avg_train_loss)
                history["val_loss"].append(avg_val_loss)
                history["train_mse"].append(avg_train_mse)
                history["val_mse"].append(avg_val_mse)

                logger.info(
                    f"Epoch [{epoch}/{epochs}] - "
                    f"Train Loss: {avg_train_loss:.6f}, MSE: {avg_train_mse:.6f}, "
                    f"Val Loss: {avg_val_loss:.6f}, MSE: {avg_val_mse:.6f}"
                )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Optionally, save best model state here.
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch} epochs")
                        break

            return history

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise DenoiserInputError(f"Training failed: {str(e)}")


    def denoise(
        self,
        signal: np.ndarray,
        batch_size: Optional[int] = None,
        return_reconstruction_error: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Denoise an ECG signal (or batch) using the trained model.

        signal can be a single 1D numpy array or a batch (2D numpy array). We pad each
        signal (if necessary), reshape to (batch, 1, length), convert to a torch tensor,
        and run the forward pass.
        """
        try:
            if not isinstance(signal, np.ndarray):
                raise DenoiserInputError("Input must be a numpy array")
            if not np.isfinite(signal).all():
                raise DenoiserInputError("Input contains invalid values")
            if batch_size is not None and batch_size <= 0:
                raise DenoiserConfigError(f"batch_size must be positive, got {batch_size}")

            self.model.eval()
            logger.info("Starting denoising process")
            single_signal = signal.ndim == 1
            if single_signal:
                signal = np.expand_dims(signal, axis=0)

            if batch_size is None:
                batch_size = len(signal)

            denoised_signals = []
            reconstruction_errors = []

            for i in range(0, len(signal), batch_size):
                batch = signal[i : i + batch_size]
                padded_batch = []
                for sig in batch:
                    padded, _ = pad_to_multiple(sig, 8)
                    padded_batch.append(padded)
                padded_batch = np.array(padded_batch)
                padded_batch = np.expand_dims(padded_batch, axis=1)

                batch_tensor = torch.tensor(padded_batch, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    output_tensor = self.model(batch_tensor)
                    if return_reconstruction_error:
                        error = nn.MSELoss(reduction="mean")(
                            output_tensor[:, :, :self.input_length], batch_tensor[:, :, :self.input_length]
                        ).item()
                        reconstruction_errors.append(error)

                output_np = output_tensor.cpu().numpy().squeeze(axis=1)
                for j, sig in enumerate(output_np):
                    denoised = crop_to_length(sig, len(batch[j]))
                    denoised_signals.append(denoised)

            denoised_array = np.array(denoised_signals)
            if single_signal:
                denoised_array = denoised_array.squeeze(axis=0)
            if return_reconstruction_error:
                avg_error = np.mean(reconstruction_errors)
                logger.info(f"Average reconstruction error: {avg_error:.6f}")
                return denoised_array, avg_error
            return denoised_array

        except Exception as e:
            logger.error(f"Denoising failed: {str(e)}")
            raise DenoiserInputError(f"Denoising failed: {str(e)}")


# ******************************
# Additional autoencoder for anomaly detection
# ******************************

class ECGAutoencoder(nn.Module):
    """
    Fully connected autoencoder for ECG signal reconstruction and feature learning.
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
            nn.Tanh(),  # output scaled between -1 and 1
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class ECGAnomalyDetector:
    """
    Anomaly detection based on an autoencoder.
    """
    def __init__(self, input_size: int = 512, latent_dim: int = 32, threshold_percentile: float = 95):
        self.autoencoder = ECGAutoencoder(input_size, latent_dim)
        self.threshold = None
        self.threshold_percentile = threshold_percentile
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(self.device)

    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001) -> dict:
        """
        Train the autoencoder and compute an anomaly threshold from reconstruction error.
        """
        try:
            # Scale data
            X_scaled = self.scaler.fit_transform(X)
            dataset = torch.FloatTensor(X_scaled)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            history = {"loss": [], "val_loss": []}

            self.autoencoder.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for batch in dataloader:
                    batch = batch.to(self.device)
                    reconstructed, _ = self.autoencoder(batch)
                    loss = criterion(reconstructed, batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(dataloader)
                history["loss"].append(avg_loss)
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

            # Compute reconstruction error to set threshold
            self.autoencoder.eval()
            reconstruction_errors = []
            with torch.no_grad():
                for batch in dataloader:
                    batch = batch.to(self.device)
                    reconstructed, _ = self.autoencoder(batch)
                    errors = torch.mean((batch - reconstructed) ** 2, dim=1)
                    reconstruction_errors.extend(errors.cpu().numpy())
            self.threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
            return history

        except Exception as e:
            logger.error(f"Error in anomaly detector training: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies based on reconstruction error.
        Returns a binary array (1=anomaly, 0=normal) and the error for each sample.
        """
        try:
            if self.threshold is None:
                raise ValueError("Model must be fitted before prediction")
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            self.autoencoder.eval()
            with torch.no_grad():
                reconstructed, _ = self.autoencoder(X_tensor)
                reconstruction_error = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            anomaly_scores = reconstruction_error.cpu().numpy()
            anomalies = (anomaly_scores > self.threshold).astype(int)
            return anomalies, anomaly_scores

        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            raise

    def get_latent_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract latent representations from the autoencoder.
        """
        try:
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            self.autoencoder.eval()
            with torch.no_grad():
                _, latent = self.autoencoder(X_tensor)
            return latent.cpu().numpy()
        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            raise

    def save_model(self, path: str):
        """
        Save the autoencoder state along with scaler parameters and threshold.
        """
        try:
            torch.save({
                "autoencoder_state_dict": self.autoencoder.state_dict(),
                "threshold": float(self.threshold) if self.threshold is not None else None,
                "scaler_mean_": self.scaler.mean_.tolist() if hasattr(self.scaler, "mean_") else None,
                "scaler_scale_": self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") else None,
                "scaler_var_": self.scaler.var_.tolist() if hasattr(self.scaler, "var_") else None,
            }, path)
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str):
        """
        Load the model state, threshold, and scaler parameters.
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
            self.threshold = checkpoint["threshold"]
            if checkpoint["scaler_mean_"] is not None:
                self.scaler.mean_ = np.array(checkpoint["scaler_mean_"])
                self.scaler.scale_ = np.array(checkpoint["scaler_scale_"])
                self.scaler.var_ = np.array(checkpoint["scaler_var_"])
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


# *************************
# Testing the deep denoiser
# *************************

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import neurokit2 as nk
    from sklearn.model_selection import train_test_split

    def test_ecg_denoiser():
        try:
            print("Generating synthetic ECG data...")
            n_samples = 100       # number of signals
            signal_length = 1000  # length of each signal
            sampling_rate = 500   # Hz

            # Use an integer duration so that NeuroKit2 slicing works correctly
            duration = int(signal_length / sampling_rate)  # e.g., 2 seconds

            clean_signals = []
            for _ in range(n_samples):
                ecg = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, noise=0)
                ecg = np.array(ecg)
                # Ensure fixed length:
                if len(ecg) > signal_length:
                    ecg = ecg[:signal_length]
                elif len(ecg) < signal_length:
                    ecg = np.pad(ecg, (0, signal_length - len(ecg)), mode="constant")
                clean_signals.append(ecg)
            clean_signals = np.array(clean_signals)
            print(f"Generated signals shape: {clean_signals.shape}")

            # Create noisy signals:
            noise_level = 0.1
            noisy_signals = clean_signals + noise_level * np.random.randn(*clean_signals.shape)

            print("Initializing ECG Deep Denoiser...")
            denoiser = ECGDeepDenoiser(input_length=signal_length)

            X_train, X_test = train_test_split(noisy_signals, test_size=0.2, random_state=42)

            print("Training the model...")
            history = denoiser.train(
                x_train=X_train,
                epochs=10,  # reduced for testing
                batch_size=32,
                validation_split=0.2,
                early_stopping_patience=3,
            )

            # Plot training history
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

            print("\nTesting denoising on a single signal...")
            test_signal = X_test[0]
            denoised_signal, reconstruction_error = denoiser.denoise(test_signal, return_reconstruction_error=True)

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
            plt.title(f"Denoised Signal (Reconstruction Error: {reconstruction_error:.6f})")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            plt.show()

            print(f"\nReconstruction Error: {reconstruction_error:.6f}")

            print("\nTesting batch processing...")
            batch_signals = X_test[:5]
            denoised_batch = denoiser.denoise(batch_signals)
            print(f"Successfully denoised batch of {len(denoised_batch)} signals")

            # Optionally save example signals
            np.save("example_clean.npy", clean_signals[0])
            np.save("example_noisy.npy", test_signal)
            np.save("example_denoised.npy", denoised_signal)

            return True

        except Exception as e:
            print(f"Error in test_ecg_denoiser: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    print("Starting ECG Deep Denoiser test...")
    success = test_ecg_denoiser()
    print(f"\nTest {'successful' if success else 'failed'}")