import torch
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Dict
import numpy as np
import pywt

# We still use SciPy for some digital filter design and filtering.
from scipy.signal import medfilt, savgol_filter, butter, filtfilt
# For median absolute deviation – note that here we use the NumPy version.
from scipy.stats import median_abs_deviation as mad

try:
    from PyEMD import EMD
except ImportError:
    raise ImportError("Please install PyEMD with `pip install EMD-signal`")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_signal(signal: torch.Tensor, min_length: int = 2) -> None:
    """
    Check that signal is a 1D torch.Tensor of enough length and finite values.
    """
    if not isinstance(signal, torch.Tensor):
        raise TypeError("Signal must be a torch.Tensor")
    if signal.dim() != 1:
        raise ValueError(f"Signal must be 1-dimensional, got {signal.dim()} dimensions")
    if signal.size(0) < min_length:
        raise ValueError(f"Signal length must be at least {min_length}")
    if not torch.isfinite(signal).all().item():
        raise ValueError("Signal contains NaN or infinite values")


def wavelet_denoise(
    signal: torch.Tensor,
    wavelet: str = "db4",
    level: Optional[int] = None,
    mode: str = "soft",
) -> torch.Tensor:
    """
    Denoise using wavelet thresholding.

    Note: The wavelet decomposition and reconstruction are still carried out by PyWavelets,
    which works on NumPy arrays. We convert the incoming torch.Tensor to NumPy,
    process it, and then convert the result back to a torch.Tensor.
    """
    try:
        validate_signal(signal)
        if mode not in ["soft", "hard"]:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'soft' or 'hard'")

        # Move the input to CPU and convert to NumPy.
        sig_np = signal.cpu().numpy()

        # If level is not provided, compute the maximum level.
        if level is None:
            level = pywt.dwt_max_level(len(sig_np), pywt.Wavelet(wavelet).dec_len)
        elif level <= 0:
            raise ValueError(f"Level must be positive, got {level}")

        coeffs = pywt.wavedec(sig_np, wavelet, level=level)

        # Estimate the noise sigma using the median absolute deviation.
        sigma = mad(coeffs[-1]) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(sig_np)))

        denoised_coeffs = []
        for i, coeff in enumerate(coeffs):
            if i == 0:  # Keep the approximation coefficients untouched.
                denoised_coeffs.append(coeff)
            else:
                if mode == "soft":
                    magnitude = np.abs(coeff)
                    mask = magnitude > threshold
                    coeff_thresh = np.zeros_like(coeff)
                    # Apply soft thresholding
                    coeff_thresh[mask] = coeff[mask] * np.maximum(
                        0, 1 - threshold / magnitude[mask]
                    )
                else:  # hard thresholding
                    coeff_thresh = pywt.threshold(coeff, threshold, mode="hard")
                denoised_coeffs.append(coeff_thresh)

        denoised_np = pywt.waverec(denoised_coeffs, wavelet)
        return torch.tensor(denoised_np, device=signal.device, dtype=signal.dtype)

    except Exception as e:
        logger.error(f"Error in wavelet_denoise: {str(e)}")
        raise


def adaptive_lms_filter(
    desired: torch.Tensor, reference: torch.Tensor, mu: float = 0.01, filter_order: int = 32
) -> torch.Tensor:
    """
    Adaptive Least Mean Squares filtering using torch operations.
    Processes the data in chunks for efficiency.
    """
    try:
        validate_signal(desired)
        validate_signal(reference)

        if desired.size(0) != reference.size(0):
            raise ValueError("Desired and reference signals must have the same length")

        if not 0 < mu < 1:
            raise ValueError(f"Learning rate mu must be between 0 and 1, got {mu}")

        if not isinstance(filter_order, int) or filter_order <= 0:
            raise ValueError(f"Filter order must be a positive integer, got {filter_order}")

        if filter_order >= desired.size(0):
            raise ValueError("Filter order must be less than signal length")

        n_samples = desired.size(0)
        weights = torch.zeros(filter_order, device=desired.device, dtype=desired.dtype)
        output = torch.zeros(n_samples, device=desired.device, dtype=desired.dtype)

        chunk_size = 1000
        for i in range(filter_order, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            # Create a list of indices for the current chunk.
            chunk_indices = torch.arange(i, end, device=desired.device)

            # For each index j, create an input vector from reference[j-filter_order:j] in reverse order.
            X = torch.stack([reference[j - filter_order : j].flip(0) for j in chunk_indices])

            # Compute the output for the chunk.
            Y = torch.matmul(X, weights)
            output[chunk_indices] = Y

            errors = desired[chunk_indices] - Y
            weights = weights + mu * (X.t() @ errors) / errors.numel()

        # Return the error (noise estimate removed from the original signal)
        return desired - output

    except Exception as e:
        logger.error(f"Error in adaptive_lms_filter: {str(e)}")
        raise


def emd_denoise(signal: torch.Tensor, imf_to_remove: int = 1) -> torch.Tensor:
    """
    Denoise the ECG signal using Empirical Mode Decomposition.
    The EMD decomposition is performed with PyEMD (which works on NumPy arrays).
    """
    try:
        validate_signal(signal)

        if not isinstance(imf_to_remove, int) or imf_to_remove < 0:
            raise ValueError(
                f"imf_to_remove must be a non-negative integer, got {imf_to_remove}"
            )

        sig_np = signal.cpu().numpy()
        emd = EMD()
        imfs = emd.emd(sig_np)

        if imfs is None or imfs.shape[0] == 0:
            logger.warning("EMD decomposition failed, returning original signal")
            return signal

        if imfs.shape[0] > imf_to_remove:
            denoised_np = np.sum(imfs[imf_to_remove:], axis=0)
            reconstruction_error = np.mean(np.abs(sig_np - denoised_np))
            if reconstruction_error > 0.1 * np.std(sig_np):
                logger.warning(
                    f"High reconstruction error in EMD: {reconstruction_error:.2e}"
                )
        else:
            logger.warning(
                f"Not enough IMFs ({imfs.shape[0]}) to remove {imf_to_remove}, returning original signal"
            )
            denoised_np = sig_np

        return torch.tensor(denoised_np, device=signal.device, dtype=signal.dtype)

    except Exception as e:
        logger.error(f"Error in emd_denoise: {str(e)}")
        raise


def median_filter_signal(signal: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Apply a median filter on the signal using torch.
    The sliding window is created via the tensor.unfold() method.
    """
    try:
        validate_signal(signal)

        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError(
                f"kernel_size must be a positive integer, got {kernel_size}"
            )

        if kernel_size % 2 == 0:
            kernel_size += 1
            logger.warning(f"Adjusted even kernel_size to {kernel_size}")

        if kernel_size >= signal.size(0):
            raise ValueError("kernel_size must be less than signal length")

        pad = kernel_size // 2
        # Use reflection padding to handle the edges.
        sig_padded = F.pad(signal.unsqueeze(0).unsqueeze(0), (pad, pad), mode="reflect")
        sig_padded = sig_padded.squeeze(0).squeeze(0)

        # Unfold to create sliding windows.
        windows = sig_padded.unfold(0, kernel_size, 1)  # shape: (L, kernel_size)
        filtered = windows.median(dim=1).values
        return filtered

    except Exception as e:
        logger.error(f"Error in median_filter_signal: {str(e)}")
        raise


def smooth_signal(signal: torch.Tensor, window_length: int = 51, polyorder: int = 3) -> torch.Tensor:
    """
    Smooth the signal with a Savitzky–Golay filter implemented in PyTorch.
    The filter coefficients are computed from a Vandermonde matrix and applied via conv1d.
    """
    try:
        validate_signal(signal)

        if not isinstance(window_length, int) or window_length <= 0:
            raise ValueError(
                f"window_length must be a positive integer, got {window_length}"
            )

        if not isinstance(polyorder, int) or polyorder < 0:
            raise ValueError(
                f"polyorder must be a non-negative integer, got {polyorder}"
            )

        if polyorder >= window_length:
            raise ValueError("polyorder must be less than window_length")

        if window_length % 2 == 0:
            window_length += 1
            logger.warning(f"Adjusted even window_length to {window_length}")

        max_window = signal.size(0) if signal.size(0) % 2 == 1 else signal.size(0) - 1
        if window_length > max_window:
            window_length = max_window
            logger.warning(f"Adjusted window_length to signal length: {window_length}")

        half_window = window_length // 2
        # Create a vector of indices centered about zero.
        ind = torch.arange(-half_window, half_window + 1, dtype=signal.dtype, device=signal.device)
        # Build a Vandermonde matrix (each column is exponentiated index).
        A = torch.stack([ind**i for i in range(polyorder + 1)], dim=1)
        # Compute the pseudoinverse.
        A_pinv = torch.pinverse(A)
        # The coefficients for the smoothing filter correspond to the projection for the 0th derivative.
        coeffs = A_pinv[0]
        # Flip coefficients because conv1d performs a correlation.
        coeffs = coeffs.flip(0).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, window_length)

        pad = half_window
        sig_reshaped = signal.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, L)
        sig_padded = F.pad(sig_reshaped, (pad, pad), mode="reflect")
        smoothed = F.conv1d(sig_padded, coeffs)
        return smoothed.squeeze(0).squeeze(0)

    except Exception as e:
        logger.error(f"Error in smooth_signal: {str(e)}")
        raise


def remove_respiratory_noise(signal: torch.Tensor, fs: float, resp_freq_range: Tuple[float, float] = (0.15, 0.4)) -> torch.Tensor:
    try:
        validate_signal(signal)
        nyquist = fs / 2
        low, high = resp_freq_range
        b, a = butter(4, [low / nyquist, high / nyquist], btype="bandstop")
        sig_np = signal.cpu().numpy()
        filtered = filtfilt(b, a, sig_np)
        return torch.tensor(filtered.copy(), device=signal.device, dtype=signal.dtype)
    except Exception as e:
        logger.error(f"Error in respiratory noise removal: {str(e)}")
        raise


def remove_emg_noise(signal: torch.Tensor, fs: float, emg_freq_threshold: float = 20.0) -> torch.Tensor:
    try:
        validate_signal(signal)
        denoised = wavelet_denoise(signal, wavelet="sym4", level=4, mode="soft")

        nyquist = fs / 2
        b, a = butter(4, emg_freq_threshold / nyquist, btype="low")
        denoised_np = denoised.cpu().numpy()
        filtered = filtfilt(b, a, denoised_np)
        return torch.tensor(filtered.copy(), device=signal.device, dtype=signal.dtype)
    except Exception as e:
        logger.error(f"Error in EMG noise removal: {str(e)}")
        raise


def remove_eda_noise(signal: torch.Tensor, fs: float, eda_freq_range: Tuple[float, float] = (0.01, 1.0)) -> torch.Tensor:
    try:
        validate_signal(signal)
        nyquist = fs / 2
        low, high = eda_freq_range
        b, a = butter(4, [low / nyquist, high / nyquist], btype="bandstop")
        sig_np = signal.cpu().numpy()
        filtered = filtfilt(b, a, sig_np)
        return torch.tensor(filtered.copy(), device=signal.device, dtype=signal.dtype)
    except Exception as e:
        logger.error(f"Error in EDA noise removal: {str(e)}")
        raise


def advanced_denoise_pipeline(
    signal: torch.Tensor,
    fs: float,
    remove_resp: bool = True,
    remove_emg: bool = True,
    remove_eda: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Run the full denoising pipeline sequentially.
    Returns a dictionary with intermediate results.
    """
    try:
        validate_signal(signal)
        result = {"original": signal.clone()}
        current_signal = signal.clone()

        if remove_resp:
            current_signal = remove_respiratory_noise(current_signal, fs)
            result["resp_removed"] = current_signal.clone()

        if remove_emg:
            current_signal = remove_emg_noise(current_signal, fs)
            result["emg_removed"] = current_signal.clone()

        if remove_eda:
            current_signal = remove_eda_noise(current_signal, fs)
            result["eda_removed"] = current_signal.clone()

        result["denoised"] = current_signal
        return result
    except Exception as e:
        logger.error(f"Error in denoising pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create a simulated ECG-like signal.
    fs = 500  # Sampling frequency in Hz.
    t = np.linspace(0, 1, fs, endpoint=False)
    freq = 5  # Frequency for the simulated sine wave.
    clean_signal_np = np.sin(2 * np.pi * freq * t)
    noise_np = np.random.normal(0, 0.2, size=clean_signal_np.shape)
    noisy_signal_np = clean_signal_np + noise_np

    # Convert signals to torch.Tensor and send to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_signal_torch = torch.tensor(
        clean_signal_np, device=device, dtype=torch.float32
    )
    noisy_signal_torch = torch.tensor(
        noisy_signal_np, device=device, dtype=torch.float32
    )

    # Test each processing step.
    denoised_wavelet = wavelet_denoise(
        noisy_signal_torch, wavelet="db4", level=4, mode="soft"
    )

    noise_reference = torch.tensor(
        np.random.normal(0, 0.2, size=clean_signal_np.shape), device=device, dtype=torch.float32
    )
    denoised_adaptive = adaptive_lms_filter(
        noisy_signal_torch, noise_reference, mu=0.01, filter_order=32
    )

    denoised_emd = emd_denoise(noisy_signal_torch, imf_to_remove=1)

    denoised_median = median_filter_signal(noisy_signal_torch, kernel_size=5)
    denoised_smoothing = smooth_signal(noisy_signal_torch, window_length=51, polyorder=3)

    denoised_resp = remove_respiratory_noise(noisy_signal_torch, fs)
    denoised_emg = remove_emg_noise(noisy_signal_torch, fs)
    denoised_eda = remove_eda_noise(noisy_signal_torch, fs)

    denoised_pipeline = advanced_denoise_pipeline(noisy_signal_torch, fs)

    # Plot the results.
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.plot(t, noisy_signal_torch.cpu().numpy(), label="Noisy Signal", color="gray")
    plt.plot(
        t,
        clean_signal_torch.cpu().numpy(),
        label="Clean Signal",
        linestyle="--",
        color="blue",
    )
    plt.title("Original vs. Clean Signal")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(t, denoised_wavelet.cpu().numpy(), label="Wavelet Denoised", color="green")
    plt.title("Wavelet Denoise")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(t, denoised_adaptive.cpu().numpy(), label="Adaptive LMS Denoised", color="purple")
    plt.title("Adaptive LMS Filter")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(t, denoised_emd.cpu().numpy(), label="EMD Denoised", color="orange")
    plt.title("EMD Denoise")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(t, denoised_median.cpu().numpy(), label="Median Filter", color="red")
    plt.title("Median Filter")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(t, denoised_smoothing.cpu().numpy(), label="Savitzky–Golay Denoise", color="brown")
    plt.title("Savitzky–Golay Smoothing")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()