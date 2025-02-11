from typing import Optional, Tuple, Dict
import logging
import numpy as np
from scipy.signal import medfilt, savgol_filter, butter, filtfilt
import pywt

# Import mad from scipy.stats if available, otherwise define our own
try:
    from scipy.stats import median_abs_deviation as mad
except ImportError:

    def mad(data, axis=None):
        """Compute the Median Absolute Deviation (MAD) along the specified axis."""
        median = np.median(data, axis=axis)
        if axis is not None:
            median = np.expand_dims(median, axis)
        return np.median(np.abs(data - median), axis=axis)


# Third-party imports with error handling
try:
    import pywt
except ImportError:
    raise ImportError("Please install PyWavelets with `pip install PyWavelets`")

try:
    from PyEMD import EMD
except ImportError:
    raise ImportError("Please install PyEMD with `pip install EMD-signal`")

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_signal(signal: np.ndarray, min_length: int = 2) -> None:
    """Validate input signal array.

    Parameters:
    -----------
    signal : np.ndarray
        Input signal to validate
    min_length : int
        Minimum required length of signal

    Raises:
    -------
    ValueError
        If signal validation fails
    TypeError
        If signal is not a numpy array
    """
    try:
        if not isinstance(signal, np.ndarray):
            raise TypeError("Signal must be a numpy array")

        if signal.ndim != 1:
            raise ValueError(
                f"Signal must be 1-dimensional, got {signal.ndim} dimensions"
            )

        if len(signal) < min_length:
            raise ValueError(f"Signal length must be at least {min_length}")

        if not np.isfinite(signal).all():
            raise ValueError("Signal contains NaN or infinite values")

    except Exception as e:
        logger.error(f"Error in wavelet_denoise: {str(e)}")
        raise


def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: Optional[int] = None,
    mode: str = "soft",
) -> np.ndarray:
    """
    Perform wavelet denoising on the signal using wavelet thresholding.

    Parameters:
    -----------
    signal : np.ndarray
        Input ECG signal (1D numpy array)
    wavelet : str
        Wavelet type (e.g., 'db4', 'sym4', 'coif1')
    level : Optional[int]
        Decomposition level. If None, will be automatically computed
    mode : str
        Thresholding mode ('soft' or 'hard')

    Returns:
    --------
    np.ndarray
        Denoised signal

    Raises:
    -------
    ValueError
        If parameters are invalid
    """
    try:
        # Validate input
        validate_signal(signal)
        if mode not in ["soft", "hard"]:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'soft' or 'hard'")

        # Compute the maximum possible decomposition level if not specified
        if level is None:
            level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
        elif level <= 0:
            raise ValueError(f"Level must be positive, got {level}")

        # Perform wavelet decomposition
        coeffs = pywt.wavedec(signal, wavelet, level=level)

        # Estimate noise standard deviation from the last detail coefficients
        # sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        sigma = mad(coeffs[-1]) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))

        # Process coefficients with safe division
        denoised_coeffs = []
        for i, coeff in enumerate(coeffs):
            if i == 0:  # Keep approximation coefficients
                denoised_coeffs.append(coeff)
            else:
                if mode == "soft":
                    with np.errstate(divide="ignore", invalid="ignore"):
                        magnitude = np.abs(coeff)
                        # Avoid division by zero
                        mask = magnitude > threshold
                        coeff_thresh = np.zeros_like(coeff)
                        coeff_thresh[mask] = coeff[mask] * np.maximum(
                            0, 1 - threshold / magnitude[mask]
                        )
                else:  # hard thresholding
                    coeff_thresh = pywt.threshold(coeff, threshold, mode="hard")
                denoised_coeffs.append(coeff_thresh)

        return pywt.waverec(denoised_coeffs, wavelet)

    except Exception as e:
        logger.error(f"Error in wavelet_denoise: {str(e)}")
        raise


def adaptive_lms_filter(
    desired: np.ndarray, reference: np.ndarray, mu: float = 0.01, filter_order: int = 32
) -> np.ndarray:
    """
    Perform adaptive filtering using the Least Mean Squares (LMS) algorithm.

    Parameters:
    -----------
    desired : np.ndarray
        The noisy ECG signal
    reference : np.ndarray
        A reference noise signal (must be same length as desired)
    mu : float
        Learning rate (step size) for the adaptive filter (0 < mu < 1)
    filter_order : int
        Number of taps in the adaptive filter

    Returns:
    --------
    np.ndarray
        Denoised signal

    Raises:
    -------
    ValueError
        If input parameters are invalid
    """
    try:
        # Validate inputs
        validate_signal(desired)
        validate_signal(reference)

        if len(desired) != len(reference):
            raise ValueError("Desired and reference signals must have the same length")

        if not 0 < mu < 1:
            raise ValueError(f"Learning rate mu must be between 0 and 1, got {mu}")

        if not isinstance(filter_order, int) or filter_order <= 0:
            raise ValueError(
                f"Filter order must be a positive integer, got {filter_order}"
            )

        if filter_order >= len(desired):
            raise ValueError("Filter order must be less than signal length")

        # Initialize filter
        n_samples = len(desired)
        weights = np.zeros(filter_order)
        output = np.zeros(n_samples)

        # Process signal in chunks for better performance
        chunk_size = 1000
        for i in range(filter_order, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            chunk_range = np.arange(i, end)

            # Create input matrix for the chunk
            X = np.array([reference[j - filter_order : j][::-1] for j in chunk_range])

            # Filter the chunk
            Y = np.dot(X, weights)
            output[chunk_range] = Y

            # Update weights
            errors = desired[chunk_range] - Y
            weights += mu * np.dot(X.T, errors) / len(errors)

        # Remove the noise estimate from desired
        return desired - output

    except Exception as e:
        logger.error(f"Error in adaptive_lms_filter: {str(e)}")
        raise


def emd_denoise(signal: np.ndarray, imf_to_remove: int = 1) -> np.ndarray:
    """
    Denoise ECG signal using Empirical Mode Decomposition (EMD).
    Removes the specified number of Intrinsic Mode Functions (IMFs) assumed to be noise.

    Parameters:
    -----------
    signal : np.ndarray
        Input ECG signal
    imf_to_remove : int
        Number of initial IMFs (assumed to be mostly noise) to remove

    Returns:
    --------
    np.ndarray
        Reconstructed signal after removing noisy IMF components

    Raises:
    -------
    ValueError
        If parameters are invalid or EMD fails
    """
    try:
        # Validate inputs
        validate_signal(signal)

        if not isinstance(imf_to_remove, int) or imf_to_remove < 0:
            raise ValueError(
                f"imf_to_remove must be a non-negative integer, got {imf_to_remove}"
            )

        # Perform EMD
        emd = EMD()
        imfs = emd.emd(signal)

        # Validate decomposition
        if imfs is None or len(imfs) == 0:
            logger.warning("EMD decomposition failed, returning original signal")
            return signal

        # Remove specified IMFs and reconstruct
        if imfs.shape[0] > imf_to_remove:
            denoised_signal = np.sum(imfs[imf_to_remove:], axis=0)

            # Verify reconstruction quality
            reconstruction_error = np.mean(np.abs(signal - denoised_signal))
            if reconstruction_error > 0.1 * np.std(signal):
                logger.warning(
                    f"High reconstruction error in EMD: {reconstruction_error:.2e}"
                )
        else:
            logger.warning(
                f"Not enough IMFs ({imfs.shape[0]}) to remove {imf_to_remove}, returning original signal"
            )
            denoised_signal = signal

        return denoised_signal

    except Exception as e:
        logger.error(f"Error in emd_denoise: {str(e)}")
        raise


def median_filter_signal(signal: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply a median filter to remove impulsive noise from the signal.

    Parameters:
    -----------
    signal : np.ndarray
        Input ECG signal
    kernel_size : int
        Size of the median filter window (must be odd)

    Returns:
    --------
    np.ndarray
        Median filtered signal

    Raises:
    -------
    ValueError
        If parameters are invalid
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

        if kernel_size >= len(signal):
            raise ValueError("kernel_size must be less than signal length")

        return medfilt(signal, kernel_size=kernel_size)

    except Exception as e:
        logger.error(f"Error in median_filter_signal: {str(e)}")
        raise


def smooth_signal(
    signal: np.ndarray, window_length: int = 51, polyorder: int = 3
) -> np.ndarray:
    """
    Smooth the ECG signal using a Savitzky-Golay filter.
    This filter performs better than moving average as it preserves higher moments of the signal.

    Parameters:
    -----------
    signal : np.ndarray
        Input ECG signal
    window_length : int
        Length of the filter window (must be odd)
    polyorder : int
        Order of the polynomial used to fit the samples

    Returns:
    --------
    np.ndarray
        Smoothed signal

    Raises:
    -------
    ValueError
        If parameters are invalid
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

        max_window = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
        if window_length > max_window:
            window_length = max_window
            logger.warning(f"Adjusted window_length to signal length: {window_length}")

        return savgol_filter(signal, window_length, polyorder)

    except Exception as e:
        logger.error(f"Error in smooth_signal: {str(e)}")
        raise


def remove_respiratory_noise(
    signal: np.ndarray, fs: float, resp_freq_range: Tuple[float, float] = (0.15, 0.4)
) -> np.ndarray:
    """
    Remove respiratory noise from ECG signal using bandstop filtering.

    Parameters:
    -----------
    signal : np.ndarray
        Input ECG signal
    fs : float
        Sampling frequency
    resp_freq_range : Tuple[float, float]
        Frequency range of respiratory activity (typically 0.15-0.4 Hz)

    Returns:
    --------
    np.ndarray
        ECG signal with respiratory noise removed
    """
    try:
        validate_signal(signal)
        nyquist = fs / 2
        low, high = resp_freq_range
        b, a = butter(4, [low / nyquist, high / nyquist], btype="bandstop")
        return filtfilt(b, a, signal)
    except Exception as e:
        logger.error(f"Error in respiratory noise removal: {str(e)}")
        raise


def remove_emg_noise(
    signal: np.ndarray, fs: float, emg_freq_threshold: float = 20.0
) -> np.ndarray:
    """
    Remove high-frequency EMG noise using wavelet denoising and low-pass filtering.

    Parameters:
    -----------
    signal : np.ndarray
        Input ECG signal
    fs : float
        Sampling frequency
    emg_freq_threshold : float
        Cutoff frequency for EMG noise (typically > 20 Hz)

    Returns:
    --------
    np.ndarray
        ECG signal with EMG noise removed
    """
    try:
        validate_signal(signal)

        # First apply wavelet denoising
        denoised = wavelet_denoise(signal, wavelet="sym4", level=4, mode="soft")

        # Then apply low-pass filter to remove remaining high-freq noise
        nyquist = fs / 2
        b, a = butter(4, emg_freq_threshold / nyquist, btype="low")
        return filtfilt(b, a, denoised)
    except Exception as e:
        logger.error(f"Error in EMG noise removal: {str(e)}")
        raise


def remove_eda_noise(
    signal: np.ndarray, fs: float, eda_freq_range: Tuple[float, float] = (0.01, 1.0)
) -> np.ndarray:
    """
    Remove electrodermal activity (EDA) noise which typically occurs at very low frequencies.

    Parameters:
    -----------
    signal : np.ndarray
        Input ECG signal
    fs : float
        Sampling frequency
    eda_freq_range : Tuple[float, float]
        Frequency range of EDA activity (typically 0.01-1.0 Hz)

    Returns:
    --------
    np.ndarray
        ECG signal with EDA noise removed
    """
    try:
        validate_signal(signal)
        nyquist = fs / 2
        low, high = eda_freq_range
        b, a = butter(4, [low / nyquist, high / nyquist], btype="bandstop")
        return filtfilt(b, a, signal)
    except Exception as e:
        logger.error(f"Error in EDA noise removal: {str(e)}")
        raise


def advanced_denoise_pipeline(
    signal: np.ndarray,
    fs: float,
    remove_resp: bool = True,
    remove_emg: bool = True,
    remove_eda: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Complete denoising pipeline that removes multiple types of noise in sequence.

    Parameters:
    -----------
    signal : np.ndarray
        Input ECG signal
    fs : float
        Sampling frequency
    remove_resp : bool
        Whether to remove respiratory noise
    remove_emg : bool
        Whether to remove EMG noise
    remove_eda : bool
        Whether to remove EDA noise

    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'denoised': Final denoised signal
        - 'resp_removed': Signal after respiratory noise removal
        - 'emg_removed': Signal after EMG noise removal
        - 'eda_removed': Signal after EDA noise removal
    """
    try:
        validate_signal(signal)
        result = {"original": signal.copy()}
        current_signal = signal.copy()

        if remove_resp:
            current_signal = remove_respiratory_noise(current_signal, fs)
            result["resp_removed"] = current_signal.copy()

        if remove_emg:
            current_signal = remove_emg_noise(current_signal, fs)
            result["emg_removed"] = current_signal.copy()

        if remove_eda:
            current_signal = remove_eda_noise(current_signal, fs)
            result["eda_removed"] = current_signal.copy()

        result["denoised"] = current_signal
        return result

    except Exception as e:
        logger.error(f"Error in denoising pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # Import matplotlib for plotting
    import matplotlib.pyplot as plt

    # Set the simulation parameters
    fs = 500  # Sampling frequency (Hz)
    t = np.linspace(0, 1, fs, endpoint=False)  # 1 second of data
    freq = 5  # Frequency of the simulated ECG-like sine wave (Hz)

    # Generate a clean 'ECG-like' signal (sine wave) and add Gaussian noise
    clean_signal = np.sin(2 * np.pi * freq * t)
    noise = np.random.normal(0, 0.2, size=clean_signal.shape)
    noisy_signal = clean_signal + noise

    # Test wavelet denoising
    denoised_wavelet = wavelet_denoise(
        noisy_signal, wavelet="db4", level=4, mode="soft"
    )

    # Test adaptive LMS filter
    # For adaptive filtering, we simulate a noise reference (e.g., a sensor measuring noise)
    noise_reference = np.random.normal(0, 0.2, size=clean_signal.shape)
    denoised_adaptive = adaptive_lms_filter(
        noisy_signal, noise_reference, mu=0.01, filter_order=32
    )

    # Test EMD denoising (remove first IMF assumed to be noise)
    denoised_emd = emd_denoise(noisy_signal, imf_to_remove=1)

    # Test median filtering
    denoised_median = median_filter_signal(noisy_signal, kernel_size=5)

    # Test Savitzky–Golay smoothing
    denoised_smoothing = smooth_signal(noisy_signal, window_length=51, polyorder=3)

    # Test respiratory noise removal
    denoised_resp = remove_respiratory_noise(noisy_signal, fs)

    # Test EMG noise removal
    denoised_emg = remove_emg_noise(noisy_signal, fs)

    # Test EDA noise removal
    denoised_eda = remove_eda_noise(noisy_signal, fs)

    # Test advanced denoising pipeline
    denoised_pipeline = advanced_denoise_pipeline(noisy_signal, fs)

    # Plot all signals for comparison
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.plot(t, noisy_signal, label="Noisy Signal", color="gray")
    plt.plot(t, clean_signal, label="Clean Signal", linestyle="--", color="blue")
    plt.title("Original vs. Clean Signal")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(t, denoised_wavelet, label="Wavelet Denoised", color="green")
    plt.title("Wavelet Denoise")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(t, denoised_adaptive, label="Adaptive LMS Denoised", color="purple")
    plt.title("Adaptive LMS Filter")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(t, denoised_emd, label="EMD Denoised", color="orange")
    plt.title("EMD Denoise")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(t, denoised_median, label="Median Filter", color="red")
    plt.title("Median Filter")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(t, denoised_smoothing, label="Savitzky–Golay Denoise", color="brown")
    plt.title("Savitzky–Golay Smoothing")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
