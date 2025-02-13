import os
import tempfile
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy import stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def create_bandpass_filter(
    lowcut: float, highcut: float, fs: float, order: int = 4
) -> tuple:
    """
    Create bandpass filter coefficients with input validation and error handling.

    Returns the coefficients as torch tensors.
    """
    try:
        if not all(isinstance(x, (int, float)) for x in [lowcut, highcut, fs]):
            raise ValueError("Frequency parameters must be numeric")
        if not isinstance(order, int):
            raise ValueError("Filter order must be an integer")
        if order < 1:
            raise ValueError("Filter order must be positive")
        if lowcut >= highcut:
            raise ValueError("Lower cutoff must be less than upper cutoff")
        if lowcut <= 0:
            raise ValueError("Lower cutoff must be positive")
        if highcut >= fs / 2:
            raise ValueError(
                f"Upper cutoff must be less than Nyquist frequency ({fs / 2} Hz)"
            )

        from scipy.signal import butter, tf2zpk

        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = butter(order, [low, high], btype="band")
        z, p, k = tf2zpk(b, a)
        if np.any(np.abs(p) >= 1):
            logger.warning("Created filter may be unstable")

        return torch.tensor(b, dtype=torch.float64), torch.tensor(
            a, dtype=torch.float64
        )

    except Exception as e:
        logger.error(f"Error creating bandpass filter: {str(e)}")
        raise


def create_notch_filter(freq: float, q: float, fs: float) -> tuple:
    """
    Create notch filter coefficients with input validation and error handling.

    Returns the coefficients as torch tensors.
    """
    try:
        if not all(isinstance(x, (int, float)) for x in [freq, q, fs]):
            raise ValueError("All parameters must be numeric")
        if freq <= 0 or freq >= fs / 2:
            raise ValueError(f"Frequency must be between 0 and {fs / 2} Hz")
        if q <= 0:
            raise ValueError("Q factor must be positive")

        from scipy.signal import iirnotch, tf2zpk

        nyquist = 0.5 * fs
        norm_freq = freq / nyquist
        b, a = iirnotch(norm_freq, q)
        z, p, k = tf2zpk(b, a)
        if np.any(np.abs(p) >= 1):
            logger.warning("Created notch filter may be unstable")

        return torch.tensor(b, dtype=torch.float64), torch.tensor(
            a, dtype=torch.float64
        )

    except Exception as e:
        logger.error(f"Error creating notch filter: {str(e)}")
        raise


def normalize_signal(
    signal: np.ndarray, method: str = "minmax", eps: float = 1e-10
) -> np.ndarray:
    """
    Normalize a signal using various methods with improved numerical stability.

    If the input is not a torch tensor, it is converted to one and then converted back to a numpy array.
    """
    try:
        flag = False
        if not torch.is_tensor(signal):
            # Ensure the array is contiguous with positive strides.
            signal = np.ascontiguousarray(signal)
            signal = torch.tensor(signal, dtype=torch.float64)
            flag = True
        if signal.numel() == 0:
            raise ValueError("Input array is empty")
        if not torch.isfinite(signal).all():
            raise ValueError("Input contains non-finite values")

        if method == "minmax":
            min_val = torch.min(signal)
            max_val = torch.max(signal)
            if torch.abs(max_val - min_val) < eps:
                norm = torch.zeros_like(signal)
            else:
                norm = (signal - min_val) / (max_val - min_val)
        elif method == "zscore":
            mean = torch.mean(signal)
            std = torch.std(signal)
            if std < eps:
                norm = torch.zeros_like(signal)
            else:
                norm = (signal - mean) / std
        elif method == "robust":
            median = torch.median(signal)
            q25 = torch.quantile(signal, 0.25)
            q75 = torch.quantile(signal, 0.75)
            iqr = q75 - q25
            if iqr < eps:
                norm = torch.zeros_like(signal)
            else:
                norm = (signal - median) / iqr
        elif method == "l2":
            norm_val = torch.norm(signal, p=2)
            if norm_val < eps:
                norm = torch.zeros_like(signal)
            else:
                norm = signal / norm_val
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return norm.numpy() if flag else norm
    except Exception as e:
        logger.error(f"Error normalizing signal: {str(e)}")
        raise


def load_data(file_path: str, **kwargs) -> torch.Tensor:
    """
    Load data from various file formats with error handling.

    Returns the loaded data as a torch tensor.
    """
    try:
        if not isinstance(file_path, str):
            raise ValueError("File path must be a string")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".npy":
            data = np.load(file_path, **kwargs)
        elif ext == ".csv":
            csv_kwargs = {"delimiter": ",", "skiprows": 0, "usecols": None}
            csv_kwargs.update(kwargs)
            data = np.loadtxt(file_path, **csv_kwargs)
        elif ext == ".txt":
            txt_kwargs = {"delimiter": None, "skiprows": 0, "usecols": None}
            txt_kwargs.update(kwargs)
            data = np.loadtxt(file_path, **txt_kwargs)
        elif ext == ".mat":
            from scipy.io import loadmat

            mat_data = loadmat(file_path, **kwargs)
            data = next(
                value for key, value in mat_data.items() if not key.startswith("__")
            )
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        if not isinstance(data, np.ndarray):
            raise ValueError("Loaded data is not a numpy array")
        if data.size == 0:
            raise ValueError("Loaded data is empty")
        if not np.isfinite(data).all():
            logger.warning("Loaded data contains non-finite values")

        return torch.tensor(data, dtype=torch.float64)
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise


def resample_signal(signal: np.ndarray, fs_in: float, fs_out: float) -> torch.Tensor:
    """
    Resample a signal to a new sampling frequency using anti-aliasing.

    Returns the resampled signal as a torch tensor.
    """
    try:
        if not isinstance(signal, np.ndarray):
            if torch.is_tensor(signal):
                signal = signal.detach().cpu().numpy()
            else:
                raise ValueError("Signal must be a numpy array or torch tensor")
        if not np.isfinite(signal).all():
            raise ValueError("Signal contains non-finite values")
        if not all(isinstance(x, (int, float)) for x in [fs_in, fs_out]):
            raise ValueError("Sampling frequencies must be numeric")
        if fs_in <= 0 or fs_out <= 0:
            raise ValueError("Sampling frequencies must be positive")
        if np.isclose(fs_in, fs_out):
            return torch.tensor(signal, dtype=torch.float64)

        ratio = fs_out / fs_in
        n_samples = int(len(signal) * ratio)
        from scipy.signal import resample

        resampled = resample(signal.astype(np.float64), n_samples)
        return torch.tensor(resampled, dtype=torch.float64)
    except Exception as e:
        logger.error(f"Error resampling signal: {str(e)}")
        raise


def segment_signal(
    signal: np.ndarray, segment_length: int, overlap: float = 0.5
) -> torch.Tensor:
    """
    Segment a signal into overlapping windows.

    Returns a torch tensor of segments with shape (n_segments x segment_length)
    """
    try:
        if not isinstance(signal, np.ndarray):
            if torch.is_tensor(signal):
                signal = signal.detach().cpu().numpy()
            else:
                raise ValueError("Signal must be a numpy array or torch tensor")
        if not isinstance(segment_length, int):
            raise ValueError("Segment length must be an integer")
        if segment_length <= 0:
            raise ValueError("Segment length must be positive")
        if not 0 <= overlap < 1:
            raise ValueError("Overlap must be between 0 and 1")
        step = int(segment_length * (1 - overlap))
        if step < 1:
            step = 1
        n_segments = (len(signal) - segment_length) // step + 1
        if n_segments < 1:
            raise ValueError("Signal too short for given segment length")
        segments = np.zeros((n_segments, segment_length), dtype=np.float64)
        for i in range(n_segments):
            start = i * step
            segments[i] = signal[start : start + segment_length]
        return torch.tensor(segments, dtype=torch.float64)
    except Exception as e:
        logger.error(f"Error segmenting signal: {str(e)}")
        raise


def detect_outliers(
    signal: np.ndarray, threshold: float = 3.0, method: str = "zscore"
) -> np.ndarray:
    """
    Detect outliers in a signal using various methods.

    Returns a boolean numpy array where True indicates outliers.
    """
    try:
        if not isinstance(signal, np.ndarray):
            if torch.is_tensor(signal):
                signal = signal.detach().cpu().numpy()
            else:
                raise ValueError("Signal must be a numpy array or torch tensor")
        if not np.isfinite(signal).all():
            raise ValueError("Signal contains non-finite values")
        if threshold <= 0:
            raise ValueError("Threshold must be positive")

        if method == "zscore":
            z_scores = np.abs(stats.zscore(signal))
            return z_scores > threshold

        elif method == "iqr":
            q75, q25 = np.percentile(signal, [75, 25])
            iqr = q75 - q25
            bounds = (q25 - threshold * iqr, q75 + threshold * iqr)
            return (signal < bounds[0]) | (signal > bounds[1])

        elif method == "mad":
            median = np.median(signal)
            mad = np.median(np.abs(signal - median))
            modified_zscore = 0.6745 * (signal - median) / mad
            return np.abs(modified_zscore) > threshold

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    except Exception as e:
        logger.error(f"Error detecting outliers: {str(e)}")
        raise


def interpolate_missing(
    signal: np.ndarray, mask: np.ndarray, method: str = "linear"
) -> torch.Tensor:
    """
    Interpolate missing or invalid values in a signal.

    Returns the signal with interpolated values as a torch tensor.
    """
    try:
        if not isinstance(signal, np.ndarray) or not isinstance(mask, np.ndarray):
            raise ValueError("Inputs must be numpy arrays")
        if signal.shape != mask.shape:
            raise ValueError("Signal and mask must have same shape")
        if mask.dtype != bool:
            raise ValueError("Mask must be boolean")
        if not np.any(mask):
            return torch.tensor(signal.copy(), dtype=torch.float64)

        x = np.arange(len(signal))
        x_known = x[~mask]
        y_known = signal[~mask]
        if len(x_known) < 2:
            raise ValueError("Not enough valid points for interpolation")
        from scipy.interpolate import interp1d

        f = interp1d(
            x_known, y_known, kind=method, bounds_error=False, fill_value="extrapolate"
        )
        y_interp = f(x)
        result = signal.copy()
        result[mask] = y_interp[mask]
        return torch.tensor(result, dtype=torch.float64)
    except Exception as e:
        logger.error(f"Error interpolating signal: {str(e)}")
        raise


def test_bandpass_filter():
    fs = 250.0  # Sampling frequency in Hz
    lowcut = 0.5
    highcut = 40.0
    order = 4
    b, a = create_bandpass_filter(lowcut, highcut, fs, order)
    # Convert to numpy for freqz
    b_np = b.detach().cpu().numpy()
    a_np = a.detach().cpu().numpy()
    w, h = freqz(b_np, a_np, worN=8000)
    freqs = w * fs / (2 * np.pi)
    plt.figure()
    plt.plot(freqs, 20 * np.log10(np.abs(h)))
    plt.title("Bandpass Filter Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.grid(True)
    plt.show()


def test_notch_filter():
    fs = 250.0
    freq_target = 50.0
    q = 30.0
    b, a = create_notch_filter(freq_target, q, fs)
    b_np = b.detach().cpu().numpy()
    a_np = a.detach().cpu().numpy()
    w, h = freqz(b_np, a_np, worN=8000)
    freqs = w * fs / (2 * np.pi)
    plt.figure()
    plt.plot(freqs, 20 * np.log10(np.abs(h)))
    plt.title("Notch Filter Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.grid(True)
    plt.show()


def test_normalize_signal():
    x = np.linspace(-10, 10, 500)
    original_signal = np.sin(x) + np.random.normal(0, 0.5, x.shape) + 5  # offset added
    normalized_signal = normalize_signal(original_signal)
    plt.figure()
    plt.plot(x, original_signal, label="Original Signal")
    plt.plot(x, normalized_signal, label="Normalized Signal")
    plt.title("Signal Normalization")
    plt.xlabel("x")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()


def test_load_data():
    data = np.linspace(0, 1, 100)
    temp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    np.savetxt(temp_file, data, delimiter=",")
    temp_file.close()
    loaded_data = load_data(temp_file.name)
    print("Original Data (first 5 elements):", data[:5])
    print("Loaded Data   (first 5 elements):", loaded_data[:5].detach().cpu().numpy())
    plt.figure()
    plt.plot(data, "o-", label="Original Data")
    plt.plot(loaded_data.detach().cpu().numpy(), "x-", label="Loaded Data")
    plt.title("Load Data Test")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
    os.remove(temp_file.name)


if __name__ == "__main__":
    print("Testing Bandpass Filter Function...")
    test_bandpass_filter()
    print("Testing Notch Filter Function...")
    test_notch_filter()
    print("Testing Signal Normalization Function...")
    test_normalize_signal()
    print("Testing Data Loading Function...")
    test_load_data()
