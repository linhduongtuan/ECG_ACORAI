import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy import stats
import tempfile
import os
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def create_bandpass_filter(
    lowcut: float, highcut: float, fs: float, order: int = 4
) -> tuple:
    """
    Create bandpass filter coefficients with input validation and error handling.

    Parameters
    ----------
    lowcut : float
        Lower cutoff frequency in Hz
    highcut : float
        Upper cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    order : int, optional
        Filter order (default=4)

    Returns
    -------
    tuple
        Filter coefficients (b, a)

    Raises
    ------
    ValueError
        If parameters are invalid
    """
    try:
        # Input validation
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

        # Import here to avoid issues if not used
        from scipy.signal import butter

        # Calculate normalized frequencies
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist

        # Create filter
        b, a = butter(order, [low, high], btype="band")

        # Verify filter stability
        from scipy.signal import tf2zpk

        z, p, k = tf2zpk(b, a)
        if np.any(np.abs(p) >= 1):
            logger.warning("Created filter may be unstable")

        return b, a

    except Exception as e:
        logger.error(f"Error creating bandpass filter: {str(e)}")
        raise


def create_notch_filter(freq: float, q: float, fs: float) -> tuple:
    """
    Create notch filter coefficients with input validation and error handling.

    Parameters
    ----------
    freq : float
        Center frequency to remove (Hz)
    q : float
        Quality factor. Higher Q gives a narrower notch
    fs : float
        Sampling frequency (Hz)

    Returns
    -------
    tuple
        Filter coefficients (b, a)

    Raises
    ------
    ValueError
        If parameters are invalid
    """
    try:
        # Input validation
        if not all(isinstance(x, (int, float)) for x in [freq, q, fs]):
            raise ValueError("All parameters must be numeric")
        if freq <= 0 or freq >= fs / 2:
            raise ValueError(f"Frequency must be between 0 and {fs / 2} Hz")
        if q <= 0:
            raise ValueError("Q factor must be positive")

        # Import here to avoid issues if not used
        from scipy.signal import iirnotch

        # Calculate normalized frequency
        nyquist = 0.5 * fs
        norm_freq = freq / nyquist

        # Create filter
        b, a = iirnotch(norm_freq, q)

        # Verify filter stability
        from scipy.signal import tf2zpk

        z, p, k = tf2zpk(b, a)
        if np.any(np.abs(p) >= 1):
            logger.warning("Created notch filter may be unstable")

        return b, a

    except Exception as e:
        logger.error(f"Error creating notch filter: {str(e)}")
        raise


def normalize_signal(
    signal: np.ndarray, method: str = "minmax", eps: float = 1e-10
) -> np.ndarray:
    """
    Normalize signal using various methods with improved numerical stability.

    Parameters
    ----------
    signal : np.ndarray
        Input signal to normalize
    method : str, optional
        Normalization method ('minmax', 'zscore', 'robust', 'l2')
    eps : float, optional
        Small constant for numerical stability

    Returns
    -------
    np.ndarray
        Normalized signal

    Raises
    ------
    ValueError
        If input is invalid or method is not supported
    """
    try:
        # Input validation
        if not isinstance(signal, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if signal.size == 0:
            raise ValueError("Input array is empty")
        if not np.isfinite(signal).all():
            raise ValueError("Input contains non-finite values")

        # Convert to float64 for better precision
        signal = signal.astype(np.float64)

        if method == "minmax":
            # Min-max normalization to [0, 1]
            min_val = np.min(signal)
            max_val = np.max(signal)
            if np.abs(max_val - min_val) < eps:
                logger.warning("Signal has very small range, returning zeros")
                return np.zeros_like(signal)
            return (signal - min_val) / (max_val - min_val)

        elif method == "zscore":
            # Z-score normalization (mean=0, std=1)
            mean = np.mean(signal)
            std = np.std(signal)
            if std < eps:
                logger.warning(
                    "Signal has very small standard deviation, returning zeros"
                )
                return np.zeros_like(signal)
            return (signal - mean) / std

        elif method == "robust":
            # Robust scaling using median and IQR
            median = np.median(signal)
            q75, q25 = np.percentile(signal, [75, 25])
            iqr = q75 - q25
            if iqr < eps:
                logger.warning("Signal has very small IQR, returning zeros")
                return np.zeros_like(signal)
            return (signal - median) / iqr

        elif method == "l2":
            # L2 norm normalization
            norm = np.sqrt(np.sum(signal**2))
            if norm < eps:
                logger.warning("Signal has very small L2 norm, returning zeros")
                return np.zeros_like(signal)
            return signal / norm

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    except Exception as e:
        logger.error(f"Error normalizing signal: {str(e)}")
        raise


def load_data(file_path: str, **kwargs) -> np.ndarray:
    """
    Load data from various file formats with enhanced error handling and options.

    Parameters
    ----------
    file_path : str
        Path to the data file
    **kwargs : dict
        Additional arguments passed to the loading function

    Returns
    -------
    np.ndarray
        Loaded data array

    Raises
    ------
    ValueError
        If file format is unsupported or file is invalid
    FileNotFoundError
        If file does not exist
    """
    try:
        # Input validation
        if not isinstance(file_path, str):
            raise ValueError("File path must be a string")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file extension
        ext = os.path.splitext(file_path)[1].lower()

        # Load based on file type
        if ext == ".npy":
            data = np.load(file_path, **kwargs)
        elif ext == ".csv":
            # Default CSV options
            csv_kwargs = {"delimiter": ",", "skiprows": 0, "usecols": None}
            csv_kwargs.update(kwargs)
            data = np.loadtxt(file_path, **csv_kwargs)
        elif ext == ".txt":
            # Default TXT options
            txt_kwargs = {"delimiter": None, "skiprows": 0, "usecols": None}
            txt_kwargs.update(kwargs)
            data = np.loadtxt(file_path, **txt_kwargs)
        elif ext == ".mat":
            # Handle MATLAB files
            from scipy.io import loadmat

            mat_data = loadmat(file_path, **kwargs)
            # Assume the first array in the dict is the data
            data = next(
                value for key, value in mat_data.items() if not key.startswith("__")
            )
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Validate loaded data
        if not isinstance(data, np.ndarray):
            raise ValueError("Loaded data is not a numpy array")
        if data.size == 0:
            raise ValueError("Loaded data is empty")
        if not np.isfinite(data).all():
            logger.warning("Loaded data contains non-finite values")

        return data

    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise


def test_bandpass_filter():
    fs = 250.0  # Sampling frequency in Hz
    lowcut = 0.5
    highcut = 40.0
    order = 4

    # Obtain filter coefficients
    b, a = create_bandpass_filter(lowcut, highcut, fs, order)
    w, h = freqz(b, a, worN=8000)
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

    # Obtain notch filter coefficients
    b, a = create_notch_filter(freq_target, q, fs)
    w, h = freqz(b, a, worN=8000)
    freqs = w * fs / (2 * np.pi)

    plt.figure()
    plt.plot(freqs, 20 * np.log10(np.abs(h)))
    plt.title("Notch Filter Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.grid(True)
    plt.show()


def test_normalize_signal():
    # Create a dummy signal: a sinusoidal signal with noise and an offset
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
    # Create dummy data and write it to a temporary CSV file
    data = np.linspace(0, 1, 100)
    temp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    np.savetxt(temp_file, data, delimiter=",")
    temp_file.close()

    # Load the data from the temporary file
    loaded_data = load_data(temp_file.name)
    print("Original Data (first 5 elements):", data[:5])
    print("Loaded Data   (first 5 elements):", loaded_data[:5])

    # Plot the original vs. the loaded data
    plt.figure()
    plt.plot(data, "o-", label="Original Data")
    plt.plot(loaded_data, "x-", label="Loaded Data")
    plt.title("Load Data Test")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Clean up the temporary file
    os.remove(temp_file.name)


def resample_signal(signal: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """
    Resample signal to a new sampling frequency with anti-aliasing.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs_in : float
        Input sampling frequency
    fs_out : float
        Output sampling frequency

    Returns
    -------
    np.ndarray
        Resampled signal
    """
    try:
        # Input validation
        if not isinstance(signal, np.ndarray):
            raise ValueError("Signal must be a numpy array")
        if not np.isfinite(signal).all():
            raise ValueError("Signal contains non-finite values")
        if not all(isinstance(x, (int, float)) for x in [fs_in, fs_out]):
            raise ValueError("Sampling frequencies must be numeric")
        if fs_in <= 0 or fs_out <= 0:
            raise ValueError("Sampling frequencies must be positive")

        # If same frequency, return original signal
        if np.isclose(fs_in, fs_out):
            return signal

        # Calculate resampling ratio
        ratio = fs_out / fs_in
        n_samples = int(len(signal) * ratio)

        # Use scipy's resample function with better precision
        from scipy.signal import resample

        resampled = resample(signal.astype(np.float64), n_samples)

        return resampled

    except Exception as e:
        logger.error(f"Error resampling signal: {str(e)}")
        raise


def segment_signal(
    signal: np.ndarray, segment_length: int, overlap: float = 0.5
) -> np.ndarray:
    """
    Segment a signal into overlapping windows.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    segment_length : int
        Length of each segment
    overlap : float
        Overlap between segments (0-1)

    Returns
    -------
    np.ndarray
        Array of segments (n_segments x segment_length)
    """
    try:
        # Input validation
        if not isinstance(signal, np.ndarray):
            raise ValueError("Signal must be a numpy array")
        if not isinstance(segment_length, int):
            raise ValueError("Segment length must be an integer")
        if segment_length <= 0:
            raise ValueError("Segment length must be positive")
        if not 0 <= overlap < 1:
            raise ValueError("Overlap must be between 0 and 1")

        # Calculate step size
        step = int(segment_length * (1 - overlap))
        if step < 1:
            step = 1

        # Calculate number of segments
        n_segments = (len(signal) - segment_length) // step + 1
        if n_segments < 1:
            raise ValueError("Signal too short for given segment length")

        # Create segments
        segments = np.zeros((n_segments, segment_length))
        for i in range(n_segments):
            start = i * step
            segments[i] = signal[start : start + segment_length]

        return segments

    except Exception as e:
        logger.error(f"Error segmenting signal: {str(e)}")
        raise


def detect_outliers(
    signal: np.ndarray, threshold: float = 3.0, method: str = "zscore"
) -> np.ndarray:
    """
    Detect outliers in a signal using various methods.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    threshold : float
        Threshold for outlier detection
    method : str
        Detection method ('zscore', 'iqr', 'mad')

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates outliers
    """
    try:
        # Input validation
        if not isinstance(signal, np.ndarray):
            raise ValueError("Signal must be a numpy array")
        if not np.isfinite(signal).all():
            raise ValueError("Signal contains non-finite values")
        if threshold <= 0:
            raise ValueError("Threshold must be positive")

        if method == "zscore":
            # Z-score method
            z_scores = np.abs(stats.zscore(signal))
            return z_scores > threshold

        elif method == "iqr":
            # Interquartile range method
            q75, q25 = np.percentile(signal, [75, 25])
            iqr = q75 - q25
            bounds = (q25 - threshold * iqr, q75 + threshold * iqr)
            return (signal < bounds[0]) | (signal > bounds[1])

        elif method == "mad":
            # Median Absolute Deviation method
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
) -> np.ndarray:
    """
    Interpolate missing or invalid values in a signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    mask : np.ndarray
        Boolean mask where True indicates values to interpolate
    method : str
        Interpolation method ('linear', 'cubic', 'nearest')

    Returns
    -------
    np.ndarray
        Signal with interpolated values
    """
    try:
        # Input validation
        if not isinstance(signal, np.ndarray) or not isinstance(mask, np.ndarray):
            raise ValueError("Inputs must be numpy arrays")
        if signal.shape != mask.shape:
            raise ValueError("Signal and mask must have same shape")
        if not mask.dtype == bool:
            raise ValueError("Mask must be boolean")

        # If no missing values, return original signal
        if not np.any(mask):
            return signal.copy()

        # Create interpolation points
        x = np.arange(len(signal))
        x_known = x[~mask]
        y_known = signal[~mask]

        if len(x_known) < 2:
            raise ValueError("Not enough valid points for interpolation")

        # Interpolate
        from scipy.interpolate import interp1d

        f = interp1d(
            x_known, y_known, kind=method, bounds_error=False, fill_value="extrapolate"
        )
        y_interp = f(x)

        # Combine original and interpolated values
        result = signal.copy()
        result[mask] = y_interp[mask]

        return result

    except Exception as e:
        logger.error(f"Error interpolating signal: {str(e)}")
        raise


if __name__ == "__main__":
    print("Testing Bandpass Filter Function...")
    test_bandpass_filter()

    print("Testing Notch Filter Function...")
    test_notch_filter()

    print("Testing Signal Normalization Function...")
    test_normalize_signal()

    print("Testing Data Loading Function...")
    test_load_data()
