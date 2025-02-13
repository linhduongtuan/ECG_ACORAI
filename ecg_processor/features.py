import numpy as np
import pywt
from typing import Dict, List, Optional
import neurokit2 as nk
from scipy import signal, stats
from scipy.integrate import trapezoid
import logging
from .visualization import plot_signal_comparison, plot_comprehensive_analysis

logger = logging.getLogger(__name__)


def extract_statistical_features(beat: np.ndarray) -> Dict:
    """
    Extract statistical features from a beat.

    Parameters
    ----------
    beat : np.ndarray
        Single heartbeat signal

    Returns
    -------
    Dict
        Dictionary containing statistical features:
        - Basic statistics (mean, std, max, min)
        - Shape statistics (skewness, kurtosis)
        - Energy metrics (energy, RMS)
        - Additional metrics (entropy, zero crossings)

    Raises
    ------
    ValueError
        If input is invalid or empty
    """
    try:
        if not isinstance(beat, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if len(beat) == 0:
            raise ValueError("Input array is empty")
        if not np.isfinite(beat).all():
            raise ValueError("Input contains invalid values (inf or nan)")

        # Basic statistics
        basic_stats = {
            "mean": float(np.mean(beat)),
            "std": float(np.std(beat)),
            "max": float(np.max(beat)),
            "min": float(np.min(beat)),
            "median": float(np.median(beat)),
            "mad": float(stats.median_abs_deviation(beat)),
        }

        # Shape statistics
        shape_stats = {
            "skewness": float(stats.skew(beat)),
            "kurtosis": float(stats.kurtosis(beat)),
            "peak_factor": float(
                np.max(np.abs(beat)) / np.sqrt(np.mean(np.square(beat)))
            ),
        }

        # Energy metrics
        energy_metrics = {
            "energy": float(np.sum(beat**2)),
            "rms": float(np.sqrt(np.mean(np.square(beat)))),
            "peak_to_rms": float(
                np.max(np.abs(beat)) / np.sqrt(np.mean(np.square(beat)))
            ),
        }

        # Additional metrics
        hist, _ = np.histogram(beat, bins="auto", density=True)
        additional_metrics = {
            "entropy": float(stats.entropy(hist)),
            "zero_crossings": int(np.sum(np.diff(np.signbit(beat)))),
            "range": float(np.ptp(beat)),
            "variance": float(np.var(beat)),
        }

        return {**basic_stats, **shape_stats, **energy_metrics, **additional_metrics}

    except Exception as e:
        logger.error(f"Error extracting statistical features: {str(e)}")
        raise


def extract_wavelet_features(
    beat: np.ndarray, wavelet: str = "db4", level: int = 4
) -> Dict:
    """
    Extract wavelet features from a beat using discrete wavelet transform.

    Parameters
    ----------
    beat : np.ndarray
        Single heartbeat signal
    wavelet : str, optional
        Wavelet type to use, by default 'db4'
    level : int, optional
        Decomposition level, by default 4

    Returns
    -------
    Dict
        Dictionary containing wavelet features:
        - Energy at each decomposition level
        - Entropy at each level
        - Statistical features of coefficients

    Raises
    ------
    ValueError
        If input is invalid or wavelet parameters are incorrect
    """
    try:
        if not isinstance(beat, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if len(beat) < 2**level:
            raise ValueError(
                f"Signal length must be at least {2**level} for level {level} decomposition"
            )

        # Perform wavelet decomposition
        max_level = pywt.dwt_max_level(len(beat), pywt.Wavelet(wavelet).dec_len)
        level = min(level, max_level)  # Adjust level to avoid boundary effects
        coeffs = pywt.wavedec(beat, wavelet, level=level)
        features = {}

        # Calculate features for each level
        total_energy = 0
        for i, coeff in enumerate(coeffs):
            # Energy features
            energy = np.sum(np.square(coeff))
            total_energy += energy
            features[f"wavelet_energy_{i}"] = float(energy)

            # Entropy features
            hist, _ = np.histogram(coeff, bins="auto", density=True)
            features[f"wavelet_entropy_{i}"] = float(stats.entropy(hist))

            # Statistical features of coefficients
            features[f"wavelet_mean_{i}"] = float(np.mean(coeff))
            features[f"wavelet_std_{i}"] = float(np.std(coeff))
            features[f"wavelet_max_{i}"] = float(np.max(np.abs(coeff)))

        # Calculate relative energies
        for i in range(len(coeffs)):
            features[f"wavelet_relative_energy_{i}"] = float(
                features[f"wavelet_energy_{i}"] / total_energy
            )

        return features

    except Exception as e:
        logger.error(f"Error extracting wavelet features: {str(e)}")
        raise


def extract_morphological_features(beat: np.ndarray, fs: float) -> Dict:
    """
    Extract morphological features from a beat using neurokit2.

    Parameters
    ----------
    beat : np.ndarray
        Single heartbeat signal
    fs : float
        Sampling frequency in Hz

    Returns
    -------
    Dict
        Dictionary containing morphological features:
        - Wave durations (P, QRS, T)
        - Intervals (PR, QT)
        - Amplitudes and ratios
        - Wave symmetry metrics

    Raises
    ------
    ValueError
        If input is invalid or sampling rate is incorrect
    """
    try:
        if not isinstance(beat, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("Sampling rate must be a positive number")
        if len(beat) < int(0.2 * fs):  # At least 200ms of data
            raise ValueError("Beat too short for morphological analysis")

        # Ensure minimum signal length (at least 1 second of data)
        min_length = int(fs)
        if len(beat) < min_length:
            logger.warning(
                f"Beat too short for morphological analysis: {len(beat)} samples"
            )
            return {
                "p_wave_duration": None,
                "qrs_duration": None,
                "t_wave_duration": None,
                "pr_interval": None,
                "qt_interval": None,
                "r_amplitude": None,
                "s_amplitude": None,
                "p_amplitude": None,
                "t_amplitude": None,
                "st_level": None,
            }

        # Process the beat to get peaks and waves
        _, info = nk.ecg_process(beat, sampling_rate=fs)
        peaks = info["ECG_R_Peaks"]
        _, waves = nk.ecg_delineate(beat, peaks, sampling_rate=fs, method="peak")

        # Calculate QRS duration
        qrs_onset = waves.get("ECG_Q_Onsets", [None])[0]
        qrs_offset = waves.get("ECG_S_Offsets", [None])[0]
        if qrs_onset is not None and qrs_offset is not None:
            qrs_duration = (qrs_offset - qrs_onset) / fs * 1000  # Convert to ms
        else:
            qrs_duration = (
                None
                if qrs_onset is None or qrs_offset is None
                else (qrs_offset - qrs_onset) / fs * 1000
            )  # Convert to ms

        # Calculate wave amplitudes
        r_peak_idx = peaks[0] if len(peaks) > 0 else None
        r_amplitude = _safe_amplitude(beat, waves, "ECG_R_Peaks")
        s_amplitude = _safe_amplitude(beat, waves, "ECG_S_Peaks")

        return {
            "p_wave_duration": _safe_duration(waves, "P_Onset", "P_Offset", fs),
            "qrs_duration": _safe_duration(waves, "QRS_Onset", "QRS_Offset", fs),
            "t_wave_duration": _safe_duration(waves, "T_Onset", "T_Offset", fs),
            "pr_interval": _safe_duration(waves, "P_Onset", "QRS_Onset", fs),
            "qt_interval": _safe_duration(waves, "QRS_Onset", "T_Offset", fs),
            "r_amplitude": float(np.max(waves)),
            "s_amplitude": float(np.min(waves)),
            "p_amplitude": _safe_amplitude(waves, "P_Peak"),
            "t_amplitude": _safe_amplitude(waves, "T_Peak"),
            "st_level": _calculate_st_level(waves, fs),
        }

    except Exception as e:
        logger.error(f"Error extracting morphological features: {str(e)}")
        return {}


def test_extract_statistical_features():
    """
    Test the extract_statistical_features function with a synthetic beat.
    """
    try:
        # Create a synthetic beat with more realistic length and shape
        t = np.linspace(0, 1, 100)  # 100 samples
        beat = np.sin(2 * np.pi * 2 * t) * np.exp(-2 * t)  # Damped sine wave

        features = extract_statistical_features(beat)
        print("Statistical Features:")
        for key, value in features.items():
            print(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Error in statistical features test: {str(e)}")


def test_extract_wavelet_features():
    """
    Test the extract_wavelet_features function with a synthetic beat.
    """
    try:
        # Create a synthetic beat with sufficient length for wavelet decomposition
        # For level 4 decomposition, we need at least 2^4 = 16 samples
        # Using 128 samples for better resolution
        t = np.linspace(0, 1, 128)
        beat = np.sin(2 * np.pi * 2 * t) * np.exp(-2 * t)  # Damped sine wave

        # Test with lower level first
        print("\nTesting wavelet decomposition with level 2:")
        features_l2 = extract_wavelet_features(beat, wavelet="db4", level=2)
        print("Wavelet Features (Level 2):")
        for key, value in features_l2.items():
            print(f"  {key}: {value}")

        # Test with level 4
        print("\nTesting wavelet decomposition with level 4:")
        features_l4 = extract_wavelet_features(beat, wavelet="db4", level=4)
        print("Wavelet Features (Level 4):")
        for key, value in features_l4.items():
            print(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Error extracting wavelet features: {str(e)}")


def test_extract_morphological_features():
    """
    Test the extract_morphological_features function.

    This function uses NeuroKit2 to simulate an ECG signal and then passes it to
    the morphological feature extractor. Depending on the simulated signal and the
    internal implementation of nk.ecg_peaks/nk.ecg_delineate, some outputs may be None.
    """
    fs = 500  # sampling rate
    # Simulate an ECG beat (you might change the duration to get a cleaner beat)
    beat = nk.ecg_simulate(duration=1, sampling_rate=fs)
    features = extract_morphological_features(beat, fs)
    print("Morphological Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")


def extract_stft_features(beat: np.ndarray, fs: float, nperseg: int = 128) -> Dict:
    """
    Extract Short-Time Fourier Transform features from a beat.

    Parameters
    ----------
    beat : np.ndarray
        Single heartbeat signal
    fs : float
        Sampling frequency in Hz
    nperseg : int, optional
        Length of each segment, by default 128

    Returns
    -------
    Dict
        Dictionary containing STFT features:
        - Energy statistics
        - Frequency band powers
        - Spectral shape metrics

    Raises
    ------
    ValueError
        If input parameters are invalid
    """
    try:
        if not isinstance(beat, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("Sampling rate must be a positive number")
        if len(beat) < nperseg:
            raise ValueError(f"Signal length must be at least {nperseg}")

        # Compute STFT
        f, t, Zxx = signal.stft(beat, fs=fs, nperseg=nperseg)
        magnitude = np.abs(Zxx)

        # Basic energy statistics
        energy_stats = {
            "stft_mean_energy": float(np.mean(magnitude)),
            "stft_max_energy": float(np.max(magnitude)),
            "stft_std_energy": float(np.std(magnitude)),
            "stft_total_energy": float(np.sum(magnitude)),
        }

        # Frequency band powers
        freq_bands = {
            "low": (0, 5),  # 0-5 Hz
            "medium": (5, 20),  # 5-20 Hz
            "high": (20, 50),  # 20-50 Hz
        }

        band_powers = {}
        for band_name, (low, high) in freq_bands.items():
            mask = (f >= low) & (f <= high)
            band_powers[f"stft_power_{band_name}"] = float(
                np.sum(magnitude[mask, :]) / magnitude.size
            )

        # Spectral shape metrics
        shape_metrics = {
            "stft_spectral_centroid": float(_spectral_centroid(f, magnitude)),
            "stft_spectral_bandwidth": float(_spectral_bandwidth(f, magnitude)),
            "stft_spectral_rolloff": float(_spectral_rolloff(magnitude)),
        }

        return {**energy_stats, **band_powers, **shape_metrics}

    except Exception as e:
        logger.error(f"Error extracting STFT features: {str(e)}")
        raise


def extract_hybrid_features(
    beat: np.ndarray,
    fs: float,
    wavelet: str = "db4",
    level: int = 4,
    nperseg: int = 128,
) -> Dict:
    """
    Extract hybrid features combining multiple domains.

    Parameters
    ----------
    beat : np.ndarray
        Single heartbeat signal
    fs : float
        Sampling frequency in Hz
    wavelet : str, optional
        Wavelet type to use, by default 'db4'
    level : int, optional
        Decomposition level, by default 4
    nperseg : int, optional
        Length of each STFT segment, by default 128

    Returns
    -------
    Dict
        Dictionary containing combined features from:
        - Statistical analysis
        - Wavelet decomposition
        - Morphological analysis
        - Time-frequency analysis (STFT)

    Raises
    ------
    ValueError
        If input parameters are invalid
    """
    try:
        if not isinstance(beat, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("Sampling rate must be a positive number")

        features = {}

        # Extract features from each domain with proper error handling
        try:
            features.update(extract_statistical_features(beat))
        except Exception as e:
            logger.warning(f"Failed to extract statistical features: {str(e)}")

        try:
            features.update(extract_wavelet_features(beat, wavelet, level))
        except Exception as e:
            logger.warning(f"Failed to extract wavelet features: {str(e)}")

        try:
            features.update(extract_morphological_features(beat, fs))
        except Exception as e:
            logger.warning(f"Failed to extract morphological features: {str(e)}")

        try:
            features.update(extract_stft_features(beat, fs, nperseg))
        except Exception as e:
            logger.warning(f"Failed to extract STFT features: {str(e)}")

        # Validate that we have at least some features
        if not features:
            raise ValueError("No features could be extracted from the signal")

        return features

    except Exception as e:
        logger.error(f"Error in hybrid feature extraction: {str(e)}")
        raise


def _safe_duration(
    waves: Dict, start_key: str, end_key: str, fs: float
) -> Optional[float]:
    """Safely calculate duration between two wave points."""
    try:
        if waves[start_key] is not None and waves[end_key] is not None:
            return float((waves[end_key] - waves[start_key]) / fs)
        return None
    except (KeyError, TypeError):
        return None


def _safe_interval(waves: Dict, end_key: str, start_key: str) -> Optional[float]:
    """Safely calculate interval between two wave points."""
    if waves[end_key] and waves[start_key]:
        return float(waves[end_key] - waves[start_key])
    return None


def _safe_amplitude(beat: np.ndarray, waves: Dict, key: str) -> Optional[float]:
    """Safely get amplitude at a specific wave point."""
    if waves[key]:
        return float(beat[int(waves[key])])
    return None


def _calculate_st_level(beat: np.ndarray, waves: Dict, fs: float) -> Optional[float]:
    """Calculate ST segment level relative to baseline."""
    try:
        if waves["R_offset"] and waves["T_onset"]:
            st_start = int(waves["R_offset"])
            st_end = int(waves["T_onset"])
            st_segment = beat[st_start:st_end]
            return float(np.mean(st_segment))
    except Exception as e:
        logger.error(f"Error calculating ST level: {str(e)}")
        return None

    return None


def _calculate_wave_symmetry(
    beat: np.ndarray, waves: Dict, start_key: str, end_key: str
) -> Optional[float]:
    """Calculate symmetry of a wave segment."""
    try:
        if waves[start_key] and waves[end_key]:
            start = int(waves[start_key])
            end = int(waves[end_key])
            wave = beat[start:end]
            if len(wave) > 1:
                mid = len(wave) // 2
                left = wave[:mid]
                right = wave[mid:][::-1]  # Reverse for comparison
                return float(np.corrcoef(left, right)[0, 1])
    except Exception as e:
        logger.warning(f"Failed to calculate wave symmetry: {str(e)}")
        pass
    return None


def _spectral_centroid(freqs: np.ndarray, magnitude: np.ndarray) -> float:
    """Calculate spectral centroid."""
    magnitude_sum = np.sum(magnitude, axis=1)
    return np.sum(freqs * magnitude_sum) / np.sum(magnitude_sum)


def _spectral_bandwidth(freqs: np.ndarray, magnitude: np.ndarray) -> float:
    """Calculate spectral bandwidth."""
    centroid = _spectral_centroid(freqs, magnitude)
    magnitude_sum = np.sum(magnitude, axis=1)
    variance = np.sum(((freqs - centroid) ** 2) * magnitude_sum) / np.sum(magnitude_sum)
    return np.sqrt(variance)


def _spectral_rolloff(magnitude: np.ndarray, percentile: float = 0.85) -> float:
    """Calculate frequency below which percentile of the magnitude spectrum energy lies."""
    power = np.sum(magnitude**2, axis=1)
    cumsum = np.cumsum(power)
    return np.where(cumsum >= percentile * cumsum[-1])[0][0]


def run_tests():
    print("Running tests for feature extraction functions...\n")
    test_extract_statistical_features()
    print("\n" + "-" * 40 + "\n")
    test_extract_wavelet_features()
    print("\n" + "-" * 40 + "\n")
    test_extract_morphological_features()


if __name__ == "__main__":
    run_tests()
    # --- Create dummy signals ---
    fs = 500  # Sampling rate in Hz
    duration = 10  # seconds
    t = np.linspace(0, duration, duration * fs, endpoint=False)

    # Create a dummy original ECG-like signal: a 1-Hz sine wave plus random noise
    original_signal = np.sin(2 * np.pi * 1 * t) + 0.2 * np.random.randn(len(t))

    # Create a dummy processed signal simulating filtering:
    # for simplicity, use a sine wave (less noise) as "processed" output.
    processed_signal = np.sin(2 * np.pi * 1 * t)

    # --- Plot the dummy signals comparison using the first function ---
    plot_signal_comparison(
        original=original_signal,
        processed=processed_signal,
        fs=fs,
        title="Dummy ECG Signal Comparison",
    )

    # --- Create a dummy results dictionary for comprehensive analysis ---
    # Here we simulate:
    #   - peaks: every 500 samples (once per second),
    #   - RR intervals: dummy values in milliseconds,
    #   - Frequencies and PSD: dummy power spectral density data.
    dummy_peaks = np.arange(0, len(processed_signal), fs)
    dummy_rr_intervals = np.diff(dummy_peaks) / fs * 1000  # in ms

    dummy_frequencies = np.linspace(0.1, 30, 100)
    dummy_psd = np.abs(np.sin(dummy_frequencies))  # dummy PSD values

    results = {
        "original_signal": original_signal,
        "processed_signal": processed_signal,
        "peaks": dummy_peaks,
        "hrv_metrics": {"rr_intervals": dummy_rr_intervals},
        "frequencies": dummy_frequencies,
        "psd": dummy_psd,
    }

    # --- Plot comprehensive analysis ---
    plot_comprehensive_analysis(results)


def analyze_qt_interval(beat: np.ndarray, waves: Dict, fs: float) -> Dict:
    """
    Analyze QT interval and its variants (QTc, JT, JTc).
    """
    try:
        # Validate inputs
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("Sampling frequency must be positive")

        if not isinstance(waves, dict):
            raise ValueError("Waves must be a dictionary")

        # Check for required wave points
        required_points = ["Q_start", "T_end", "R_start", "R_end"]
        if not all(
            point in waves and waves[point] is not None for point in required_points
        ):
            return {}

        # Calculate intervals in samples
        qt_samples = waves["T_end"] - waves["Q_start"]
        rr_samples = waves["R_end"] - waves["R_start"]

        # Convert to seconds
        qt_sec = qt_samples / fs
        rr_sec = rr_samples / fs

        # Validate intervals
        if qt_sec <= 0 or rr_sec <= 0:
            return {}

        # Calculate heart rate
        hr = 60 / rr_sec

        # Calculate QTc using different formulas
        qtc_bazett = qt_sec / np.sqrt(rr_sec)
        qtc_fridericia = qt_sec / np.cbrt(rr_sec)
        qtc_framingham = qt_sec + 0.154 * (1 - rr_sec)

        # Calculate JT interval if J point is available
        jt_sec = None
        jtc = None
        if "J_point" in waves and waves["J_point"] is not None:
            jt_samples = waves["T_end"] - waves["J_point"]
            jt_sec = jt_samples / fs
            jtc = jt_sec / np.sqrt(rr_sec)

        # Convert to milliseconds for output
        results = {
            "QT_interval": float(qt_sec * 1000),
            "QTc_Bazett": float(qtc_bazett * 1000),
            "QTc_Fridericia": float(qtc_fridericia * 1000),
            "QTc_Framingham": float(qtc_framingham * 1000),
            "JT_interval": float(jt_sec * 1000) if jt_sec is not None else None,
            "JTc": float(jtc * 1000) if jtc is not None else None,
            "Heart_Rate": float(hr),
        }

        return results

    except Exception as e:
        logger.error(f"Error in QT interval analysis: {str(e)}")
        return {}


def detect_t_wave_alternans(
    beats: List[np.ndarray], waves: List[Dict], fs: float
) -> Dict:
    """
    Detect and quantify T-wave alternans (TWA).

    Parameters
    ----------
    beats : List[np.ndarray]
        List of consecutive heartbeat signals
    waves : List[Dict]
        List of wave delineation dictionaries for each beat
    fs : float
        Sampling frequency in Hz

    Returns
    -------
    Dict
        Dictionary containing TWA metrics:
        - TWA magnitude
        - TWA ratio
        - Alternans voltage
        - K score
    """
    try:
        if len(beats) < 4:  # Need at least 4 beats for analysis
            return {}

        t_wave_amplitudes = []
        for i, (beat, wave) in enumerate(zip(beats, waves)):
            # Extract T-wave segment
            t_start = wave.get("T_start")
            t_end = wave.get("T_end")
            if t_start is not None and t_end is not None:
                t_wave = beat[t_start:t_end]
                t_wave_amplitudes.append(np.max(t_wave))

        if len(t_wave_amplitudes) < 4:
            return {}

        # Calculate alternans metrics
        t_wave_amplitudes = np.array(t_wave_amplitudes)
        even_beats = t_wave_amplitudes[::2]
        odd_beats = t_wave_amplitudes[1::2]

        # TWA magnitude
        twa_magnitude = np.mean(np.abs(even_beats - odd_beats))

        # TWA ratio
        twa_ratio = np.mean(np.abs(even_beats - odd_beats)) / np.mean(t_wave_amplitudes)

        # K score (statistical significance)
        k_score = _calculate_k_score(even_beats, odd_beats)

        return {
            "TWA_magnitude": float(twa_magnitude),
            "TWA_ratio": float(twa_ratio),
            "Alternans_voltage": float(twa_magnitude / 2),
            "K_score": float(k_score),
            "TWA_present": bool(k_score > 3 and twa_ratio > 0.1),
        }
    except Exception as e:
        logger.error(f"Error in T-wave alternans detection: {str(e)}")
        return {}


def analyze_st_segment(beat: np.ndarray, waves: Dict, fs: float) -> Dict:
    """
    Analyze ST segment characteristics.

    Parameters
    ----------
    beat : np.ndarray
        Single heartbeat signal
    waves : Dict
        Dictionary containing wave delineation points
    fs : float
        Sampling frequency in Hz

    Returns
    -------
    Dict
        Dictionary containing ST segment metrics:
        - ST level (elevation/depression)
        - ST slope
        - ST integral
        - ST shape classification
    """
    try:
        # Get ST segment points
        j_point = waves.get("J_point")
        t_start = waves.get("T_start")

        if j_point is None or t_start is None:
            return {}

        # Extract ST segment
        st_segment = beat[j_point:t_start]

        # Calculate baseline (using PR segment)
        pr_start = waves.get("P_start")
        pr_end = waves.get("P_end")
        if pr_start is not None and pr_end is not None:
            baseline = np.mean(beat[pr_start:pr_end])
        else:
            baseline = 0

        # ST metrics
        st_level = np.mean(st_segment[: int(0.04 * fs)]) - baseline  # First 40ms
        st_slope = np.polyfit(np.arange(len(st_segment)), st_segment, 1)[0]
        # st_integral = np.trapz(st_segment - baseline) / fs
        st_integral = trapezoid(st_segment - baseline) / fs

        # Classify ST shape
        st_shape = _classify_st_shape(st_segment, st_slope)

        return {
            "ST_level": float(st_level),
            "ST_slope": float(st_slope),
            "ST_integral": float(st_integral),
            "ST_shape": st_shape,
            "ST_elevation": bool(st_level > 0.1),  # 0.1mV threshold
            "ST_depression": bool(st_level < -0.1),
        }
    except Exception as e:
        logger.error(f"Error in ST segment analysis: {str(e)}")
        return {}


def _calculate_k_score(even_beats: np.ndarray, odd_beats: np.ndarray) -> float:
    """Calculate K-score for T-wave alternans."""
    try:
        if len(even_beats) != len(odd_beats):
            raise ValueError("Even and odd beat arrays must have the same length")

        # Calculate mean absolute difference
        mean_diff = np.mean(np.abs(even_beats - odd_beats))

        # Calculate noise estimate (standard deviation of differences)
        noise = np.std(even_beats - odd_beats)

        # Avoid division by zero
        if noise < 1e-10:
            noise = 1e-10

        # Calculate K-score
        k_score = float(mean_diff / noise)

        return k_score

    except Exception as e:
        logger.error(f"Error calculating K-score: {str(e)}")
        return 0.0


def _classify_st_shape(st_segment: np.ndarray, threshold: float = 0.1) -> str:
    """Classify ST segment shape."""
    try:
        if len(st_segment) < 2:
            return "horizontal"

        # Calculate slope using linear regression
        x = np.arange(len(st_segment))
        slope, _ = np.polyfit(x, st_segment, 1)

        if abs(slope) < threshold:
            return "horizontal"
        elif slope > threshold:
            return "upsloping"
        else:
            return "downsloping"

    except Exception as e:
        logger.error(f"Error classifying ST shape: {str(e)}")
        return "horizontal"


def _detect_qrs_peaks(self, signal: np.ndarray) -> np.ndarray:
    """Detect QRS complexes using NeuroKit2."""
    try:
        if len(signal) < int(0.5 * self.fs):  # Ensure at least 0.5 seconds of data
            logger.warning("Signal too short for QRS detection")
            return np.array([])

        # Clean the signal first
        cleaned = nk.ecg_clean(signal, sampling_rate=self.fs)

        # Detect R-peaks
        peaks = nk.ecg_peaks(cleaned, sampling_rate=self.fs)[1]["ECG_R_Peaks"]

        if len(peaks) == 0:
            logger.warning("No QRS peaks detected")
            return np.array([])

        return peaks.astype(int)

    except Exception as e:
        logger.error(f"Error detecting QRS peaks: {str(e)}")
        return np.array([])


def calculate_baseline_metrics(signal: np.ndarray) -> Dict:
    """Calculate baseline wander metrics."""
    try:
        from scipy import signal as sig

        # Use scipy's detrend instead of custom implementation
        detrended = sig.detrend(signal)
        baseline = signal - detrended

        return {
            "baseline_mean": float(np.mean(np.abs(baseline))),
            "baseline_std": float(np.std(baseline)),
            "baseline_max": float(np.max(np.abs(baseline))),
        }
    except Exception as e:
        logger.error(f"Error calculating baseline metrics: {str(e)}")
        return {"baseline_mean": None, "baseline_std": None, "baseline_max": None}
