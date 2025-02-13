"""
PyTorch‐centric feature extraction utilities for ECG signal processing.

This module provides functions for transforming waveforms into features using PyTorch
for numerical calculations. Where possible, torch operations are used in place of NumPy,
but external libraries (e.g. PyWavelets, NeuroKit2, SciPy) are still employed when needed.
"""

import torch
import numpy as np
import pywt
import neurokit2 as nk
from scipy import signal, stats
from scipy.integrate import trapezoid
import logging
from typing import Dict, List, Optional, Union

# Assume your visualization functions are available (they expect numpy arrays)
from .visualization import plot_signal_comparison, plot_comprehensive_analysis

logger = logging.getLogger(__name__)


def extract_statistical_features(beat: Union[np.ndarray, torch.Tensor]) -> Dict:
    """
    Extract statistical features from a beat using PyTorch operations.

    Parameters
    ----------
    beat : np.ndarray or torch.Tensor
         Single heartbeat signal.

    Returns
    -------
    Dict
         Dictionary containing:
         - Basic statistics: mean, std, max, min, median, MAD
         - Shape statistics: skewness, kurtosis, peak factor
         - Energy metrics: energy, RMS, peak-to-RMS
         - Additional metrics: entropy, zero crossings, range, variance

    Raises
    ------
    ValueError
         If input is empty or contains invalid values.
    """
    try:
        # Convert input to a torch tensor if needed
        if not isinstance(beat, torch.Tensor):
            beat = torch.tensor(beat, dtype=torch.float32)
        if beat.numel() == 0:
            raise ValueError("Input array is empty")
        if not torch.isfinite(beat).all():
            raise ValueError("Input contains invalid values (inf or nan)")

        # Basic statistics
        mean_val = torch.mean(beat)
        std_val = torch.std(beat, unbiased=False)
        max_val = torch.max(beat)
        min_val = torch.min(beat)
        median_val = torch.median(beat)
        mad_val = torch.median(torch.abs(beat - median_val))

        basic_stats = {
            "mean": mean_val.item(),
            "std": std_val.item(),
            "max": max_val.item(),
            "min": min_val.item(),
            "median": median_val.item(),
            "mad": mad_val.item(),
        }

        # Shape statistics: Compute standardized moments
        # To avoid division by zero, ensure std_val != 0
        normalized = (beat - mean_val) / (std_val + 1e-10)
        skewness = torch.mean(normalized**3)
        kurtosis = torch.mean(normalized**4) - 3  # excess kurtosis

        # Peak factor: max(|beat|) / sqrt(mean(beat^2))
        rms_val = torch.sqrt(torch.mean(beat**2))
        peak_factor = torch.max(torch.abs(beat)) / (rms_val + 1e-10)

        shape_stats = {
            "skewness": skewness.item(),
            "kurtosis": kurtosis.item(),
            "peak_factor": peak_factor.item(),
        }

        # Energy metrics
        energy = torch.sum(beat**2)
        rms = rms_val
        # peak_to_rms is same as peak_factor above
        energy_metrics = {
            "energy": energy.item(),
            "rms": rms.item(),
            "peak_to_rms": peak_factor.item(),
        }

        # Additional metrics
        # Use torch.histc for histogram then compute probability distribution
        hist = torch.histc(beat, bins=50, min=float(min_val), max=float(max_val))
        hist_np = hist.detach().cpu().numpy()
        if np.sum(hist_np) == 0:
            entropy = 0.0
        else:
            p = hist_np / np.sum(hist_np)
            # Using scipy.stats to compute entropy over the histogram probabilities
            entropy = stats.entropy(p)
        # Zero-crossings using torch.diff and sign changes
        sign_beat = torch.sign(beat)
        zero_crossings = torch.sum(torch.abs(torch.diff((sign_beat + 1) / 2))).item()

        additional_metrics = {
            "entropy": float(entropy),
            "zero_crossings": int(zero_crossings),
            "range": (max_val - min_val).item(),
            "variance": torch.var(beat, unbiased=False).item(),
        }

        features = {
            **basic_stats,
            **shape_stats,
            **energy_metrics,
            **additional_metrics,
        }
        return features

    except Exception as e:
        logger.error(f"Error extracting statistical features: {str(e)}")
        raise


def extract_wavelet_features(
    beat: Union[np.ndarray, torch.Tensor], wavelet: str = "db4", level: int = 4
) -> Dict:
    """
    Extract wavelet features from a beat using PyWavelets.

    Parameters
    ----------
    beat : np.ndarray or torch.Tensor
         Single heartbeat signal.
    wavelet : str, optional
         Wavelet type (default is 'db4').
    level : int, optional
         Decomposition level (default is 4).

    Returns
    -------
    Dict
         Dictionary containing wavelet features.

    Raises
    ------
    ValueError
         If input is invalid or too short.
    """
    try:
        # Ensure beat is a NumPy array
        if isinstance(beat, torch.Tensor):
            beat = beat.detach().cpu().numpy()
        if not isinstance(beat, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if len(beat) < 2**level:
            raise ValueError(
                f"Signal length must be at least {2**level} for level {level} decomposition"
            )

        # Optionally adjust level to maximum allowed by the wavelet
        max_level = pywt.dwt_max_level(len(beat), pywt.Wavelet(wavelet).dec_len)
        level = min(level, max_level)
        coeffs = pywt.wavedec(beat, wavelet, level=level)
        features = {}
        total_energy = 0
        for i, coeff in enumerate(coeffs):
            energy = np.sum(np.square(coeff))
            total_energy += energy
            features[f"wavelet_energy_{i}"] = float(energy)
            hist, _ = np.histogram(coeff, bins="auto", density=True)
            features[f"wavelet_entropy_{i}"] = float(stats.entropy(hist))
            features[f"wavelet_mean_{i}"] = float(np.mean(coeff))
            features[f"wavelet_std_{i}"] = float(np.std(coeff))
            features[f"wavelet_max_{i}"] = float(np.max(np.abs(coeff)))
        for i in range(len(coeffs)):
            features[f"wavelet_relative_energy_{i}"] = float(
                features[f"wavelet_energy_{i}"] / total_energy
            )
        return features

    except Exception as e:
        logger.error(f"Error extracting wavelet features: {str(e)}")
        raise


def extract_morphological_features(
    beat: Union[np.ndarray, torch.Tensor], fs: float
) -> Dict:
    """
    Extract morphological features from a beat using NeuroKit2.

    Parameters
    ----------
    beat : np.ndarray or torch.Tensor
         Single heartbeat signal.
    fs : float
         Sampling frequency in Hz.

    Returns
    -------
    Dict
         Dictionary containing morphological features.

    Raises
    ------
    ValueError
         If input is invalid.
    """
    try:
        # Ensure beat is NumPy
        if isinstance(beat, torch.Tensor):
            beat = beat.detach().cpu().numpy()
        if not isinstance(beat, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("Sampling rate must be a positive number")
        if len(beat) < int(0.2 * fs):
            raise ValueError("Beat too short for morphological analysis")

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

        _, info = nk.ecg_process(beat, sampling_rate=fs)
        peaks = info["ECG_R_Peaks"]
        _, waves = nk.ecg_delineate(beat, peaks, sampling_rate=fs, method="peak")

        # QRS duration calculation (in milliseconds)
        qrs_onset = waves.get("ECG_Q_Onsets", [None])[0]
        qrs_offset = waves.get("ECG_S_Offsets", [None])[0]
        if qrs_onset is not None and qrs_offset is not None:
            qrs_duration = (qrs_offset - qrs_onset) / fs * 1000
        else:
            qrs_duration = None

        # Safe amplitude extraction (calls below)
        r_amplitude = _safe_amplitude(beat, waves, "ECG_R_Peaks")
        s_amplitude = _safe_amplitude(beat, waves, "ECG_S_Peaks")

        return {
            "p_wave_duration": _safe_duration(waves, "P_Onset", "P_Offset", fs),
            "qrs_duration": _safe_duration(waves, "QRS_Onset", "QRS_Offset", fs),
            "t_wave_duration": _safe_duration(waves, "T_Onset", "T_Offset", fs),
            "pr_interval": _safe_duration(waves, "P_Onset", "QRS_Onset", fs),
            "qt_interval": _safe_duration(waves, "QRS_Onset", "T_Offset", fs),
            "r_amplitude": float(np.max(waves)) if waves.get("ECG_R_Peaks") else None,
            "s_amplitude": float(np.min(waves)) if waves.get("ECG_S_Peaks") else None,
            "p_amplitude": _safe_amplitude(waves, "P_Peak"),
            "t_amplitude": _safe_amplitude(waves, "T_Peak"),
            "st_level": _calculate_st_level(beat, waves, fs),
        }

    except Exception as e:
        logger.error(f"Error extracting morphological features: {str(e)}")
        return {}


def extract_stft_features(
    beat: Union[np.ndarray, torch.Tensor], fs: float, nperseg: int = 128
) -> Dict:
    """
    Extract Short-Time Fourier Transform features from a beat using torch.stft.

    Parameters
    ----------
    beat : np.ndarray or torch.Tensor
         Single heartbeat signal.
    fs : float
         Sampling frequency in Hz.
    nperseg : int, optional
         Length of each segment (default is 128).

    Returns
    -------
    Dict
         Dictionary containing:
         - Energy statistics
         - Frequency band powers
         - Spectral shape metrics

    Raises
    ------
    ValueError
         If input parameters are invalid.
    """
    try:
        if not isinstance(beat, torch.Tensor):
            beat = torch.tensor(beat, dtype=torch.float32)
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("Sampling rate must be positive")
        if beat.numel() < nperseg:
            raise ValueError(f"Signal length must be at least {nperseg}")

        hop_length = nperseg // 2
        window = torch.hann_window(nperseg)
        X = torch.stft(
            beat,
            n_fft=nperseg,
            hop_length=hop_length,
            win_length=nperseg,
            window=window,
            center=True,
            return_complex=True,
        )
        magnitude = torch.abs(X)  # shape: (n_freq, n_time)

        energy_stats = {
            "stft_mean_energy": magnitude.mean().item(),
            "stft_max_energy": magnitude.max().item(),
            "stft_std_energy": magnitude.std().item(),
            "stft_total_energy": magnitude.sum().item(),
        }

        # Default frequency bands
        freq_bands = {"low": (0, 5), "medium": (5, 20), "high": (20, 50)}
        # Create frequency bins for a one-sided STFT
        f = torch.linspace(0, fs / 2, steps=nperseg // 2 + 1)

        band_powers = {}
        for band_name, (low, high) in freq_bands.items():
            mask = (f >= low) & (f <= high)
            if mask.sum() > 0:
                band_power = torch.sum(magnitude[mask, :]) / magnitude.numel()
                band_powers[f"stft_power_{band_name}"] = band_power.item()
            else:
                band_powers[f"stft_power_{band_name}"] = 0.0

        # Spectral shape metrics using torch helper functions
        shape_metrics = {
            "stft_spectral_centroid": _spectral_centroid(f, magnitude),
            "stft_spectral_bandwidth": _spectral_bandwidth(f, magnitude),
            "stft_spectral_rolloff": _spectral_rolloff(magnitude),
        }

        return {**energy_stats, **band_powers, **shape_metrics}

    except Exception as e:
        logger.error(f"Error extracting STFT features: {str(e)}")
        raise


def extract_hybrid_features(
    beat: Union[np.ndarray, torch.Tensor],
    fs: float,
    wavelet: str = "db4",
    level: int = 4,
    nperseg: int = 128,
) -> Dict:
    """
    Extract hybrid features by combining statistical, wavelet, morphological, and STFT features.

    Parameters
    ----------
    beat : np.ndarray or torch.Tensor
         Single heartbeat signal.
    fs : float
         Sampling frequency in Hz.
    wavelet : str, optional
         Wavelet type (default is 'db4').
    level : int, optional
         Decomposition level (default is 4).
    nperseg : int, optional
         STFT segment length (default is 128).

    Returns
    -------
    Dict
         Dictionary containing combined features.
    """
    try:
        if not isinstance(beat, (np.ndarray, torch.Tensor)):
            raise ValueError("Input must be a numpy array or a tensor")
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("Sampling rate must be positive")

        features = {}
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

        if not features:
            raise ValueError("No features could be extracted from the signal")
        return features

    except Exception as e:
        logger.error(f"Error in hybrid feature extraction: {str(e)}")
        raise


# --- Helper Functions (Morphological) ---


def _safe_duration(
    waves: Dict, start_key: str, end_key: str, fs: float
) -> Optional[float]:
    """Safely calculate duration between two wave points."""
    try:
        if waves.get(start_key) is not None and waves.get(end_key) is not None:
            return float((waves[end_key] - waves[start_key]) / fs)
        return None
    except (KeyError, TypeError):
        return None


def _safe_interval(waves: Dict, end_key: str, start_key: str) -> Optional[float]:
    """Safely calculate interval between two wave points."""
    if waves.get(end_key) and waves.get(start_key):
        return float(waves[end_key] - waves[start_key])
    return None


def _safe_amplitude(
    signal_or_waves: Union[np.ndarray, Dict], key: str
) -> Optional[float]:
    """Safely get amplitude at a specific wave point."""
    try:
        # When passing a waveform (e.g. beat) and key from delineated waves
        if isinstance(signal_or_waves, dict):
            if signal_or_waves.get(key) is not None:
                return float(signal_or_waves[key])
        elif isinstance(signal_or_waves, np.ndarray):
            # key is assumed to be an index here
            if signal_or_waves[int(key)]:
                return float(signal_or_waves[int(key)])
    except Exception as e:
        logger.warning(f"Failed to safely extract amplitude for {key}: {str(e)}")
    return None


def _calculate_st_level(beat: np.ndarray, waves: Dict, fs: float) -> Optional[float]:
    """Calculate ST segment level relative to baseline."""
    try:
        if waves.get("R_offset") is not None and waves.get("T_onset") is not None:
            st_start = int(waves["R_offset"])
            st_end = int(waves["T_onset"])
            st_segment = beat[st_start:st_end]
            return float(np.mean(st_segment))
    except Exception as e:
        logger.error(f"Error calculating ST level: {str(e)}")
    return None


def _calculate_wave_symmetry(
    beat: np.ndarray, waves: Dict, start_key: str, end_key: str
) -> Optional[float]:
    """Calculate symmetry of a wave segment."""
    try:
        if waves.get(start_key) is not None and waves.get(end_key) is not None:
            start = int(waves[start_key])
            end = int(waves[end_key])
            wave = beat[start:end]
            if len(wave) > 1:
                mid = len(wave) // 2
                left = wave[:mid]
                right = wave[mid:][::-1]
                return float(np.corrcoef(left, right)[0, 1])
    except Exception as e:
        logger.warning(f"Failed to calculate wave symmetry: {str(e)}")
    return None


# --- Helper Functions (Spectral Analysis using torch) ---


def _spectral_centroid(freqs: torch.Tensor, magnitude: torch.Tensor) -> float:
    """
    Calculate spectral centroid using torch operations.
    Parameters
    ----------
    freqs : torch.Tensor
         1D tensor of frequency values.
    magnitude : torch.Tensor
         2D tensor of magnitudes (freq x time).
    Returns
    -------
    float
         Spectral centroid.
    """
    mag_sum = torch.sum(magnitude, dim=1)
    centroid = torch.sum(freqs * mag_sum) / (torch.sum(mag_sum) + 1e-10)
    return centroid.item()


def _spectral_bandwidth(freqs: torch.Tensor, magnitude: torch.Tensor) -> float:
    """Calculate spectral bandwidth using torch."""
    centroid = _spectral_centroid(freqs, magnitude)
    mag_sum = torch.sum(magnitude, dim=1)
    variance = torch.sum(((freqs - centroid) ** 2) * mag_sum) / (
        torch.sum(mag_sum) + 1e-10
    )
    return torch.sqrt(variance).item()


def _spectral_rolloff(magnitude: torch.Tensor, percentile: float = 0.85) -> float:
    """
    Calculate the rolloff frequency bin index using torch.
    """
    power = torch.sum(magnitude**2, dim=1)
    cumulative = torch.cumsum(power, dim=0)
    threshold = percentile * cumulative[-1]
    idx = (cumulative >= threshold).nonzero(as_tuple=False)
    rolloff = idx[0].item() if idx.numel() > 0 else 0
    return float(rolloff)


# --- Remaining Functions (Morphological & Advanced) ---
# For functions using neurokit2, plotting, or extensive NumPy work, we retain NumPy.
# They convert torch input to numpy if needed.


def analyze_qt_interval(beat: np.ndarray, waves: Dict, fs: float) -> Dict:
    """
    Analyze QT interval and its variants using NumPy.
    (Retained as in original.)
    """
    try:
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("Sampling frequency must be positive")
        if not isinstance(waves, dict):
            raise ValueError("Waves must be a dictionary")
        required_points = ["Q_start", "T_end", "R_start", "R_end"]
        if not all(
            point in waves and waves[point] is not None for point in required_points
        ):
            return {}
        qt_samples = waves["T_end"] - waves["Q_start"]
        rr_samples = waves["R_end"] - waves["R_start"]
        qt_sec = qt_samples / fs
        rr_sec = rr_samples / fs
        if qt_sec <= 0 or rr_sec <= 0:
            return {}
        hr = 60 / rr_sec
        qtc_bazett = qt_sec / np.sqrt(rr_sec)
        qtc_fridericia = qt_sec / np.cbrt(rr_sec)
        qtc_framingham = qt_sec + 0.154 * (1 - rr_sec)
        jt_sec = None
        jtc = None
        if "J_point" in waves and waves["J_point"] is not None:
            jt_samples = waves["T_end"] - waves["J_point"]
            jt_sec = jt_samples / fs
            jtc = jt_sec / np.sqrt(rr_sec)
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
    Detect and quantify T-wave alternans (TWA) using NumPy.
    """
    try:
        if len(beats) < 4:
            return {}
        t_wave_amplitudes = []
        for beat, wave in zip(beats, waves):
            t_start = wave.get("T_start")
            t_end = wave.get("T_end")
            if t_start is not None and t_end is not None:
                t_wave = beat[t_start:t_end]
                t_wave_amplitudes.append(np.max(t_wave))
        if len(t_wave_amplitudes) < 4:
            return {}
        t_wave_amplitudes = np.array(t_wave_amplitudes)
        even_beats = t_wave_amplitudes[::2]
        odd_beats = t_wave_amplitudes[1::2]
        twa_magnitude = np.mean(np.abs(even_beats - odd_beats))
        twa_ratio = twa_magnitude / np.mean(t_wave_amplitudes)
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
    Analyze ST segment characteristics using NumPy.
    """
    try:
        j_point = waves.get("J_point")
        t_start = waves.get("T_start")
        if j_point is None or t_start is None:
            return {}
        st_segment = beat[j_point:t_start]
        pr_start = waves.get("P_start")
        pr_end = waves.get("P_end")
        baseline = (
            np.mean(beat[pr_start:pr_end])
            if pr_start is not None and pr_end is not None
            else 0
        )
        st_level = np.mean(st_segment[: int(0.04 * fs)]) - baseline
        st_slope = np.polyfit(np.arange(len(st_segment)), st_segment, 1)[0]
        st_integral = trapezoid(st_segment - baseline) / fs
        st_shape = _classify_st_shape(st_segment, st_slope)
        return {
            "ST_level": float(st_level),
            "ST_slope": float(st_slope),
            "ST_integral": float(st_integral),
            "ST_shape": st_shape,
            "ST_elevation": bool(st_level > 0.1),
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
        mean_diff = np.mean(np.abs(even_beats - odd_beats))
        noise = np.std(even_beats - odd_beats)
        if noise < 1e-10:
            noise = 1e-10
        return float(mean_diff / noise)
    except Exception as e:
        logger.error(f"Error calculating K-score: {str(e)}")
        return 0.0


def _classify_st_shape(st_segment: np.ndarray, threshold: float = 0.1) -> str:
    """Classify ST segment shape based on slope."""
    try:
        if len(st_segment) < 2:
            return "horizontal"
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


def calculate_baseline_metrics(signal: np.ndarray) -> Dict:
    """Calculate baseline wander metrics using SciPy."""
    try:
        from scipy import signal as sig

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


# --- Test Functions ---


def test_extract_statistical_features():
    try:
        t = np.linspace(0, 1, 100)
        beat = np.sin(2 * np.pi * 2 * t) * np.exp(-2 * t)
        features = extract_statistical_features(beat)
        print("Statistical Features:")
        for key, value in features.items():
            print(f"  {key}: {value}")
    except Exception as e:
        logger.error(f"Error in statistical features test: {str(e)}")


def test_extract_wavelet_features():
    try:
        t = np.linspace(0, 1, 128)
        beat = np.sin(2 * np.pi * 2 * t) * np.exp(-2 * t)
        print("\nTesting wavelet decomposition with level 2:")
        features_l2 = extract_wavelet_features(beat, wavelet="db4", level=2)
        print("Wavelet Features (Level 2):")
        for key, value in features_l2.items():
            print(f"  {key}: {value}")
        print("\nTesting wavelet decomposition with level 4:")
        features_l4 = extract_wavelet_features(beat, wavelet="db4", level=4)
        print("Wavelet Features (Level 4):")
        for key, value in features_l4.items():
            print(f"  {key}: {value}")
    except Exception as e:
        logger.error(f"Error extracting wavelet features: {str(e)}")


def test_extract_morphological_features():
    fs = 500
    beat = nk.ecg_simulate(duration=1, sampling_rate=fs)
    features = extract_morphological_features(beat, fs)
    print("Morphological Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")


def test_extract_stft_features():
    fs = 500
    t = np.linspace(0, 1, 500, endpoint=False)
    beat = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(len(t))
    features = extract_stft_features(beat, fs, nperseg=128)
    print("STFT Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")


def run_tests():
    print("Running tests for feature extraction functions...\n")
    test_extract_statistical_features()
    print("\n" + "-" * 40 + "\n")
    test_extract_wavelet_features()
    print("\n" + "-" * 40 + "\n")
    test_extract_morphological_features()
    print("\n" + "-" * 40 + "\n")
    test_extract_stft_features()


if __name__ == "__main__":
    run_tests()

    # --- Plotting dummy data for comprehensive analysis ---
    fs = 500
    duration = 10
    t = np.linspace(0, duration, duration * fs, endpoint=False)
    original_signal = np.sin(2 * np.pi * 1 * t) + 0.2 * np.random.randn(len(t))
    processed_signal = np.sin(2 * np.pi * 1 * t)
    plot_signal_comparison(
        original=original_signal,
        processed=processed_signal,
        fs=fs,
        title="Dummy ECG Signal Comparison",
    )

    dummy_peaks = np.arange(0, len(processed_signal), fs)
    dummy_rr_intervals = np.diff(dummy_peaks) / fs * 1000
    dummy_frequencies = np.linspace(0.1, 30, 100)
    dummy_psd = np.abs(np.sin(dummy_frequencies))
    results = {
        "original_signal": original_signal,
        "processed_signal": processed_signal,
        "peaks": dummy_peaks,
        "hrv_metrics": {"rr_intervals": dummy_rr_intervals},
        "frequencies": dummy_frequencies,
        "psd": dummy_psd,
    }
    plot_comprehensive_analysis(results)
