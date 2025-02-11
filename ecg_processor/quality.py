import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


# --- Your calculate_signal_quality Function ---
def calculate_signal_quality(
    original: np.ndarray, processed: np.ndarray, fs: float
) -> Dict:
    """
    Calculate comprehensive signal quality metrics for ECG signals with improved
    numerical stability and additional metrics.

    Parameters
    ----------
    original : np.ndarray
        Original ECG signal
    processed : np.ndarray
        Processed (filtered/denoised) ECG signal
    fs : float
        Sampling frequency in Hz

    Returns
    -------
    Dict
        Dictionary containing various quality metrics:
        - SNR and power metrics
        - Time domain statistics
        - Noise characteristics
        - Baseline wander metrics
        - Frequency domain metrics
        - Signal complexity measures

    Raises
    ------
    ValueError
        If inputs are invalid or incompatible
    """
    try:
        # Input validation with detailed error messages
        if not isinstance(original, np.ndarray) or not isinstance(
            processed, np.ndarray
        ):
            raise ValueError("Signals must be numpy arrays")
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("Sampling frequency must be positive")
        if original.shape != processed.shape:
            raise ValueError(
                f"Signal shape mismatch: {original.shape} vs {processed.shape}"
            )
        if not np.isfinite(original).all() or not np.isfinite(processed).all():
            raise ValueError("Signals contain invalid values (inf or nan)")
        if len(original) < fs:
            raise ValueError(f"Signal too short. Must be at least {fs} samples")

        # Convert to float64 for better numerical precision
        original = original.astype(np.float64)
        processed = processed.astype(np.float64)

        # 1. Signal-to-Noise Ratio and Power Metrics
        try:
            noise = original - processed
            # Use RMS values for more stable power calculations
            signal_rms = np.sqrt(np.mean(processed**2))
            noise_rms = np.sqrt(np.mean(noise**2))

            # Calculate SNR with protection against very small noise
            epsilon = np.finfo(float).eps
            if noise_rms > epsilon:
                snr = 20 * np.log10(signal_rms / noise_rms)
            else:
                snr = 100.0  # Arbitrary high value for very clean signals

            power_metrics = {
                "snr": float(snr),
                "signal_rms": float(signal_rms),
                "noise_rms": float(noise_rms),
                "signal_power": float(signal_rms**2),
                "noise_power": float(noise_rms**2),
            }

        except Exception as e:
            logger.error(f"Error calculating power metrics: {str(e)}")
            power_metrics = {
                "snr": np.nan,
                "signal_rms": np.nan,
                "noise_rms": np.nan,
                "signal_power": np.nan,
                "noise_power": np.nan,
            }

        # 2. Time Domain Statistical Metrics
        try:
            abs_proc = np.abs(processed)
            statistical_metrics = {
                "mean": float(np.mean(processed)),
                "std": float(np.std(processed)),
                "kurtosis": float(stats.kurtosis(processed)),
                "skewness": float(stats.skew(processed)),
                "peak_to_peak": float(np.ptp(processed)),
                "zero_crossings": int(np.sum(np.diff(np.signbit(processed)))),
                "crest_factor": float(np.max(abs_proc) / (signal_rms + epsilon)),
            }

        except Exception as e:
            logger.error(f"Error calculating statistical metrics: {str(e)}")
            statistical_metrics = {
                "mean": np.nan,
                "std": np.nan,
                "kurtosis": np.nan,
                "skewness": np.nan,
                "peak_to_peak": np.nan,
                "zero_crossings": 0,
                "crest_factor": np.nan,
            }

        # 3. Advanced Noise Assessment
        try:
            # Adaptive window size based on sampling rate
            window_size = max(int(0.1 * fs), 10)  # At least 10 samples
            stride = max(window_size // 4, 1)  # 75% overlap

            # Calculate noise levels using multiple methods
            noise_levels = []
            diff_levels = []
            envelope_var = []

            for i in range(0, len(processed) - window_size, stride):
                window = processed[i : i + window_size]
                # Standard deviation of window
                noise_levels.append(np.std(window))
                # First difference variation
                diff_levels.append(np.std(np.diff(window)))
                # Envelope variation (using Hilbert transform)
                try:
                    envelope = np.abs(signal.hilbert(window))
                    envelope_var.append(np.std(envelope))
                except Exception:
                    envelope_var.append(np.nan)

            noise_metrics = {
                "noise_std": float(np.std(noise_levels)),
                "noise_max": float(np.max(noise_levels)),
                "noise_mean": float(np.mean(noise_levels)),
                "diff_std": float(np.std(diff_levels)),
                "envelope_std": float(np.nanstd(envelope_var)),
                "noise_percentile_95": float(np.percentile(noise_levels, 95)),
            }

        except Exception as e:
            logger.error(f"Error calculating noise metrics: {str(e)}")
            noise_metrics = {
                "noise_std": np.nan,
                "noise_max": np.nan,
                "noise_mean": np.nan,
                "diff_std": np.nan,
                "envelope_std": np.nan,
                "noise_percentile_95": np.nan,
            }

        # 4. Enhanced Baseline Wander Assessment
        try:
            # Remove linear and polynomial trends
            linear_detrend = signal.detrend(processed, type="linear")
            poly_detrend = signal.detrend(processed, type="polynomial", deg=2)

            # Low-frequency content
            window_length = min(int(fs), len(processed))
            if window_length % 2 == 0:
                window_length -= 1
            baseline = signal.savgol_filter(processed, window_length, 2)

            baseline_metrics = {
                "baseline_drift_linear": float(np.std(linear_detrend)),
                "baseline_drift_poly": float(np.std(poly_detrend)),
                "baseline_power": float(np.sum(baseline**2)),
                "baseline_max_dev": float(np.max(np.abs(baseline))),
                "baseline_crossing_rate": float(np.mean(np.abs(np.diff(baseline > 0)))),
            }

        except Exception as e:
            logger.error(f"Error calculating baseline metrics: {str(e)}")
            baseline_metrics = {
                "baseline_drift_linear": np.nan,
                "baseline_drift_poly": np.nan,
                "baseline_power": np.nan,
                "baseline_max_dev": np.nan,
                "baseline_crossing_rate": np.nan,
            }

        # 5. Enhanced Frequency Domain Analysis
        try:
            # Use Welch's method with appropriate window size
            nperseg = min(len(processed), int(4 * fs))  # 4-second windows or shorter
            freqs, psd = signal.welch(processed, fs, nperseg=nperseg)
            total_power = _calculate_power_band(freqs, psd, 0, fs / 2)

            # Calculate power in physiologically relevant bands
            freq_metrics = {
                "vlf_power": float(
                    _calculate_power_band(freqs, psd, 0, 0.04) / total_power
                ),
                "lf_power": float(
                    _calculate_power_band(freqs, psd, 0.04, 0.15) / total_power
                ),
                "hf_power": float(
                    _calculate_power_band(freqs, psd, 0.15, 0.4) / total_power
                ),
                "line_noise": float(
                    _calculate_power_band(freqs, psd, 45, 55) / total_power
                ),
                "high_freq_noise": float(
                    _calculate_power_band(freqs, psd, 100, fs / 2) / total_power
                ),
            }

            # Spectral entropy
            psd_norm = psd / np.sum(psd)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + epsilon))
            freq_metrics["spectral_entropy"] = float(spectral_entropy)

        except Exception as e:
            logger.error(f"Error calculating frequency metrics: {str(e)}")
            freq_metrics = {
                "vlf_power": np.nan,
                "lf_power": np.nan,
                "hf_power": np.nan,
                "line_noise": np.nan,
                "high_freq_noise": np.nan,
                "spectral_entropy": np.nan,
            }

        # 6. Signal Complexity Measures
        try:
            # Approximate entropy parameters
            m = 2  # embedding dimension
            r = 0.2 * np.std(processed)  # tolerance

            # Calculate sample entropy for small segments
            n_segments = min(5, len(processed) // fs)
            if n_segments > 0:
                segment_length = len(processed) // n_segments
                entropy_values = []
                for i in range(n_segments):
                    start = i * segment_length
                    end = start + segment_length
                    segment = processed[start:end]
                    try:
                        # Simplified entropy calculation for speed
                        diff_matrix = np.abs(segment[:, None] - segment)
                        cmm = np.sum(diff_matrix <= r, axis=1)
                        entropy = np.mean(np.log(cmm + epsilon))
                        entropy_values.append(entropy)
                    except Exception:
                        continue

                complexity_metrics = {
                    "entropy_mean": float(np.mean(entropy_values))
                    if entropy_values
                    else np.nan,
                    "entropy_std": float(np.std(entropy_values))
                    if len(entropy_values) > 1
                    else np.nan,
                }
            else:
                complexity_metrics = {"entropy_mean": np.nan, "entropy_std": np.nan}

        except Exception as e:
            logger.error(f"Error calculating complexity metrics: {str(e)}")
            complexity_metrics = {"entropy_mean": np.nan, "entropy_std": np.nan}

        # Combine all metrics
        metrics = {
            **power_metrics,
            **statistical_metrics,
            **noise_metrics,
            **baseline_metrics,
            **freq_metrics,
            **complexity_metrics,
            "quality_score": float(np.clip(snr / 20, 0, 1)),  # Normalized quality score
        }

        return metrics

    except Exception as e:
        logger.error(f"Error calculating signal quality: {str(e)}")
        raise


# --- Plotting Function (similar to your uploaded version) ---
def plot_signal_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    fs: float,
    title: str = "ECG Signal Comparison",
    show_metrics: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot original and processed signals with quality metrics.

    Parameters
    ----------
    original : np.ndarray
        Original ECG signal
    processed : np.ndarray
        Processed ECG signal
    fs : float
        Sampling frequency in Hz
    title : str, optional
        Plot title
    show_metrics : bool, optional
        Whether to show quality metrics in the plot

    Returns
    -------
    Optional[plt.Figure]
        The matplotlib figure if show=False, None otherwise

    Raises
    ------
    ValueError
        If signals have different lengths or invalid values
    """
    try:
        if len(original) != len(processed):
            raise ValueError("Signals must have the same length")

        time = np.arange(len(original)) / fs
        fig = plt.figure(figsize=(15, 8))

        # Original signal plot
        ax1 = plt.subplot(3, 1, 1)
        plt.plot(time, original, color="tab:blue", label="Original")
        plt.title("Original Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        # Processed signal plot
        _ = plt.subplot(3, 1, 2, sharex=ax1)
        plt.plot(time, processed, color="tab:orange", label="Processed")
        plt.title("Processed Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        # Noise plot
        _ = plt.subplot(3, 1, 3, sharex=ax1)
        noise = original - processed
        plt.plot(time, noise, color="tab:red", label="Noise")
        plt.title("Extracted Noise")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        if show_metrics:
            # Calculate and display quality metrics
            metrics = calculate_signal_quality(original, processed, fs)
            metrics_text = (
                f"SNR: {metrics['snr']:.2f} dB\n"
                f"Signal RMS: {metrics['signal_rms']:.2f}\n"
                f"Noise RMS: {metrics['noise_rms']:.2f}"
            )
            plt.figtext(
                0.02,
                0.02,
                metrics_text,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8),
            )

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if show_metrics:
            plt.show()
            return None
        return fig

    except Exception as e:
        logger.error(f"Error plotting signal comparison: {str(e)}")
        raise


def _calculate_power_line_noise(ecg_signal: np.ndarray, fs: float) -> Dict:
    """
    Calculate the power of power line noise (e.g. 45-55 Hz) in the signal.

    Parameters
    ----------
    ecg_signal : np.ndarray
        Input ECG signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    Dict
        Dictionary containing the 'power_line_noise' value.
    """
    try:
        # Compute the power spectral density using Welch's method.
        freqs, psd = signal.welch(
            ecg_signal, fs, nperseg=min(len(ecg_signal), int(4 * fs))
        )
        # Define the power line frequency range (e.g., 45 to 55 Hz)
        mask = (freqs >= 45) & (freqs <= 55)
        # Sum the power within this band.
        power_line = np.sum(psd[mask])
        return {"power_line_noise": float(power_line)}
    except Exception as e:
        logger.error(f"Error calculating power line noise: {str(e)}")
        return {"power_line_noise": 0.0}


def _calculate_power_band(
    freqs: np.ndarray, psd: np.ndarray, low_freq: float, high_freq: float
) -> float:
    """Calculate power in a specific frequency band."""
    if not isinstance(freqs, np.ndarray) or not isinstance(psd, np.ndarray):
        raise ValueError("Frequencies and PSD must be numpy arrays")
    if not np.isfinite(freqs).all() or not np.isfinite(psd).all():
        raise ValueError("Input arrays contain non-finite values")
    if low_freq >= high_freq:
        raise ValueError("Low frequency must be less than high frequency")
    if low_freq < 0 or high_freq < 0:
        raise ValueError("Frequencies must be non-negative")

    try:
        # Find indices for the frequency band
        band_idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)

        # Calculate power in band
        band_power = np.trapz(psd[band_idx], freqs[band_idx])

        return float(max(band_power, 0.0))
    except Exception as e:
        logger.error(f"Error calculating power band: {str(e)}")
        return 0.0


def _detect_powerline_interference(ecg_signal: np.ndarray, fs: float) -> Dict:
    """Detect powerline interference with improved sensitivity."""
    if not isinstance(ecg_signal, np.ndarray):
        raise ValueError("Signal must be a numpy array")
    if not np.isfinite(ecg_signal).all():
        raise ValueError("Signal contains non-finite values")
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    if len(ecg_signal) < fs:
        raise ValueError("Signal too short for interference detection")

    try:
        # Calculate power spectrum
        freqs, psd = signal.welch(ecg_signal, fs, nperseg=min(len(ecg_signal), fs * 2))

        # Calculate power in powerline frequency bands (50/60 Hz)
        powerline_freqs = [50, 60]
        band_width = 2  # Hz

        powerline_power = sum(
            _calculate_power_band(freqs, psd, f - band_width / 2, f + band_width / 2)
            for f in powerline_freqs
        )
        total_power = np.sum(psd)

        # Calculate interference ratio
        interference_ratio = powerline_power / total_power if total_power > 0 else 0.0

        return {
            "powerline_interference_ratio": float(np.clip(interference_ratio, 0, 1)),
            "powerline_interference_present": bool(interference_ratio > 0.1),
        }
    except Exception as e:
        logger.error(f"Error detecting powerline interference: {str(e)}")
        return {
            "powerline_interference_ratio": 1.0,
            "powerline_interference_present": True,
        }


def _assess_rpeak_quality(ecg_signal: np.ndarray, waves: Dict) -> Dict:
    """Assess R-peak detection quality.

    Raises:
        ValueError: if the signal is invalid, the waves dictionary is missing the
                    'R_peaks' key, if there are fewer than 2 R-peaks, or if any R-peak index is out of range.
    """
    if not isinstance(ecg_signal, np.ndarray):
        raise ValueError("Signal must be a numpy array")
    if not isinstance(waves, dict):
        raise ValueError("Waves must be a dictionary")
    if not np.isfinite(ecg_signal).all():
        raise ValueError("Signal contains non-finite values")
    if "R_peaks" not in waves:
        raise ValueError("Waves dictionary must contain 'R_peaks' key")

    r_peaks = waves["R_peaks"]
    # Instead of returning default metrics, raise an error if there are not enough R-peaks.
    if len(r_peaks) < 2:
        raise ValueError("Insufficient R-peaks detected for quality assessment")
    # Ensure that all R-peak indices are valid for the supplied signal.
    if np.any(r_peaks >= len(ecg_signal)):
        raise ValueError("R-peaks indices are out of range for the provided signal")

    try:
        # Calculate R-peak amplitudes and intervals
        amplitudes = ecg_signal[r_peaks]
        rr_intervals = np.diff(r_peaks)

        # Calculate variability metrics
        amplitude_std = np.std(amplitudes)
        mean_amplitude = np.mean(amplitudes)
        rr_std = np.std(rr_intervals)
        mean_rr = np.mean(rr_intervals)

        # Normalized metrics, with protection against divide-by-zero
        amplitude_variability = (
            amplitude_std / mean_amplitude if mean_amplitude > 0 else 1.0
        )
        timing_variability = rr_std / mean_rr if mean_rr > 0 else 1.0

        detection_quality = 1.0 - (amplitude_variability + timing_variability) / 2

        return {
            "rpeak_detection_quality": float(np.clip(detection_quality, 0, 1)),
            "rpeak_amplitude_variability": float(np.clip(amplitude_variability, 0, 1)),
            "rpeak_timing_variability": float(np.clip(timing_variability, 0, 1)),
        }
    except Exception as e:
        logger.error(f"Error in R-peak quality assessment: {str(e)}")
        return {
            "rpeak_detection_quality": 0.0,
            "rpeak_amplitude_variability": 1.0,
            "rpeak_timing_variability": 1.0,
        }


def _calculate_signal_complexity(ecg_signal: np.ndarray) -> Dict:
    """Calculate signal complexity metrics.

    Raises:
        ValueError: if the input signal is too short (less than 5 samples)
                    or invalid.
    """
    if not isinstance(ecg_signal, np.ndarray):
        raise ValueError("Signal must be a numpy array")
    if not np.isfinite(ecg_signal).all():
        raise ValueError("Signal contains non-finite values")
    # Adjusted minimum length requirement (e.g., less than 5 samples is too short)
    if len(ecg_signal) < 5:
        raise ValueError("Signal too short for complexity calculation")

    try:
        # Calculate zero crossings (divide by (2 * length) to normalize)
        zero_crossings = np.sum(np.diff(np.signbit(ecg_signal - np.mean(ecg_signal))))
        complexity = zero_crossings / (2 * len(ecg_signal))

        return {"signal_complexity": float(np.clip(complexity, 0, 1))}
    except Exception as e:
        logger.error(f"Error calculating signal complexity: {str(e)}")
        return {"signal_complexity": 0.0}


def assess_beat_quality(beat: np.ndarray, fs: float) -> Tuple[bool, Dict]:
    """Assess the quality of a single heartbeat."""
    if not isinstance(beat, np.ndarray):
        raise ValueError("Beat must be a numpy array")
    if not np.isfinite(beat).all():
        raise ValueError("Beat contains non-finite values")
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    if len(beat) < fs / 2:  # At least 0.5 seconds of data
        raise ValueError("Beat too short for quality assessment")

    try:
        # Calculate quality metrics for the beat
        metrics = {}
        metrics.update(_calculate_snr(beat, fs))
        metrics.update(_assess_baseline_wander(beat, fs))
        metrics.update(_detect_powerline_interference(beat, fs))
        metrics.update(_calculate_signal_complexity(beat))

        # Normalize metrics
        metrics = _normalize_metrics(metrics)

        # Determine if beat is good quality
        is_good = (
            metrics["SNR"] > 10
            and metrics["baseline_wander_severity"] < 0.3
            and metrics["powerline_interference_ratio"] < 0.2
            and metrics["signal_complexity"] < 0.7
        )

        return bool(is_good), metrics

    except Exception as e:
        logger.error(f"Error in beat quality assessment: {str(e)}")
        return False, {}


def assess_signal_quality(
    ecg_signal: np.ndarray, fs: float, waves: Optional[Dict] = None
) -> Dict:
    """Assess ECG signal quality with comprehensive metrics."""
    try:
        # Input validation
        if not isinstance(ecg_signal, np.ndarray):
            raise ValueError("Signal must be a numpy array")
        if not np.isfinite(ecg_signal).all():
            raise ValueError("Signal contains non-finite values")
        if fs <= 0:
            raise ValueError("Sampling frequency must be positive")
        if waves is not None and not isinstance(waves, dict):
            raise ValueError("Waves must be a dictionary if provided")

        # Calculate basic metrics
        metrics = {}
        metrics.update(_calculate_snr(ecg_signal, fs))
        metrics.update(_assess_baseline_wander(ecg_signal, fs))
        metrics.update(_detect_powerline_interference(ecg_signal, fs))
        metrics.update(_calculate_signal_complexity(ecg_signal))

        # Add R-peak quality if waves are provided
        if waves is not None:
            metrics.update(_assess_rpeak_quality(ecg_signal, waves))
        else:
            # Add default R-peak metrics
            metrics.update(
                {
                    "rpeak_detection_quality": 0.5,  # Neutral value when no waves provided
                    "rpeak_amplitude_variability": 0.5,
                    "rpeak_timing_variability": 0.5,
                }
            )

        # Normalize metrics
        metrics = _normalize_metrics(metrics)

        # Calculate overall quality
        metrics["overall_quality"] = _calculate_overall_quality(metrics)

        return metrics

    except Exception as e:
        logger.error(f"Error in signal quality assessment: {str(e)}")
        raise


def _calculate_chunk_metrics(chunk: np.ndarray, fs: float) -> Dict:
    """Calculate metrics for a single chunk of signal."""
    if not isinstance(chunk, np.ndarray):
        raise ValueError("Chunk must be a numpy array")
    if not np.isfinite(chunk).all():
        raise ValueError("Chunk contains non-finite values")
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")

    metrics = {}
    metrics.update(_calculate_snr(chunk, fs))
    metrics.update(_assess_baseline_wander(chunk, fs))
    metrics.update(_detect_powerline_interference(chunk, fs))
    return metrics


def _average_chunk_metrics(chunk_metrics: List[Dict]) -> Dict:
    """Average metrics across chunks."""
    if not chunk_metrics:
        raise ValueError("Empty chunk metrics list")
    if not all(isinstance(m, dict) for m in chunk_metrics):
        raise ValueError("All chunk metrics must be dictionaries")

    averaged_metrics = {}
    for key in chunk_metrics[0].keys():
        values = [metrics[key] for metrics in chunk_metrics if key in metrics]
        if not values:
            raise ValueError(f"No valid values found for metric {key}")
        if not all(np.isfinite(v) for v in values):
            raise ValueError(f"Non-finite values found for metric {key}")
        averaged_metrics[key] = float(np.mean(values))

    return averaged_metrics


def _normalize_metrics(metrics: Dict) -> Dict:
    """Ensure all metrics are within valid ranges."""
    if not isinstance(metrics, dict):
        raise ValueError("Metrics must be a dictionary")
    if not metrics:
        raise ValueError("Empty metrics dictionary")

    normalized = {}
    for key, value in metrics.items():
        if key.endswith("_stable") or key.endswith("_present"):
            if not isinstance(value, bool):
                raise ValueError(f"Invalid boolean value for metric {key}")
            normalized[key] = value
        else:
            if not isinstance(value, (int, float)):
                raise ValueError(f"Invalid value type for metric {key}")
            if not np.isfinite(value):
                raise ValueError(f"Non-finite value for metric {key}")

            if key == "SNR":
                normalized[key] = float(np.clip(value, 0, 40))
            else:
                normalized[key] = float(np.clip(value, 0, 1))

    return normalized


def _calculate_overall_quality(metrics: Dict) -> float:
    """Calculate overall quality score."""
    if not isinstance(metrics, dict):
        raise ValueError("Metrics must be a dictionary")
    if not metrics:
        raise ValueError("Empty metrics dictionary")

    required_metrics = [
        "SNR",
        "baseline_wander_severity",
        "powerline_interference_ratio",
        "signal_complexity",
        "rpeak_detection_quality",
    ]

    if not all(key in metrics for key in required_metrics):
        raise ValueError("Missing required metrics")

    if not all(isinstance(metrics[key], (int, float)) for key in required_metrics):
        raise ValueError("Invalid metric value types")

    if not all(np.isfinite(metrics[key]) for key in required_metrics):
        raise ValueError("Non-finite metric values")

    if not all(0 <= metrics[key] <= 1 for key in required_metrics if key != "SNR"):
        raise ValueError("Metric values out of range [0,1]")

    # Normalize SNR to [0,1]
    snr_norm = np.clip(metrics["SNR"] / 40, 0, 1)

    # Calculate weighted average
    weights = {
        "SNR": 0.3,
        "baseline_wander_severity": 0.2,
        "powerline_interference_ratio": 0.2,
        "signal_complexity": 0.1,
        "rpeak_detection_quality": 0.2,
    }

    quality_score = (
        weights["SNR"] * snr_norm
        + weights["baseline_wander_severity"]
        * (1 - metrics["baseline_wander_severity"])
        + weights["powerline_interference_ratio"]
        * (1 - metrics["powerline_interference_ratio"])
        + weights["signal_complexity"] * (1 - metrics["signal_complexity"])
        + weights["rpeak_detection_quality"] * metrics["rpeak_detection_quality"]
    )

    return float(np.clip(quality_score, 0, 1))


def _calculate_snr(ecg_signal: np.ndarray, fs: float) -> Dict:
    """Calculate Signal-to-Noise Ratio with improved accuracy for clean signals."""
    if not isinstance(ecg_signal, np.ndarray):
        raise ValueError("Signal must be a numpy array")
    if not np.isfinite(ecg_signal).all():
        raise ValueError("Signal contains non-finite values")
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")

    try:
        # Calculate signal power
        signal_rms = np.sqrt(np.mean(np.square(ecg_signal)))

        # Estimate noise using wavelet decomposition
        import pywt

        wavelet = "db4"
        level = 4
        coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)

        # Use detail coefficients as noise estimate
        noise = np.concatenate(
            [
                pywt.upcoef("d", c, wavelet, level=i + 1, take=len(ecg_signal))
                for i, c in enumerate(coeffs[1:])
            ]
        )
        noise_rms = np.sqrt(np.mean(np.square(noise)))

        # Calculate SNR
        snr = 20 * np.log10(signal_rms / noise_rms) if noise_rms > 0 else 40.0

        return {"SNR": float(snr)}
    except Exception as e:
        logger.error(f"Error calculating SNR: {str(e)}")
        return {"SNR": 0.0}


def _assess_baseline_wander(ecg_signal: np.ndarray, fs: float) -> Dict:
    """Assess baseline wander severity."""
    if not isinstance(ecg_signal, np.ndarray):
        raise ValueError("Signal must be a numpy array")
    if not np.isfinite(ecg_signal).all():
        raise ValueError("Signal contains non-finite values")
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")

    try:
        # Estimate baseline using low-pass filter
        from scipy.signal import butter, filtfilt

        nyq = fs / 2
        cutoff = 0.5  # Hz
        b, a = butter(2, cutoff / nyq, btype="low")
        baseline = filtfilt(b, a, ecg_signal)

        # Calculate baseline wander severity
        wander_amplitude = np.ptp(baseline)
        signal_amplitude = np.ptp(ecg_signal)
        severity = wander_amplitude / signal_amplitude if signal_amplitude > 0 else 1.0

        # Determine if baseline is stable
        stable = severity < 0.1

        return {
            "baseline_wander_severity": float(np.clip(severity, 0, 1)),
            "baseline_stable": bool(stable),
        }
    except Exception as e:
        logger.error(f"Error calculating baseline metrics: {str(e)}")
        return {"baseline_wander_severity": 1.0, "baseline_stable": False}


def _detect_powerline_interference(ecg_signal: np.ndarray, fs: float) -> Dict:
    """Detect powerline interference with improved sensitivity."""
    if not isinstance(ecg_signal, np.ndarray):
        raise ValueError("Signal must be a numpy array")
    if not np.isfinite(ecg_signal).all():
        raise ValueError("Signal contains non-finite values")
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    if len(ecg_signal) < fs:
        raise ValueError("Signal too short for interference detection")

    try:
        # Calculate power spectrum
        freqs, psd = signal.welch(ecg_signal, fs, nperseg=min(len(ecg_signal), fs * 2))

        # Calculate power in powerline frequency bands (50/60 Hz)
        powerline_freqs = [50, 60]
        band_width = 2  # Hz

        powerline_power = sum(
            _calculate_power_band(freqs, psd, f - band_width / 2, f + band_width / 2)
            for f in powerline_freqs
        )
        total_power = np.sum(psd)

        # Calculate interference ratio
        interference_ratio = powerline_power / total_power if total_power > 0 else 0.0

        return {
            "powerline_interference_ratio": float(np.clip(interference_ratio, 0, 1)),
            "powerline_interference_present": bool(interference_ratio > 0.1),
        }
    except Exception as e:
        logger.error(f"Error detecting powerline interference: {str(e)}")
        return {
            "powerline_interference_ratio": 1.0,
            "powerline_interference_present": True,
        }


def _normalize_metrics(metrics: Dict) -> Dict:
    """Ensure all metrics are within valid ranges."""
    if not isinstance(metrics, dict):
        raise ValueError("Metrics must be a dictionary")
    if not metrics:
        raise ValueError("Empty metrics dictionary")

    normalized = {}
    for key, value in metrics.items():
        if key.endswith("_stable") or key.endswith("_present"):
            if not isinstance(value, bool):
                raise ValueError(f"Invalid boolean value for metric {key}")
            normalized[key] = value
        else:
            if not isinstance(value, (int, float)):
                raise ValueError(f"Invalid value type for metric {key}")
            if not np.isfinite(value):
                raise ValueError(f"Non-finite value for metric {key}")

            if key == "SNR":
                normalized[key] = float(np.clip(value, 0, 40))
            else:
                normalized[key] = float(np.clip(value, 0, 1))

    return normalized


# --- Test Function ---
def test_calculate_signal_quality():
    fs = 500  # Sampling frequency in Hz
    duration = 10  # Duration in seconds
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)

    # Create a dummy original signal: noisy sine wave
    original = np.sin(2 * np.pi * 1.0 * t) + 0.2 * np.random.randn(len(t))
    # Create a dummy processed signal: a clean sine wave (ideal filtered version)
    processed = np.sin(2 * np.pi * 1.0 * t)

    # Compute the signal quality metrics
    quality_metrics = calculate_signal_quality(original, processed, fs)
    print("Signal Quality Metrics:")
    for key, value in quality_metrics.items():
        print(f"{key}: {value}")

    # Plot the signals for visual comparison
    plot_signal_comparison(original, processed, fs, title="Signal Quality Comparison")


if __name__ == "__main__":
    test_calculate_signal_quality()
