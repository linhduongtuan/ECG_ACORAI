"""
PyTorch‐centric signal quality and ECG quality assessment utilities.

This module provides functions to calculate comprehensive quality metrics for ECG
signals. Operations run using PyTorch when possible, while relying on NumPy/SciPy for
detrending, filtering, and power spectrum estimation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from typing import Dict, Tuple, Optional, List, Union
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper Function: _calculate_power_band (remains mostly unchanged)
# ---------------------------------------------------------------------------
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
        band_idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        band_power = np.trapz(psd[band_idx], freqs[band_idx])
        return float(max(band_power, 0.0))
    except Exception as e:
        logger.error(f"Error calculating power band: {str(e)}")
        return 0.0

# ---------------------------------------------------------------------------
# Main Function: calculate_signal_quality
# ---------------------------------------------------------------------------
def calculate_signal_quality(
    original: Union[np.ndarray, torch.Tensor],
    processed: Union[np.ndarray, torch.Tensor],
    fs: float
) -> Dict:
    """
    Calculate comprehensive signal quality metrics for ECG signals.
    Uses PyTorch for many calculations and falls back to NumPy/SciPy where needed.

    Parameters
    ----------
    original : np.ndarray or torch.Tensor
        Original ECG signal.
    processed : np.ndarray or torch.Tensor
        Processed (filtered/denoised) ECG signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    Dict
        Dictionary containing quality metrics including SNR, power, time‐domain statistics,
        noise characteristics, baseline wander, frequency metrics, and signal complexity.
    """
    try:
        # Input validation
        if not (isinstance(original, (np.ndarray, torch.Tensor)) and isinstance(processed, (np.ndarray, torch.Tensor))):
            raise ValueError("Signals must be numpy arrays or torch tensors")
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("Sampling frequency must be positive")

        # Ensure both signals are Torch tensors (using float64 for precision)
        if not torch.is_tensor(original):
            original = torch.tensor(original, dtype=torch.float64)
        if not torch.is_tensor(processed):
            processed = torch.tensor(processed, dtype=torch.float64)

        if original.shape != processed.shape:
            raise ValueError(f"Signal shape mismatch: {original.shape} vs {processed.shape}")
        if not torch.isfinite(original).all() or not torch.isfinite(processed).all():
            raise ValueError("Signals contain invalid values (inf or nan)")
        if original.numel() < fs:
            raise ValueError(f"Signal too short. Must be at least {fs} samples")

        # Convert to float64 for numerical precision (already in float64)

        # 1. Signal-to-Noise Ratio and Power Metrics (use Torch)
        noise = original - processed
        signal_rms = torch.sqrt(torch.mean(processed ** 2))
        noise_rms = torch.sqrt(torch.mean(noise ** 2))
        epsilon = torch.finfo(torch.float64).eps
        snr = 20 * torch.log10(signal_rms / (noise_rms + epsilon)) if noise_rms > epsilon else torch.tensor(100.0)
        power_metrics = {
            "snr": float(snr.item()),
            "signal_rms": float(signal_rms.item()),
            "noise_rms": float(noise_rms.item()),
            "signal_power": float((signal_rms ** 2).item()),
            "noise_power": float((noise_rms ** 2).item()),
        }

        # 2. Time Domain Statistical Metrics (compute using Torch)
        proc_mean = torch.mean(processed)
        proc_std = torch.std(processed, unbiased=False)
        skewness = torch.mean((processed - proc_mean) ** 3) / (proc_std ** 3 + epsilon)
        kurtosis = torch.mean((processed - proc_mean) ** 4) / (proc_std ** 4 + epsilon) - 3.0
        ptp = (torch.max(processed) - torch.min(processed)).item()
        # Zero crossings: count sign changes using a boolean conversion.
        zeros = torch.sum(torch.abs(torch.diff((processed < 0).type(torch.int))))
        zero_crossings = int(zeros.item())
        crest_factor = (torch.max(torch.abs(processed)) / (signal_rms + epsilon)).item()

        statistical_metrics = {
            "mean": float(proc_mean.item()),
            "std": float(proc_std.item()),
            "kurtosis": float(kurtosis.item()),
            "skewness": float(skewness.item()),
            "peak_to_peak": ptp,
            "zero_crossings": zero_crossings,
            "crest_factor": crest_factor,
        }

        # 3. Advanced Noise Assessment (iterate using Torch; use Hilbert from SciPy)
        window_size = max(int(0.1 * fs), 10)
        stride = max(window_size // 4, 1)
        noise_levels = []
        diff_levels = []
        envelope_var = []
        for i in range(0, original.numel() - window_size, stride):
            window = processed[i : i + window_size]
            noise_levels.append(torch.std(window).item())
            diff_levels.append(torch.std(window[1:] - window[:-1]).item())
            try:
                # Convert window to numpy for Hilbert transform
                window_np = window.detach().cpu().numpy()
                envelope = np.abs(signal.hilbert(window_np))
                envelope_var.append(np.std(envelope))
            except Exception:
                envelope_var.append(np.nan)
        noise_levels_np = np.array(noise_levels)
        diff_levels_np = np.array(diff_levels)
        envelope_var_np = np.array(envelope_var)

        noise_metrics = {
            "noise_std": float(np.std(noise_levels_np)),
            "noise_max": float(np.max(noise_levels_np)),
            "noise_mean": float(np.mean(noise_levels_np)),
            "diff_std": float(np.std(diff_levels_np)),
            "envelope_std": float(np.nanstd(envelope_var_np)),
            "noise_percentile_95": float(np.percentile(noise_levels_np, 95)),
        }

        # 4. Enhanced Baseline Wander Assessment (convert to NumPy)
        processed_np = processed.detach().cpu().numpy()
        linear_detrend = signal.detrend(processed_np, type="linear")
        constant_detrend = signal.detrend(processed_np, type="constant")
        window_length = min(int(fs), len(processed_np))
        if window_length % 2 == 0:
            window_length -= 1
        baseline = signal.savgol_filter(processed_np, window_length, 2)
        baseline_metrics = {
            "baseline_drift_linear": float(np.std(linear_detrend)),
            "baseline_drift_constant": float(np.std(constant_detrend)),
            "baseline_power": float(np.sum(baseline ** 2)),
            "baseline_max_dev": float(np.max(np.abs(baseline))),
            "baseline_crossing_rate": float(np.mean(np.abs(np.diff(baseline > 0)))),
        }

        # 5. Enhanced Frequency Domain Analysis (convert to NumPy)
        nperseg = min(len(processed_np), int(4 * fs))
        freqs, psd = signal.welch(processed_np, fs, nperseg=nperseg)
        total_power = _calculate_power_band(freqs, psd, 0, fs / 2)
        epsilon_np = np.finfo(float).eps
        freq_metrics = {
            "vlf_power": float(_calculate_power_band(freqs, psd, 0, 0.04) / total_power),
            "lf_power": float(_calculate_power_band(freqs, psd, 0.04, 0.15) / total_power),
            "hf_power": float(_calculate_power_band(freqs, psd, 0.15, 0.4) / total_power),
            "line_noise": float(_calculate_power_band(freqs, psd, 45, 55) / total_power),
            "high_freq_noise": float(_calculate_power_band(freqs, psd, 100, fs / 2) / total_power),
        }
        psd_norm = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + epsilon_np))
        freq_metrics["spectral_entropy"] = float(spectral_entropy)

        # 6. Signal Complexity Measures (use NumPy on processed_np)
        n_segments = min(5, len(processed_np) // int(fs))
        if n_segments > 0:
            segment_length = len(processed_np) // n_segments
            entropy_values = []
            for i in range(n_segments):
                start = i * segment_length
                end = start + segment_length
                segment = processed_np[start:end]
                try:
                    diff_matrix = np.abs(segment[:, None] - segment)
                    cmm = np.sum(diff_matrix <= epsilon_np, axis=1)
                    entropy = np.mean(np.log(cmm + epsilon_np))
                    entropy_values.append(entropy)
                except Exception:
                    continue
            complexity_metrics = {
                "entropy_mean": float(np.mean(entropy_values)) if entropy_values else np.nan,
                "entropy_std": float(np.std(entropy_values)) if len(entropy_values) > 1 else np.nan,
            }
        else:
            complexity_metrics = {"entropy_mean": np.nan, "entropy_std": np.nan}

        # Combine all metrics and include a normalized quality score.
        # Here we normalize the quality score as snr/20 clipped to [0,1]
        quality_score = float(np.clip(float(snr.item() if torch.is_tensor(snr) else snr) / 20.0, 0, 1))
        metrics = {
            **power_metrics,
            **statistical_metrics,
            **noise_metrics,
            **baseline_metrics,
            **freq_metrics,
            **complexity_metrics,
            "quality_score": quality_score,
        }
        return metrics

    except Exception as e:
        logger.error(f"Error calculating signal quality: {str(e)}")
        raise

def assess_signal_quality(signal: np.ndarray, fs: float) -> dict:
    """
    Assess the overall quality of an ECG signal.

    This is a placeholder implementation.
    You can replace or extend it with your specific quality metrics.
    """
    # For instance, use a simple SNR-based metric, baseline wander, etc.
    # Here we just calculate the SNR of the signal compared to its mean.
    signal_mean = np.mean(signal)
    try:
        snr = 10 * np.log10(np.var(signal) / np.var(signal - signal_mean))
    except ZeroDivisionError:
        snr = float('nan')

    # Define an overall quality score (this is just an example)
    overall_quality = 1.0 if snr > 10 else 0.5

    # You might also include flags for power line interference or baseline wander:
    quality = {
        "overall_quality": overall_quality,
        "SNR": snr,
        "powerline_interference_present": False,  # Placeholder
        "baseline_wander_severity": 0.0  # Placeholder
    }
    return quality
    
# ---------------------------------------------------------------------------
# Plotting Function (largely unchanged)
# ---------------------------------------------------------------------------
def plot_signal_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    fs: float,
    title: str = "ECG Signal Comparison",
    show_metrics: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot original and processed ECG signals along with noise.

    Parameters
    ----------
    original : np.ndarray
        Original ECG signal.
    processed : np.ndarray
        Processed (filtered/denoised) ECG signal.
    fs : float
        Sampling frequency (Hz).
    title : str, optional
        Plot title.
    show_metrics : bool, optional
        Whether to display calculated quality metrics.

    Returns
    -------
    Optional[plt.Figure]
        The matplotlib Figure (if show_metrics is False), otherwise None.
    """
    try:
        if len(original) != len(processed):
            raise ValueError("Signals must have the same length")
        time = np.arange(len(original)) / fs
        fig = plt.figure(figsize=(15, 8))
        ax1 = plt.subplot(3, 1, 1)
        plt.plot(time, original, color="tab:blue", label="Original")
        plt.title("Original Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
        _ = plt.subplot(3, 1, 2, sharex=ax1)
        plt.plot(time, processed, color="tab:orange", label="Processed")
        plt.title("Processed Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
        _ = plt.subplot(3, 1, 3, sharex=ax1)
        noise = original - processed
        plt.plot(time, noise, color="tab:red", label="Noise")
        plt.title("Extracted Noise")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
        if show_metrics:
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

# ---------------------------------------------------------------------------
# (Other helper functions remain unchanged, for example: _detect_powerline_interference, _assess_rpeak_quality, etc.)
# You may choose to leave them as NumPy/SciPy implementations.
# ---------------------------------------------------------------------------

# --- Test Function ---
def test_calculate_signal_quality():
    fs = 500  # Sampling frequency in Hz
    duration = 10  # seconds
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    # Dummy original signal: noisy sine wave.
    original = np.sin(2 * np.pi * 1.0 * t) + 0.2 * np.random.randn(len(t))
    # Dummy processed signal: clean sine wave.
    processed = np.sin(2 * np.pi * 1.0 * t)
    quality_metrics = calculate_signal_quality(original, processed, fs)
    print("Signal Quality Metrics:")
    for key, value in quality_metrics.items():
        print(f"{key}: {value}")
    plot_signal_comparison(original, processed, fs, title="Signal Quality Comparison")

if __name__ == "__main__":
    test_calculate_signal_quality()