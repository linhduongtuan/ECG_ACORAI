"""
PyTorch‐centric HRV analysis utilities for ECG signal processing.

This module provides functions for time domain, frequency domain, and non-linear HRV metrics.
Where possible, Torch operations (e.g. torch.cumsum, torch.diff, and torch.logspace) replace
their NumPy counterparts. For some algorithms (e.g. Welch’s PSD or interpolation) existing SciPy
functions are retained while converting inputs from/to torch.
"""

import torch
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid  # Instead of np.trapz
from typing import Dict, Tuple, List, Union
from .config import ECGConfig
from .visualization import plot_signal_comparison, plot_comprehensive_analysis

# ---------------------------
# Helper: Validate RR intervals
# ---------------------------
def validate_rr_intervals(rr_intervals: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Validate and convert RR intervals to a torch tensor.

    Args:
        rr_intervals: Array or list of RR intervals (in milliseconds).

    Returns:
        A torch.Tensor of RR intervals.

    Raises:
        ValueError: if the input is empty, contains non-numerics, nonpositive values, etc.
    """
    if not isinstance(rr_intervals, torch.Tensor):
        rr_intervals = torch.tensor(rr_intervals, dtype=torch.float32)
    if rr_intervals.numel() == 0:
        raise ValueError("Empty RR interval array")
    if rr_intervals.numel() < 2:
        raise ValueError("At least 2 RR intervals are required")
    if not torch.isfinite(rr_intervals).all():
        raise ValueError("RR intervals contain NaN or infinite values")
    if torch.any(rr_intervals <= 0):
        raise ValueError("RR intervals must be positive")
    # Warn if outside physiological range (300–3000 ms)
    if torch.any(rr_intervals > 3000) or torch.any(rr_intervals < 300):
        warnings.warn("Some RR intervals are outside physiological range (300-3000 ms)")
    return rr_intervals

# ---------------------------
# DFA calculation using Torch where possible
# ---------------------------
def calculate_dfa(
    rr_intervals: Union[np.ndarray, torch.Tensor],
    scale_min: int = 4,
    scale_max: int = None,
    overlap: float = 0.5,
) -> Dict:
    """
    Calculate Detrended Fluctuation Analysis (DFA) using Torch operations
    where possible. For the linear trend fitting per window the segment is converted
    to NumPy and NumPy’s polyfit is used.

    Args:
        rr_intervals: Array of RR intervals.
        scale_min: Minimum box size.
        scale_max: Maximum box size (default: N//4).
        overlap: Overlap fraction between windows (0-1).

    Returns:
        Dictionary of DFA metrics.
    """
    rr_intervals = validate_rr_intervals(rr_intervals)
    N = rr_intervals.numel()

    if N < 2 * scale_min:
        raise ValueError(f"Signal length must be at least {2 * scale_min}")
    if scale_min < 4:
        raise ValueError("scale_min must be at least 4")
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be in [0, 1)")

    scale_max = scale_max or (N // 4)
    if scale_max < scale_min:
        scale_max = scale_min
    if scale_max > N // 2:
        scale_max = N // 2

    # Create integrated time series: y = cumsum(rr_intervals - mean)
    rr_mean = torch.mean(rr_intervals)
    y = torch.cumsum(rr_intervals - rr_mean, dim=0)

    # Generate scales (using Torch logspace, then floor and unique)
    num_scales = min(50, scale_max - scale_min + 1)
    scales = torch.logspace(np.log10(scale_min), np.log10(scale_max), steps=num_scales)
    scales = torch.floor(scales).to(dtype=torch.int32)
    scales = torch.unique(scales).tolist()  # Convert to list of scales (integers)

    fluctuations = []
    valid_scales = []

    # For operations that require polynomial fitting we use NumPy.
    y_np = y.detach().cpu().numpy()

    for scale in scales:
        # Calculate step (overlap)
        step = int(scale * (1 - overlap))
        if step < 1:
            step = 1
        n_windows = (N - scale) // step + 1
        if n_windows < 1:
            continue
        fluct = 0.0
        n_valid = 0
        for i in range(0, int(N - scale + 1), step):
            segment = y_np[i : i + scale]
            # Skip if almost constant (to avoid numerical issues)
            if np.ptp(segment) < np.finfo(float).eps:
                continue
            x = np.arange(scale)
            coef = np.polyfit(x, segment, 1)
            trend = np.polyval(coef, x)
            fluct += np.sum((segment - trend) ** 2)
            n_valid += 1
        if n_valid > 0:
            f_n = np.sqrt(fluct / (n_valid * scale))
            fluctuations.append(f_n)
            valid_scales.append(scale)

    if len(valid_scales) < 3:
        raise ValueError("Not enough valid scales for DFA calculation")

    valid_scales = np.array(valid_scales)
    fluctuations = np.array(fluctuations)

    log_scales = np.log10(valid_scales)
    log_fluct = np.log10(fluctuations)

    # Calculate short-term scaling (alpha1) over scales <= 16
    short_idx = valid_scales <= 16
    if np.sum(short_idx) >= 3:
        alpha1 = np.polyfit(log_scales[short_idx], log_fluct[short_idx], 1)[0]
    else:
        alpha1 = np.nan
    # Long-term scaling (alpha2) for scales > 16
    long_idx = valid_scales > 16
    if np.sum(long_idx) >= 3:
        alpha2 = np.polyfit(log_scales[long_idx], log_fluct[long_idx], 1)[0]
    else:
        alpha2 = np.nan

    alpha_overall = np.polyfit(log_scales, log_fluct, 1)[0]
    residuals = log_fluct - np.polyval(np.polyfit(log_scales, log_fluct, 1), log_scales)
    r_squared = 1 - np.var(residuals) / np.var(log_fluct)

    return {
        "alpha1": float(alpha1),                # Short-term scaling exponent
        "alpha2": float(alpha2),                # Long-term scaling exponent
        "alpha_overall": float(alpha_overall),  # Overall scaling exponent
        "r_squared": float(r_squared),          # Goodness-of-fit measure
        "n_scales": len(valid_scales),
        "scales": valid_scales.tolist(),
        "fluctuations": fluctuations.tolist(),
    }

# ---------------------------
# Time Domain HRV Metrics using Torch
# ---------------------------
def calculate_time_domain_hrv(rr_intervals: Union[np.ndarray, torch.Tensor]) -> Dict:
    """
    Calculate time domain HRV metrics using Torch operations.

    Args:
        rr_intervals: Array (or tensor) of RR intervals (in milliseconds).

    Returns:
        Dictionary of time domain HRV metrics.
    """
    rr_intervals = validate_rr_intervals(rr_intervals)
    if rr_intervals.numel() < 2:
        raise ValueError("At least 2 RR intervals required for analysis")
    mean_rr = torch.mean(rr_intervals)
    # Compute successive differences using slicing
    rr_diff = rr_intervals[1:] - rr_intervals[:-1]
    mean_hr = 60000 / mean_rr  # Convert mean RR to beats per minute (BPM)
    sdnn = torch.std(rr_intervals, unbiased=True).item()
    rmssd = torch.sqrt(torch.mean(rr_diff ** 2)).item()
    pnn50 = (torch.sum(torch.abs(rr_diff) > 50).float() / rr_intervals.numel() * 100).item()
    sdsd = torch.std(rr_diff, unbiased=True).item()

    return {
        "mean_hr": mean_hr.item() if isinstance(mean_hr, torch.Tensor) else mean_hr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "mean_rr": mean_rr.item(),
        "sdsd": sdsd,
    }

# ---------------------------
# Frequency Domain HRV Metrics (Mixed: using NumPy/SciPy)
# ---------------------------
def calculate_frequency_domain_hrv(
    rr_intervals: Union[np.ndarray, torch.Tensor], fs: float = 4.0
) -> Dict:
    """
    Calculate frequency domain HRV metrics.
    This function converts RR intervals to NumPy and uses SciPy routines.

    Args:
        rr_intervals: Array/tensor of RR intervals (ms).
        fs: Sampling frequency for interpolation (Hz).

    Returns:
        Dictionary of frequency domain HRV metrics.
    """
    if isinstance(rr_intervals, torch.Tensor):
        rr_intervals = rr_intervals.detach().cpu().numpy()
    if len(rr_intervals) < 4:
        raise ValueError("At least 4 RR intervals required for frequency analysis")

    rr_intervals = validate_rr_intervals(rr_intervals).detach().cpu().numpy()
    rr_x = np.cumsum(rr_intervals) / 1000.0  # Convert to seconds
    f_interp = interp1d(rr_x, rr_intervals, kind="cubic", bounds_error=False)
    t_interp = np.arange(rr_x[0], rr_x[-1], 1 / fs)
    rr_interp = f_interp(t_interp)
    if np.any(np.isnan(rr_interp)):
        raise ValueError("NaN values in interpolated signal")
    nperseg = min(256, len(rr_interp))
    frequencies, psd = signal.welch(rr_interp, fs=fs, nperseg=nperseg, detrend="constant", scaling="density")
    vlf_mask = (frequencies >= ECGConfig.VLF_LOW) & (frequencies < ECGConfig.VLF_HIGH)
    lf_mask = (frequencies >= ECGConfig.LF_LOW) & (frequencies < ECGConfig.LF_HIGH)
    hf_mask = (frequencies >= ECGConfig.HF_LOW) & (frequencies < ECGConfig.HF_HIGH)

    vlf_power = trapezoid(psd[vlf_mask], frequencies[vlf_mask]) if np.any(vlf_mask) else 0
    lf_power = trapezoid(psd[lf_mask], frequencies[lf_mask]) if np.any(lf_mask) else 0
    hf_power = trapezoid(psd[hf_mask], frequencies[hf_mask]) if np.any(hf_mask) else 0
    total_power = vlf_power + lf_power + hf_power
    lf_nu = 100 * lf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else 0
    hf_nu = 100 * hf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else 0

    return {
        "vlf_power": float(vlf_power),
        "lf_power": float(lf_power),
        "hf_power": float(hf_power),
        "lf_hf_ratio": float(lf_power / hf_power) if hf_power > 0 else 0.0,
        "total_power": float(total_power),
        "lf_nu": float(lf_nu),
        "hf_nu": float(hf_nu),
        "peak_vlf": float(frequencies[vlf_mask][np.argmax(psd[vlf_mask])]) if np.any(vlf_mask) else 0.0,
        "peak_lf": float(frequencies[lf_mask][np.argmax(psd[lf_mask])]) if np.any(lf_mask) else 0.0,
        "peak_hf": float(frequencies[hf_mask][np.argmax(psd[hf_mask])]) if np.any(hf_mask) else 0.0,
    }

# ---------------------------
# Poincaré Metrics using NumPy (input conversion as needed)
# ---------------------------
def calculate_poincare_metrics(
    rr_intervals: Union[np.ndarray, torch.Tensor]
) -> Dict:
    """
    Calculate Poincaré plot metrics (SD1, SD2, etc.).

    Args:
        rr_intervals: Array/tensor of RR intervals (ms).

    Returns:
        Dictionary of Poincaré metrics.
    """
    if isinstance(rr_intervals, torch.Tensor):
        rr_intervals = rr_intervals.detach().cpu().numpy()
    if len(rr_intervals) < 2:
        raise ValueError("At least 2 RR intervals required for Poincaré analysis")
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]
    diff_rr = rr_n1 - rr_n
    sd1 = np.sqrt(np.var(diff_rr, ddof=1) / 2) if len(diff_rr) > 0 else np.nan
    sd2 = np.sqrt(2 * np.var(rr_intervals, ddof=1) - np.var(diff_rr, ddof=1) / 2) if len(rr_intervals) > 1 else np.nan
    area = np.pi * sd1 * sd2
    correlation = np.corrcoef(rr_n, rr_n1)[0, 1] if len(rr_n) > 1 else np.nan

    return {
        "sd1": float(sd1),
        "sd2": float(sd2),
        "sd1_sd2_ratio": float(sd1 / sd2) if sd2 > 0 else np.nan,
        "ellipse_area": float(area),
        "energy_distribution_correlation": float(correlation),
    }

# ---------------------------
# Entropy Metrics (using NumPy)
# ---------------------------
def calculate_approximate_entropy(
    rr_intervals: np.ndarray, m: int = 2, r: float = 0.2
) -> float:
    """
    Calculate Approximate Entropy (ApEn).

    Args:
        rr_intervals: Array of RR intervals.
        m: Embedding dimension.
        r: Tolerance (typically 0.2*std).

    Returns:
        ApEn value.
    """
    N = len(rr_intervals)
    r_val = r * np.std(rr_intervals)
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    def _phi(m):
        X = [rr_intervals[i : i + m] for i in range(N - m + 1)]
        C = [len([1 for x_j in X if _maxdist(x_i, x_j) <= r_val]) / (N - m + 1.0)
             for x_i in X]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))
    return abs(_phi(m) - _phi(m + 1))


def calculate_sample_entropy(
    rr_intervals: np.ndarray, m: int = 2, r: float = 0.2
) -> float:
    """
    Calculate Sample Entropy (SampEn) with improved numerical stability.

    Args:
        rr_intervals: Array of RR intervals.
        m: Embedding dimension.
        r: Tolerance (typically 0.2*std).

    Returns:
        SampEn value.
    """
    if not isinstance(rr_intervals, np.ndarray):
        rr_intervals = np.array(rr_intervals)
    N = len(rr_intervals)
    if N < m + 2:
        raise ValueError(f"Data length must be at least {m+2} for m={m}")
    rr_normalized = (rr_intervals - np.mean(rr_intervals)) / np.std(rr_intervals)
    r_val = r * np.std(rr_intervals)
    def _count_matches(template_index, m):
        count = 0
        for i in range(N - m):
            if i != template_index:
                diff = np.abs(rr_normalized[i : i + m] - rr_normalized[template_index : template_index + m])
                if np.max(diff) <= r_val:
                    count += 1
        return count
    A = 0
    B = 0
    for i in range(N - m):
        A += _count_matches(i, m + 1)
        B += _count_matches(i, m)
    eps = np.finfo(float).eps
    A = (A + eps) / (N - m)
    B = (B + eps) / (N - m)
    sampen = -np.log(A / B)
    if not np.isfinite(sampen):
        raise ValueError("Failed to calculate sample entropy (result is not finite)")
    return sampen

# ---------------------------
# Non-linear HRV Metrics: Combining Poincaré, Entropy, and DFA
# ---------------------------
def calculate_non_linear_hrv(rr_intervals: Union[np.ndarray, torch.Tensor]) -> Dict:
    """
    Calculate non-linear HRV metrics:
      - Poincaré metrics (SD1, SD2, etc.)
      - Sample entropy, Approximate entropy
      - DFA (Detrended Fluctuation Analysis)

    Args:
        rr_intervals: Array/tensor of RR intervals (ms).

    Returns:
        Dictionary of non-linear HRV metrics.
    """
    if isinstance(rr_intervals, torch.Tensor):
        rr_intervals = rr_intervals.detach().cpu().numpy()
    if len(rr_intervals) < 4:
        raise ValueError("At least 4 RR intervals are required for non-linear analysis")
    rr_intervals = validate_rr_intervals(rr_intervals).detach().cpu().numpy()
    metrics = {}
    try:
        poincare_metrics = calculate_poincare_metrics(rr_intervals)
        metrics.update(poincare_metrics)
    except ValueError as e:
        raise ValueError(f"Failed to calculate Poincaré metrics: {str(e)}")
    try:
        metrics["sampen"] = calculate_sample_entropy(rr_intervals)
    except ValueError as e:
        raise ValueError(f"Failed to calculate sample entropy: {str(e)}")
    try:
        metrics["apen"] = calculate_approximate_entropy(rr_intervals)
    except ValueError as e:
        raise ValueError(f"Failed to calculate approximate entropy: {str(e)}")
    try:
        dfa_metrics = calculate_dfa(rr_intervals)
        metrics.update(dfa_metrics)
    except Exception as e:
        warnings.warn(f"Error calculating DFA metrics: {str(e)}. Setting DFA metrics to NaN.")
        metrics.update({
            "alpha1": np.nan,
            "alpha2": np.nan,
            "alpha_overall": np.nan,
            "r_squared": np.nan,
        })
    scalar_metrics = [v for v in metrics.values() if np.isscalar(v)]
    if not any(np.isfinite(scalar_metrics)):
        raise ValueError("No finite scalar metrics computed")
    return metrics

# ---------------------------
# Complete HRV Metrics: Time, Frequency, and Non-linear
# ---------------------------
def calculate_complete_hrv(
    rr_intervals: Union[np.ndarray, torch.Tensor], fs: float = 4.0
) -> Dict:
    """
    Calculate all HRV metrics (time domain, frequency domain, non-linear).

    Args:
        rr_intervals: Array/tensor of RR intervals in ms.
        fs: Sampling frequency for frequency domain interpolation.

    Returns:
        Dictionary containing all HRV metrics.
    """
    time_metrics = calculate_time_domain_hrv(rr_intervals)
    freq_metrics = calculate_frequency_domain_hrv(rr_intervals, fs)
    non_linear_metrics = calculate_non_linear_hrv(rr_intervals)
    return {
        "time_domain": time_metrics,
        "frequency_domain": freq_metrics,
        "non_linear": non_linear_metrics,
    }

# ---------------------------
# Advanced HRV Metrics (Time, Frequency, Non-linear)
# ---------------------------
def _calculate_sdann(rr_intervals: np.ndarray, window: int = 300000) -> float:
    """Calculate SDANN (standard deviation of 5-min interval means)."""
    try:
        cumsum = np.cumsum(rr_intervals)
        segments = []
        start = 0
        while start < cumsum[-1]:
            end = start + window
            mask = (cumsum >= start) & (cumsum < end)
            if np.any(mask):
                segments.append(np.mean(rr_intervals[mask]))
            start = end
        return np.std(segments) if segments else 0
    except Exception as e:
        warnings.warn(f"Error in SDANN calculation: {str(e)}")
        raise

def _calculate_sdnn_index(rr_intervals: np.ndarray, window: int = 300000) -> float:
    """Calculate SDNN index (mean of 5-min interval SDs)."""
    try:
        cumsum = np.cumsum(rr_intervals)
        segments = []
        start = 0
        while start < cumsum[-1]:
            end = start + window
            mask = (cumsum >= start) & (cumsum < end)
            if np.any(mask):
                segments.append(np.std(rr_intervals[mask]))
            start = end
        return np.mean(segments) if segments else 0
    except Exception as e:
        warnings.warn(f"Error in SDNN index calculation: {str(e)}")
        raise

def _calculate_time_domain_metrics(rr_intervals: np.ndarray) -> Dict:
    """Calculate time domain HRV metrics using NumPy."""
    try:
        diff_rr = np.diff(rr_intervals)
        nn50 = np.sum(np.abs(diff_rr) > 50)
        return {
            "SDNN": float(np.std(rr_intervals)),
            "RMSSD": float(np.sqrt(np.mean(diff_rr**2))),
            "pNN50": float(nn50 / len(diff_rr)) * 100,
            "SDANN": float(_calculate_sdann(rr_intervals)),
            "SDNN_index": float(_calculate_sdnn_index(rr_intervals)),
            "Mean_HR": float(60000 / np.mean(rr_intervals)),
            "STD_HR": float(np.std(60000 / rr_intervals)),
        }
    except Exception as e:
        warnings.warn(f"Error in time domain calculations: {str(e)}")
        raise

def _calculate_frequency_domain_metrics(
    rr_intervals: np.ndarray, fs: float
) -> Dict:
    """Calculate frequency domain HRV metrics using NumPy/SciPy."""
    try:
        time_points = np.cumsum(rr_intervals) / 1000.0
        f_interp = 4.0
        t_interp = np.arange(0, time_points[-1], 1 / f_interp)
        rr_interp = np.interp(t_interp, time_points, rr_intervals)
        nperseg = min(256, len(rr_interp))
        frequencies, psd = signal.welch(rr_interp, fs=f_interp, nperseg=nperseg,
                                        detrend="constant", scaling="density")
        vlf_mask = (frequencies >= 0.003) & (frequencies < 0.04)
        lf_mask = (frequencies >= 0.04) & (frequencies < 0.15)
        hf_mask = (frequencies >= 0.15) & (frequencies < 0.4)
        vlf_power = trapezoid(psd[vlf_mask], frequencies[vlf_mask]) if np.any(vlf_mask) else 0
        lf_power = trapezoid(psd[lf_mask], frequencies[lf_mask]) if np.any(lf_mask) else 0
        hf_power = trapezoid(psd[hf_mask], frequencies[hf_mask]) if np.any(hf_mask) else 0
        total_power = vlf_power + lf_power + hf_power
        return {
            "VLF_power": float(vlf_power),
            "LF_power": float(lf_power),
            "HF_power": float(hf_power),
            "LF_HF_ratio": float(lf_power / hf_power) if hf_power > 0 else 0.0,
            "Total_power": float(total_power),
            "LF_normalized": float(100 * lf_power / (lf_power + hf_power)),
            "HF_normalized": float(100 * hf_power / (lf_power + hf_power)),
        }
    except Exception as e:
        warnings.warn(f"Error in frequency domain calculations: {str(e)}")
        raise

def _calculate_nonlinear_metrics(rr_intervals: np.ndarray) -> Dict:
    """Calculate non-linear HRV metrics using NumPy."""
    try:
        diff_rr = np.diff(rr_intervals)
        sd1 = np.std(diff_rr) / np.sqrt(2) if len(diff_rr) > 0 else np.nan
        sd2 = np.sqrt(2 * np.var(rr_intervals) - np.var(diff_rr) / 2) if len(rr_intervals) > 1 else np.nan
        try:
            sampen = calculate_sample_entropy(rr_intervals)
        except Exception:
            sampen = np.nan
        try:
            apen = calculate_approximate_entropy(rr_intervals)
        except Exception:
            apen = np.nan
        try:
            if len(rr_intervals) >= 16:
                dfa_alpha1 = calculate_dfa(rr_intervals)["alpha1"]
                dfa_alpha2 = calculate_dfa(rr_intervals)["alpha2"]
            else:
                dfa_alpha1 = np.nan
                dfa_alpha2 = np.nan
        except Exception:
            dfa_alpha1 = np.nan
            dfa_alpha2 = np.nan

        return {
            "SD1": float(sd1),
            "SD2": float(sd2),
            "SD1_SD2_ratio": float(sd1 / sd2) if (sd2 > 0 and np.isfinite(sd1)) else np.nan,
            "SampEn": float(sampen),
            "ApEn": float(apen),
            "DFA_alpha1": float(dfa_alpha1),
            "DFA_alpha2": float(dfa_alpha2),
        }
    except Exception as e:
        warnings.warn(f"Error in non-linear calculations: {str(e)}")
        return _get_default_nonlinear_metrics()

def _get_default_time_metrics() -> Dict:
    return {
        "SDNN": np.nan,
        "RMSSD": np.nan,
        "pNN50": np.nan,
        "SDANN": np.nan,
        "SDNN_index": np.nan,
        "Mean_HR": np.nan,
        "STD_HR": np.nan,
    }

def _get_default_freq_metrics() -> Dict:
    return {
        "VLF_power": np.nan,
        "LF_power": np.nan,
        "HF_power": np.nan,
        "LF_HF_ratio": np.nan,
        "Total_power": np.nan,
        "LF_normalized": np.nan,
        "HF_normalized": np.nan,
    }

def _get_default_nonlinear_metrics() -> Dict:
    return {
        "SD1": np.nan,
        "SD2": np.nan,
        "SD1_SD2_ratio": np.nan,
        "SampEn": np.nan,
        "ApEn": np.nan,
        "DFA_alpha1": np.nan,
        "DFA_alpha2": np.nan,
    }

def calculate_advanced_hrv(
    rr_intervals: Union[np.ndarray, torch.Tensor], fs: float = 1000.0
) -> Dict:
    """
    Calculate comprehensive HRV metrics including time domain, frequency domain,
    and non-linear metrics.

    Args:
        rr_intervals: Array/tensor of RR intervals in ms.
        fs: Sampling frequency of the original ECG signal.

    Returns:
        Dictionary with advanced HRV metrics.
    """
    if isinstance(rr_intervals, torch.Tensor):
        rr_intervals = rr_intervals.detach().cpu().numpy()
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    rr_intervals = validate_rr_intervals(rr_intervals).detach().cpu().numpy()

    results = {}
    try:
        time_metrics = _calculate_time_domain_metrics(rr_intervals)
        results.update(time_metrics)
    except Exception as e:
        warnings.warn(f"Error in time domain calculations: {str(e)}")
        results.update(_get_default_time_metrics())
    try:
        freq_metrics = _calculate_frequency_domain_metrics(rr_intervals, fs)
        results.update(freq_metrics)
    except Exception as e:
        warnings.warn(f"Error in frequency domain calculations: {str(e)}")
        results.update(_get_default_freq_metrics())
    try:
        nonlinear_metrics = _calculate_nonlinear_metrics(rr_intervals)
        results.update(nonlinear_metrics)
    except Exception as e:
        warnings.warn(f"Error in non-linear calculations: {str(e)}")
        results.update(_get_default_nonlinear_metrics())
    return results

# ---------------------------
# Visualization: Poincaré Plot (remains largely unchanged)
# ---------------------------
def plot_poincare(
    rr_intervals: Union[np.ndarray, torch.Tensor], show: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate a Poincaré plot.

    Args:
        rr_intervals: Array/tensor of RR intervals (ms).
        show: Whether to display the plot immediately.

    Returns:
        Figure and Axes objects.
    """
    if isinstance(rr_intervals, torch.Tensor):
        rr_intervals = rr_intervals.detach().cpu().numpy()
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(rr_n, rr_n1, alpha=0.5, color="blue")
    ax.set_xlabel("RR_n (ms)")
    ax.set_ylabel("RR_n+1 (ms)")
    ax.set_title("Poincaré Plot")
    min_rr = min(rr_intervals)
    max_rr = max(rr_intervals)
    ax.plot([min_rr, max_rr], [min_rr, max_rr], "r--", alpha=0.5)
    if show:
        plt.show()
    return fig, ax

# ---------------------------
# Approximate and Sample Entropy (using NumPy)
# ---------------------------
# (Functions calculate_approximate_entropy and calculate_sample_entropy are defined above.)

# ---------------------------
# Main testing block
# ---------------------------
if __name__ == "__main__":
    # 1. Create dummy ECG signals.
    fs = 500  # Sampling rate in Hz
    duration = 10  # seconds
    t = np.arange(0, duration, 1 / fs)
    original_signal = np.sin(2 * np.pi * 1 * t) + 0.2 * np.random.randn(len(t))
    processed_signal = np.sin(2 * np.pi * 1 * t)

    # 2. Generate dummy QRS peaks and RR intervals.
    dummy_rr_intervals = 800 + 50 * np.random.randn(10)  # in ms
    dummy_peaks = [0]
    for rr in dummy_rr_intervals:
        dummy_peaks.append(dummy_peaks[-1] + int(rr / 1000 * fs))
    dummy_peaks = np.array(dummy_peaks)

    # 3. Create dummy frequency & PSD data for comprehensive analysis plot.
    dummy_frequencies = np.linspace(0.1, 30, 100)
    dummy_psd = np.abs(np.sin(dummy_frequencies))
    dummy_results = {
        "original_signal": original_signal,
        "processed_signal": processed_signal,
        "peaks": dummy_peaks,
        "hrv_metrics": {"rr_intervals": dummy_rr_intervals},
        "frequencies": dummy_frequencies,
        "psd": dummy_psd,
    }

    # 4. Plot the signal comparison.
    plot_signal_comparison(
        original=original_signal,
        processed=processed_signal,
        fs=fs,
        title="Dummy ECG Signal Comparison",
    )

    # 5. Plot the comprehensive analysis.
    plot_comprehensive_analysis(dummy_results)

    # 6. Compute and print HRV metrics using the dummy RR intervals.
    time_hrv = calculate_time_domain_hrv(dummy_rr_intervals)
    freq_hrv = calculate_frequency_domain_hrv(dummy_rr_intervals)
    complete_hrv = calculate_complete_hrv(dummy_rr_intervals)

    print("Time Domain HRV Metrics:")
    print(time_hrv)
    print("\nFrequency Domain HRV Metrics:")
    print(freq_hrv)
    print("\nComplete HRV Metrics:")
    print(complete_hrv)

    # 7. Plot the Poincaré plot.
    fig, ax = plot_poincare(dummy_rr_intervals)
    ax.set_title("Dummy Poincaré Plot")
    plt.show()