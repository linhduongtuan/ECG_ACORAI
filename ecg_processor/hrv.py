import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid  # Instead of using np.trapz
from typing import Dict, Tuple, List
from .config import ECGConfig
from .visualization import plot_signal_comparison, plot_comprehensive_analysis


def calculate_dfa(
    rr_intervals: np.ndarray,
    scale_min: int = 4,
    scale_max: int = None,
    overlap: float = 0.5,
) -> Dict:
    """
    Calculate Detrended Fluctuation Analysis (DFA) with improved numerical stability
    and additional metrics.

    Args:
        rr_intervals: Array of RR intervals
        scale_min: Minimum box size
        scale_max: Maximum box size (default: N//4)
        overlap: Overlap fraction between windows (0-1)
    Returns:
        Dictionary containing DFA metrics
    Raises:
        ValueError: If input parameters are invalid
    """
    rr_intervals = validate_rr_intervals(rr_intervals)
    N = len(rr_intervals)

    # Input validation
    if len(rr_intervals) == 0:
        raise ValueError("Empty RR interval array")
    if scale_min < 4:
        raise ValueError("scale_min must be at least 4")
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be in [0, 1)")
    if len(rr_intervals) < 2 * scale_min:
        raise ValueError(f"Signal length must be at least {2 * scale_min}")

    # Set and validate scale_max
    scale_max = scale_max or (N // 4)
    if scale_max < scale_min:
        scale_max = scale_min
    if scale_max > N // 2:
        scale_max = N // 2

    # Create integrated time series
    rr_mean = np.mean(rr_intervals)
    y = np.cumsum(rr_intervals - rr_mean)

    # Generate scales using logarithmic spacing
    scales = np.unique(
        np.floor(
            np.logspace(
                np.log10(scale_min),
                np.log10(scale_max),
                num=min(50, scale_max - scale_min + 1),  # Limit number of scales
            )
        ).astype(int)
    )

    fluctuations = []
    valid_scales = []

    for scale in scales:
        # Calculate step size with overlap
        step = int(scale * (1 - overlap))
        if step < 1:
            step = 1

        # Calculate number of windows
        n_windows = (N - scale) // step + 1
        if n_windows < 1:
            continue

        # Initialize fluctuation for this scale
        fluct = 0.0
        n_valid = 0

        try:
            for i in range(0, N - scale + 1, step):
                segment = y[i : i + scale]

                # Skip if segment has too little variation
                if np.ptp(segment) < np.finfo(float).eps:
                    continue

                # Fit linear trend
                x = np.arange(scale)
                coef = np.polyfit(x, segment, 1)
                trend = np.polyval(coef, x)

                # Calculate RMS
                fluct += np.sum((segment - trend) ** 2)
                n_valid += 1

            if n_valid > 0:
                # Calculate fluctuation for this scale
                f_n = np.sqrt(fluct / (n_valid * scale))
                fluctuations.append(f_n)
                valid_scales.append(scale)

        except Exception as e:
            raise ValueError(f"Error processing scale {scale}: {str(e)}")

    # Need at least 3 valid scales for reliable fit
    if len(valid_scales) < 3:
        raise ValueError("Not enough valid scales for DFA calculation")

    # Convert to arrays
    valid_scales = np.array(valid_scales)
    fluctuations = np.array(fluctuations)

    # Calculate scaling exponents for different ranges
    log_scales = np.log10(valid_scales)
    log_fluct = np.log10(fluctuations)

    # Short-term scaling (alpha1)
    short_idx = valid_scales <= 16
    if np.sum(short_idx) >= 3:
        alpha1 = np.polyfit(log_scales[short_idx], log_fluct[short_idx], 1)[0]
    else:
        alpha1 = np.nan

    # Long-term scaling (alpha2)
    long_idx = valid_scales > 16
    if np.sum(long_idx) >= 3:
        alpha2 = np.polyfit(log_scales[long_idx], log_fluct[long_idx], 1)[0]
    else:
        alpha2 = np.nan

    # Overall scaling
    alpha_overall = np.polyfit(log_scales, log_fluct, 1)[0]

    # Calculate goodness of fit
    residuals = log_fluct - np.polyval(np.polyfit(log_scales, log_fluct, 1), log_scales)
    r_squared = 1 - np.var(residuals) / np.var(log_fluct)

    return {
        "alpha1": float(alpha1),  # Short-term scaling
        "alpha2": float(alpha2),  # Long-term scaling
        "alpha_overall": float(alpha_overall),  # Overall scaling
        "r_squared": float(r_squared),  # Goodness of fit
        "n_scales": len(valid_scales),  # Number of valid scales
        "scales": valid_scales.tolist(),  # Valid scales used
        "fluctuations": fluctuations.tolist(),  # Fluctuations for each scale
    }


def validate_rr_intervals(rr_intervals: np.ndarray) -> np.ndarray:
    """Validate and clean RR interval data.

    Args:
        rr_intervals: Array of RR intervals in milliseconds
    Returns:
        Cleaned RR intervals array
    Raises:
        ValueError: If RR intervals are invalid
    """
    # Check if input is array-like
    if not isinstance(rr_intervals, (list, np.ndarray)):
        raise ValueError("RR intervals must be a numpy array or list")

    # Convert to numpy array if needed
    if not isinstance(rr_intervals, np.ndarray):
        rr_intervals = np.array(rr_intervals)

    # Check for empty array
    if len(rr_intervals) == 0:
        raise ValueError("Empty RR interval array")

    # Check for minimum length
    if len(rr_intervals) < 2:
        raise ValueError("At least 2 RR intervals are required")

    # Check for numeric data
    if not np.issubdtype(rr_intervals.dtype, np.number):
        raise ValueError("RR intervals must be numeric")

    # Check for negative values
    if np.any(rr_intervals <= 0):
        raise ValueError("RR intervals must be positive")

    # Check for NaN or infinite values
    if np.any(np.isnan(rr_intervals)) or np.any(np.isinf(rr_intervals)):
        raise ValueError("RR intervals contain NaN or infinite values")

    # Check for physiological range (300-3000 ms)
    if np.any(rr_intervals > 3000) or np.any(rr_intervals < 300):
        warnings.warn("Some RR intervals are outside physiological range (300-3000 ms)")

    return rr_intervals


def calculate_time_domain_hrv(rr_intervals: np.ndarray) -> Dict:
    """Calculate time domain HRV metrics.

    Args:
        rr_intervals: Array of RR intervals in milliseconds
    Returns:
        Dictionary containing time domain HRV metrics
    Raises:
        ValueError: If input parameters are invalid or calculation fails
    """
    # Input validation
    if not isinstance(rr_intervals, np.ndarray):
        raise ValueError("RR intervals must be a numpy array")
    if len(rr_intervals) == 0:
        raise ValueError("Empty RR interval array")
    if len(rr_intervals) < 2:  # Need at least 2 intervals for diff calculations
        raise ValueError("At least 2 RR intervals required for time domain analysis")

    # Validate RR intervals
    rr_intervals = validate_rr_intervals(rr_intervals)

    try:
        # Calculate basic statistics
        mean_rr = np.mean(rr_intervals)
        if mean_rr <= 0:
            raise ValueError("Invalid mean RR interval (must be positive)")

        rr_diff = np.diff(rr_intervals)

        # Calculate metrics
        mean_hr = 60000 / mean_rr  # Convert to BPM
        sdnn = float(np.std(rr_intervals, ddof=1))  # Standard deviation
        rmssd = float(
            np.sqrt(np.mean(rr_diff**2))
        )  # Root mean square of successive differences
        pnn50 = float(
            np.sum(np.abs(rr_diff) > 50) / len(rr_intervals) * 100
        )  # Percentage of intervals > 50ms
        sdsd = float(
            np.std(rr_diff, ddof=1)
        )  # Standard deviation of successive differences

        # Validate calculations
        metrics = {
            "mean_hr": mean_hr,
            "sdnn": sdnn,
            "rmssd": rmssd,
            "pnn50": pnn50,
            "mean_rr": float(mean_rr),
            "sdsd": sdsd,
        }

        # Check for invalid results
        if not all(np.isfinite(list(metrics.values()))):
            raise ValueError(
                "Failed to calculate valid time domain metrics (non-finite values)"
            )

        return metrics

    except Exception as e:
        raise ValueError(f"Error calculating time domain metrics: {str(e)}")


def calculate_frequency_domain_hrv(rr_intervals: np.ndarray, fs: float = 4.0) -> Dict:
    """Calculate frequency domain HRV metrics.

    Args:
        rr_intervals: Array of RR intervals in milliseconds
        fs: Sampling frequency for interpolation (Hz)
    Returns:
        Dictionary containing frequency domain HRV metrics
    Raises:
        ValueError: If input parameters are invalid or calculation fails
    """
    # Input validation
    if not isinstance(rr_intervals, np.ndarray):
        raise ValueError("RR intervals must be a numpy array")
    if len(rr_intervals) == 0:
        raise ValueError("Empty RR interval array")
    if not isinstance(fs, (int, float)):
        raise ValueError("Sampling frequency must be a number")
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    if len(rr_intervals) < 4:
        raise ValueError("At least 4 RR intervals required for frequency analysis")

    # Validate RR intervals
    rr_intervals = validate_rr_intervals(rr_intervals)

    # Convert to seconds and get cumulative time
    rr_x = np.cumsum(rr_intervals) / 1000.0

    # Interpolate
    try:
        f = interp1d(rr_x, rr_intervals, kind="cubic", bounds_error=False)

        # Regular sampling
        t_interp = np.arange(rr_x[0], rr_x[-1], 1 / fs)
        rr_interp = f(t_interp)

        # Check for NaN values after interpolation
        if np.any(np.isnan(rr_interp)):
            raise ValueError("NaN values in interpolated signal")

        # Calculate PSD with optimal parameters
        nperseg = min(256, len(rr_interp))  # Adjust segment length for short signals
        frequencies, psd = signal.welch(
            rr_interp, fs=fs, nperseg=nperseg, detrend="constant", scaling="density"
        )

        # Define frequency bands
        vlf_mask = (frequencies >= ECGConfig.VLF_LOW) & (
            frequencies < ECGConfig.VLF_HIGH
        )
        lf_mask = (frequencies >= ECGConfig.LF_LOW) & (frequencies < ECGConfig.LF_HIGH)
        hf_mask = (frequencies >= ECGConfig.HF_LOW) & (frequencies < ECGConfig.HF_HIGH)

        # Calculate powers
        vlf_power = (
            trapezoid(psd[vlf_mask], frequencies[vlf_mask]) if np.any(vlf_mask) else 0
        )
        lf_power = (
            trapezoid(psd[lf_mask], frequencies[lf_mask]) if np.any(lf_mask) else 0
        )
        hf_power = (
            trapezoid(psd[hf_mask], frequencies[hf_mask]) if np.any(hf_mask) else 0
        )
        total_power = vlf_power + lf_power + hf_power

        # Validate power calculations
        if not all(np.isfinite([vlf_power, lf_power, hf_power, total_power])):
            raise ValueError(
                "Failed to calculate valid frequency domain powers (non-finite values)"
            )

        # Calculate normalized powers
        lf_nu = (
            100 * lf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else 0
        )
        hf_nu = (
            100 * hf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else 0
        )

        metrics = {
            "vlf_power": float(vlf_power),
            "lf_power": float(lf_power),
            "hf_power": float(hf_power),
            "lf_hf_ratio": float(lf_power / hf_power) if hf_power > 0 else 0.0,
            "total_power": float(total_power),
            "lf_nu": float(lf_nu),  # Normalized units
            "hf_nu": float(hf_nu),  # Normalized units
            "peak_vlf": float(frequencies[vlf_mask][np.argmax(psd[vlf_mask])])
            if np.any(vlf_mask)
            else 0.0,
            "peak_lf": float(frequencies[lf_mask][np.argmax(psd[lf_mask])])
            if np.any(lf_mask)
            else 0.0,
            "peak_hf": float(frequencies[hf_mask][np.argmax(psd[hf_mask])])
            if np.any(hf_mask)
            else 0.0,
        }

        # Validate all metrics
        if not all(np.isfinite(list(metrics.values()))):
            raise ValueError(
                "Failed to calculate valid frequency domain metrics (non-finite values)"
            )

        return metrics

    except Exception as e:
        raise ValueError(f"Error in frequency analysis: {str(e)}")


def calculate_poincare_metrics(rr_intervals: np.ndarray) -> Dict:
    """
    Calculate Poincaré plot metrics (SD1, SD2) with improved error handling
    and additional metrics.

    Args:
        rr_intervals: Array of RR intervals in milliseconds
    Returns:
        Dictionary containing Poincaré plot metrics
    Raises:
        ValueError: If input parameters are invalid or calculation fails
    """
    # Input validation
    if not isinstance(rr_intervals, np.ndarray):
        raise ValueError("RR intervals must be a numpy array")
    if len(rr_intervals) == 0:
        raise ValueError("Empty RR interval array")
    if len(rr_intervals) < 2:
        raise ValueError("At least 2 RR intervals required for Poincaré analysis")

    # Validate RR intervals
    rr_intervals = validate_rr_intervals(rr_intervals)

    # Create Poincaré plot coordinates
    rr_n = rr_intervals[:-1]  # Current RR interval
    rr_n1 = rr_intervals[1:]  # Next RR interval

    # Calculate differences for SD1 and SD2
    diff_rr = rr_n1 - rr_n

    # Calculate standard descriptors
    sd1 = float(np.sqrt(np.var(diff_rr, ddof=1) / 2))  # Short-term variability
    sd2 = float(
        np.sqrt(2 * np.var(rr_intervals, ddof=1) - np.var(diff_rr, ddof=1) / 2)
    )  # Long-term variability

    # Validate SD1 and SD2
    if not np.isfinite(sd1) or not np.isfinite(sd2):
        raise ValueError("Failed to calculate SD1 or SD2 (non-finite values)")
    if sd1 < 0 or sd2 < 0:
        raise ValueError("Invalid negative values for SD1 or SD2")

    # Calculate area of the ellipse
    area = float(np.pi * sd1 * sd2)

    # Calculate additional metrics
    metrics = {
        "sd1": sd1,  # Short-term variability
        "sd2": sd2,  # Long-term variability
        "sd1_sd2_ratio": float(sd1 / sd2)
        if sd2 > 0
        else 0.0,  # Ratio of short to long term variation
        "ellipse_area": area,  # Area of the fitted ellipse
        "csi": float(sd2 / sd1) if sd1 > 0 else 0.0,  # Cardiac Sympathetic Index
        "cvi": float(np.log10(sd1 * sd2))
        if sd1 > 0 and sd2 > 0
        else 0.0,  # Cardiac Vagal Index
        "mean_distance": float(
            np.mean(np.sqrt(diff_rr**2))
        ),  # Mean distance from diagonal
        "max_distance": float(
            np.max(np.sqrt(diff_rr**2))
        ),  # Maximum distance from diagonal
    }

    # Validate all metrics
    if not all(np.isfinite(list(metrics.values()))):
        raise ValueError(
            "Failed to calculate valid Poincaré metrics (non-finite values)"
        )

    return metrics


def plot_poincare(
    rr_intervals: np.ndarray, show: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate Poincaré plot

    Args:
        rr_intervals: Array of RR intervals in milliseconds
        show: Whether to display the plot immediately
    Returns:
        Figure and Axes objects
    """
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(rr_n, rr_n1, alpha=0.5, color="blue")
    ax.set_xlabel("RR_n (ms)")
    ax.set_ylabel("RR_n+1 (ms)")
    ax.set_title("Poincaré Plot")

    # Add identity line
    min_rr = min(rr_intervals)
    max_rr = max(rr_intervals)
    ax.plot([min_rr, max_rr], [min_rr, max_rr], "r--", alpha=0.5)

    if show:
        plt.show()

    return fig, ax


def calculate_approximate_entropy(
    rr_intervals: np.ndarray, m: int = 2, r: float = 0.2
) -> float:
    """
    Calculate Approximate Entropy (ApEn)

    Args:
        rr_intervals: Array of RR intervals
        m: Embedding dimension
        r: Tolerance (typically 0.2 * std of the data)
    Returns:
        ApEn value
    """
    N = len(rr_intervals)
    r = r * np.std(rr_intervals)

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[rr_intervals[j] for j in range(i, i + m)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    return abs(_phi(m) - _phi(m + 1))


def calculate_sample_entropy(
    rr_intervals: np.ndarray, m: int = 2, r: float = 0.2
) -> float:
    """
    Calculate Sample Entropy (SampEn) with improved numerical stability and error handling.

    Args:
        rr_intervals: Array of RR intervals
        m: Embedding dimension
        r: Tolerance (typically 0.2 * std of the data)
    Returns:
        SampEn value
    Raises:
        ValueError: If input parameters are invalid or calculation fails
    """
    # Input validation
    if not isinstance(rr_intervals, np.ndarray):
        raise ValueError("RR intervals must be a numpy array")
    if len(rr_intervals) == 0:
        raise ValueError("Empty RR interval array")
    if m < 1:
        raise ValueError("Embedding dimension must be at least 1")
    if r <= 0:
        raise ValueError("Tolerance must be positive")
    if len(rr_intervals) < m + 2:
        raise ValueError(f"Data length must be at least {m + 2} for m={m}")

    # Validate RR intervals
    rr_intervals = validate_rr_intervals(rr_intervals)

    # Normalize data and calculate tolerance
    rr_normalized = (rr_intervals - np.mean(rr_intervals)) / np.std(rr_intervals)
    r = r * np.std(rr_intervals)

    def _count_matches(template, m):
        count = 0
        for i in range(len(rr_intervals) - m):
            if i != template[0]:
                diff = np.abs(
                    rr_normalized[i : i + m]
                    - rr_normalized[template[0] : template[0] + m]
                )
                if np.max(diff) <= r:
                    count += 1
        return count

    # Calculate template matches for m and m+1
    A = 0  # Count for m+1
    B = 0  # Count for m

    for i in range(len(rr_intervals) - m):
        template = [i]
        A += _count_matches(template, m + 1)
        B += _count_matches(template, m)

    # Add small constant for numerical stability
    eps = np.finfo(float).eps
    A = (A + eps) / (len(rr_intervals) - m)
    B = (B + eps) / (len(rr_intervals) - m)

    sampen = -np.log(A / B)

    # Validate result
    if not np.isfinite(sampen):
        raise ValueError("Failed to calculate sample entropy (result is not finite)")

    return sampen


def calculate_non_linear_hrv(rr_intervals: np.ndarray) -> Dict:
    """
    Calculate all non-linear HRV metrics with comprehensive error handling
    and additional derived metrics.

    Args:
        rr_intervals: Array of RR intervals in milliseconds
    Returns:
        Dictionary containing all non-linear metrics
    Raises:
        ValueError: If input parameters are invalid
    """
    # Input validation
    if not isinstance(rr_intervals, np.ndarray):
        raise ValueError("RR intervals must be a numpy array")
    if len(rr_intervals) == 0:
        raise ValueError("Empty RR interval array")
    if len(rr_intervals) < 4:  # Minimum length for most non-linear metrics
        raise ValueError("At least 4 RR intervals are required for non-linear analysis")

    try:
        rr_intervals = validate_rr_intervals(rr_intervals)
        metrics = {}

        # Calculate Poincaré plot metrics
        try:
            poincare_metrics = calculate_poincare_metrics(rr_intervals)
            metrics.update(poincare_metrics)
        except ValueError as e:
            raise ValueError(f"Failed to calculate Poincaré metrics: {str(e)}")

        # Calculate entropy metrics
        try:
            metrics["sampen"] = calculate_sample_entropy(rr_intervals)
        except ValueError as e:
            raise ValueError(f"Failed to calculate sample entropy: {str(e)}")

        try:
            metrics["apen"] = calculate_approximate_entropy(rr_intervals)
        except ValueError as e:
            raise ValueError(f"Failed to calculate approximate entropy: {str(e)}")

        # Calculate DFA metrics
        try:
            dfa_metrics = calculate_dfa(rr_intervals)
            metrics.update(dfa_metrics)
        except Exception as e:
            import warnings

            warnings.warn(
                f"Error calculating DFA metrics: {str(e)}. Setting DFA metrics to NaN."
            )
            # Instead of re-raising, set DFA metrics to NaN
            metrics.update(
                {
                    "alpha1": np.nan,
                    "alpha2": np.nan,
                    "alpha_overall": np.nan,
                    "r_squared": np.nan,
                }
            )

        # Filter only scalar values for the finiteness check
        scalar_metrics = [v for v in metrics.values() if np.isscalar(v)]
        if not any(np.isfinite(scalar_metrics)):
            raise ValueError("No finite scalar metrics computed")

        return metrics

    except Exception as e:
        warnings.warn(f"Error calculating non-linear metrics: {str(e)}")
        # Return default values for all metrics
        return {
            "sd1": np.nan,
            "sd2": np.nan,
            "sd1_sd2_ratio": np.nan,
            "ellipse_area": np.nan,
            "csi": np.nan,
            "cvi": np.nan,
            "approximate_entropy": np.nan,
            "sample_entropy_m1": np.nan,
            "sample_entropy_m2": np.nan,
            "sample_entropy_m3": np.nan,
            "multiscale_entropy": [np.nan] * 5,
            "alpha1": np.nan,
            "alpha2": np.nan,
            "alpha_overall": np.nan,
            "r_squared": np.nan,
            "recurrence_rate": np.nan,
            "determinism": np.nan,
        }


def calculate_complete_hrv(rr_intervals: np.ndarray, fs: float = 4.0) -> Dict:
    """
    Calculate all HRV metrics (time domain, frequency domain, and non-linear)

    Args:
        rr_intervals: Array of RR intervals in milliseconds
        fs: Sampling frequency for frequency domain analysis
    Returns:
        Dictionary containing all HRV metrics
    """
    time_metrics = calculate_time_domain_hrv(rr_intervals)
    freq_metrics = calculate_frequency_domain_hrv(rr_intervals, fs)
    non_linear_metrics = calculate_non_linear_hrv(rr_intervals)

    return {
        "time_domain": time_metrics,
        "frequency_domain": freq_metrics,
        "non_linear": non_linear_metrics,
    }


def calculate_advanced_hrv(rr_intervals: np.ndarray, fs: float = 1000.0) -> Dict:
    """
    Calculate comprehensive HRV metrics including time-domain, frequency-domain,
    and non-linear metrics.

    Parameters
    ----------
    rr_intervals : np.ndarray
        Array of RR intervals in milliseconds
    fs : float
        Sampling frequency of the original ECG signal

    Returns
    -------
    Dict
        Dictionary containing:
        - Time domain metrics:
            - SDNN: Standard deviation of NN intervals
            - RMSSD: Root mean square of successive differences
            - pNN50: Proportion of NN50 divided by total number of NNs
            - SDANN: Standard deviation of average NN intervals
            - SDNN index: Mean of standard deviations of NN intervals
        - Frequency domain metrics:
            - VLF power: Very low frequency (0.003-0.04 Hz)
            - LF power: Low frequency (0.04-0.15 Hz)
            - HF power: High frequency (0.15-0.4 Hz)
            - LF/HF ratio: Ratio of LF to HF power
            - Total power: Total spectral power
        - Non-linear metrics:
            - SD1, SD2: Poincaré plot descriptors
            - ApEn: Approximate entropy
            - SampEn: Sample entropy
            - DFA: Detrended fluctuation analysis (α1, α2)
            - Correlation dimension
    """
    try:
        # Validate sampling frequency
        if fs <= 0:
            raise ValueError("Sampling frequency must be positive")

        # Validate RR intervals
        rr_intervals = validate_rr_intervals(rr_intervals)

        results = {}

        # Time domain metrics
        try:
            time_metrics = _calculate_time_domain_metrics(rr_intervals)
            results.update(time_metrics)
        except Exception as e:
            warnings.warn(f"Error in time domain calculations: {str(e)}")
            results.update(_get_default_time_metrics())

        # Frequency domain metrics
        try:
            freq_metrics = _calculate_frequency_domain_metrics(rr_intervals, fs)
            results.update(freq_metrics)
        except Exception as e:
            warnings.warn(f"Error in frequency domain calculations: {str(e)}")
            results.update(_get_default_freq_metrics())

        # Non-linear metrics
        try:
            nonlinear_metrics = _calculate_nonlinear_metrics(rr_intervals)
            results.update(nonlinear_metrics)
        except Exception as e:
            warnings.warn(f"Error in non-linear calculations: {str(e)}")
            results.update(_get_default_nonlinear_metrics())

        return results

    except Exception as e:
        raise ValueError(f"Error in HRV calculation: {str(e)}")


def _calculate_time_domain_metrics(rr_intervals: np.ndarray) -> Dict:
    """Calculate time domain HRV metrics."""
    try:
        # Calculate successive differences
        diff_rr = np.diff(rr_intervals)

        # Calculate NN50 (number of successive differences > 50ms)
        nn50 = np.sum(np.abs(diff_rr) > 50)

        return {
            "SDNN": float(np.std(rr_intervals)),
            "RMSSD": float(np.sqrt(np.mean(diff_rr**2))),
            "pNN50": float(nn50 / len(diff_rr)) * 100,
            "SDANN": float(_calculate_sdann(rr_intervals)),
            "SDNN_index": float(_calculate_sdnn_index(rr_intervals)),
            "Mean_HR": float(60000 / np.mean(rr_intervals)),  # Convert to BPM
            "STD_HR": float(np.std(60000 / rr_intervals)),
        }
    except Exception as e:
        warnings.warn(f"Error in time domain calculations: {str(e)}")
        raise


def _calculate_frequency_domain_metrics(rr_intervals: np.ndarray, fs: float) -> Dict:
    """Calculate frequency domain HRV metrics."""
    try:
        # Interpolate RR intervals to get evenly sampled signal
        time_points = np.cumsum(rr_intervals) / 1000.0  # Convert to seconds
        f_interp = 4.0  # Hz, interpolation frequency
        t_interp = np.arange(0, time_points[-1], 1 / f_interp)
        rr_interp = np.interp(t_interp, time_points, rr_intervals)

        # Calculate PSD using Welch's method
        frequencies, psd = signal.welch(rr_interp, fs=f_interp, nperseg=256)

        # Define frequency bands
        vlf_mask = (frequencies >= 0.003) & (frequencies < 0.04)
        lf_mask = (frequencies >= 0.04) & (frequencies < 0.15)
        hf_mask = (frequencies >= 0.15) & (frequencies < 0.4)

        # Calculate powers in each band
        # vlf_power = np.trapz(psd[vlf_mask], frequencies[vlf_mask])
        # lf_power = np.trapz(psd[lf_mask], frequencies[lf_mask])
        # hf_power = np.trapz(psd[hf_mask], frequencies[hf_mask])
        vlf_power = trapezoid(psd[vlf_mask], frequencies[vlf_mask])
        lf_power = trapezoid(psd[lf_mask], frequencies[lf_mask])
        hf_power = trapezoid(psd[hf_mask], frequencies[hf_mask])
        total_power = vlf_power + lf_power + hf_power

        return {
            "VLF_power": float(vlf_power),
            "LF_power": float(lf_power),
            "HF_power": float(hf_power),
            "LF_HF_ratio": float(lf_power / hf_power if hf_power > 0 else 0),
            "Total_power": float(total_power),
            "LF_normalized": float(100 * lf_power / (lf_power + hf_power)),
            "HF_normalized": float(100 * hf_power / (lf_power + hf_power)),
        }
    except Exception as e:
        warnings.warn(f"Error in frequency domain calculations: {str(e)}")
        raise


def _calculate_nonlinear_metrics(rr_intervals: np.ndarray) -> Dict:
    """Calculate non-linear HRV metrics."""
    try:
        # Validate input
        if len(rr_intervals) < 4:
            raise ValueError("At least 4 RR intervals required for non-linear metrics")

        if np.any(rr_intervals <= 0):
            raise ValueError("RR intervals must be positive")

        # Poincaré plot metrics
        diff_rr = np.diff(rr_intervals)
        sd1 = np.std(diff_rr) / np.sqrt(2) if len(diff_rr) > 0 else np.nan
        sd2 = np.std(rr_intervals) if len(rr_intervals) > 1 else np.nan

        # Sample Entropy with error handling
        try:
            sampen = calculate_sample_entropy(rr_intervals)
        except Exception:
            sampen = np.nan

        # Approximate Entropy with error handling
        try:
            apen = calculate_approximate_entropy(rr_intervals)
        except Exception:
            apen = np.nan

        # Detrended Fluctuation Analysis with error handling
        try:
            if len(rr_intervals) >= 16:
                dfa_alpha1 = _calculate_dfa(rr_intervals, [4, 16])
                dfa_alpha2 = _calculate_dfa(rr_intervals, [16, 64])
            else:
                dfa_alpha1 = np.nan
                dfa_alpha2 = np.nan
        except Exception:
            dfa_alpha1 = np.nan
            dfa_alpha2 = np.nan

        return {
            "SD1": float(sd1),
            "SD2": float(sd2),
            "SD1_SD2_ratio": float(sd1 / sd2)
            if (sd2 > 0 and np.isfinite(sd1))
            else np.nan,
            "SampEn": float(sampen),
            "ApEn": float(apen),
            "DFA_alpha1": float(dfa_alpha1),
            "DFA_alpha2": float(dfa_alpha2),
        }

    except Exception as e:
        warnings.warn(f"Error in non-linear calculations: {str(e)}")
        return {
            "SD1": np.nan,
            "SD2": np.nan,
            "SD1_SD2_ratio": np.nan,
            "SampEn": np.nan,
            "ApEn": np.nan,
            "DFA_alpha1": np.nan,
            "DFA_alpha2": np.nan,
        }


def _calculate_sdann(rr_intervals: np.ndarray, window: int = 300000) -> float:
    """Calculate SDANN (Standard deviation of 5-min interval means)."""
    try:
        # Split into 5-minute segments (window in milliseconds)
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
    """Calculate SDNN index (Mean of 5-min interval SDs)."""
    try:
        # Split into 5-minute segments
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


def _calculate_dfa(signal: np.ndarray, scale_range: List[int]) -> float:
    """Calculate DFA (Detrended Fluctuation Analysis) scaling exponent."""
    try:
        if len(signal) < scale_range[1]:
            raise ValueError(
                f"Signal length ({len(signal)}) must be greater than maximum scale ({scale_range[1]})"
            )

        # Ensure signal is valid
        if np.any(signal <= 0):
            raise ValueError("Signal must contain only positive values")

        # Implementation of DFA algorithm
        scales = np.arange(scale_range[0], min(scale_range[1], len(signal) // 4))
        if len(scales) < 2:
            raise ValueError("Not enough scales for DFA calculation")

        fluctuations = np.zeros(len(scales))

        for i, scale in enumerate(scales):
            # Split signal into windows
            n_windows = len(signal) // scale
            if n_windows == 0:
                continue

            windows = np.array_split(signal[: n_windows * scale], n_windows)

            # Calculate local trends and fluctuations
            fluct = 0
            for window in windows:
                x = np.arange(len(window))
                coef = np.polyfit(x, window, 1)
                trend = np.polyval(coef, x)
                fluct += np.sum((window - trend) ** 2)

            fluctuations[i] = np.sqrt(fluct / (n_windows * scale))

        # Remove zero fluctuations
        valid_idx = fluctuations > 0
        if not np.any(valid_idx):
            raise ValueError("No valid fluctuations calculated")

        valid_scales = scales[valid_idx]
        valid_fluct = fluctuations[valid_idx]

        if len(valid_scales) < 2:
            raise ValueError("Not enough valid scales for DFA calculation")

        # Calculate scaling exponent
        coef = np.polyfit(np.log(valid_scales), np.log(valid_fluct), 1)
        return coef[0]  # Return the slope

    except Exception as e:
        warnings.warn(f"Error in DFA calculation: {str(e)}")
        return np.nan


def _get_default_time_metrics() -> Dict:
    """Return default time domain metrics with NaN values."""
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
    """Return default frequency domain metrics with NaN values."""
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
    """Return default non-linear metrics with NaN values."""
    return {
        "SD1": np.nan,
        "SD2": np.nan,
        "SD1_SD2_ratio": np.nan,
        "SampEn": np.nan,
        "ApEn": np.nan,
        "DFA_alpha1": np.nan,
        "DFA_alpha2": np.nan,
    }


if __name__ == "__main__":
    # 1. Create dummy ECG signals.
    fs = 500  # Sampling rate in Hz
    duration = 10  # Duration in seconds
    t = np.arange(0, duration, 1 / fs)

    # Simulate an "original" ECG signal as a noisy sine wave.
    original_signal = np.sin(2 * np.pi * 1 * t) + 0.2 * np.random.randn(len(t))
    # A "processed" version is simply the clean sine wave.
    processed_signal = np.sin(2 * np.pi * 1 * t)

    # 2. Generate dummy QRS peaks and RR intervals.
    # For demonstration, create some variability in the RR intervals.
    # Here, we generate 10 beats with an average RR interval of ~800 ms and some noise.
    dummy_rr_intervals = 800 + 50 * np.random.randn(10)  # in milliseconds

    # Use the dummy RR intervals to compute dummy peak locations (in sample indices).
    # Start at index 0 and add intervals converted from ms to samples.
    dummy_peaks = [0]
    for rr in dummy_rr_intervals:
        dummy_peaks.append(dummy_peaks[-1] + int(rr / 1000 * fs))
    dummy_peaks = np.array(dummy_peaks)

    # 3. Create dummy frequency & PSD data (used for the comprehensive analysis plot).
    dummy_frequencies = np.linspace(0.1, 30, 100)
    dummy_psd = np.abs(np.sin(dummy_frequencies))  # Dummy PSD values

    # Build a results dictionary (mimicking the output structure expected by plot_comprehensive_analysis).
    dummy_results = {
        "original_signal": original_signal,
        "processed_signal": processed_signal,
        "peaks": dummy_peaks,
        "hrv_metrics": {"rr_intervals": dummy_rr_intervals},
        "frequencies": dummy_frequencies,
        "psd": dummy_psd,
    }

    # 4. Plot the signal comparison (original vs. processed).
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

    # 7. Plot the Poincaré plot using the dummy RR intervals.
    fig, ax = plot_poincare(dummy_rr_intervals)
    ax.set_title("Dummy Poincaré Plot")
    plt.show()
