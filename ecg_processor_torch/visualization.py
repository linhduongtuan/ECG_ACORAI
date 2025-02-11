import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, Tuple
import logging
import datetime
from ecg_processor.quality import (
    _calculate_snr,
    _assess_baseline_wander,
    _calculate_power_line_noise,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

# Set default plot style and parameters
plt.style.use("default")
plt.rcParams.update(
    {
        "figure.figsize": (12, 8),
        "figure.dpi": 100,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    }
)

DEFAULT_FIGSIZE = (12, 8)
DEFAULT_DPI = 100

def _to_numpy(data):
    """Convert data to a numpy array if it is a torch tensor. Otherwise return as is."""
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return data

def plot_signal_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    fs: float,
    title: str = "ECG Signal Comparison",
    time_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot original and processed signals with enhanced visualization options.

    Accepts inputs either as numpy arrays or torch tensors.
    """
    try:
        # Convert to numpy if needed.
        original = _to_numpy(original)
        processed = _to_numpy(processed)

        if not isinstance(original, np.ndarray) or not isinstance(processed, np.ndarray):
            raise ValueError("Signals must be numpy arrays or torch tensors")
        if len(original) != len(processed):
            raise ValueError("Signals must have the same length")
        if not np.isfinite(original).all() or not np.isfinite(processed).all():
            raise ValueError("Signals contain non-finite values")
        if fs <= 0:
            raise ValueError("Sampling frequency must be positive")

        # Create time array
        time = np.arange(len(original)) / fs

        # Handle time range if provided
        if time_range is not None:
            start_idx = int(time_range[0] * fs)
            end_idx = int(time_range[1] * fs)
            time = time[start_idx:end_idx]
            original = original[start_idx:end_idx]
            processed = processed[start_idx:end_idx]

        # Create figure and subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=DEFAULT_DPI)
        fig.suptitle(title, fontsize=14)

        ax1.plot(time, original, "b-", label="Original", linewidth=1)
        ax1.set_title("Original Signal")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(time, processed, "g-", label="Processed", linewidth=1)
        ax2.set_title("Processed Signal")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Compute and display SNR (using variance-based calculation)
        snr_value = 10 * np.log10(np.var(processed) / np.var(original - processed))
        ax2.text(
            0.02,
            0.98,
            f"SNR: {snr_value:.1f} dB",
            transform=ax2.transAxes,
            verticalalignment="top",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")

        plt.show()

    except Exception as e:
        logger.error(f"Error plotting signal comparison: {str(e)}")
        raise

def plot_comprehensive_analysis(
    results: Dict, figsize: Tuple[int, int] = (15, 12), save_path: Optional[str] = None
) -> None:
    """
    Plot comprehensive ECG analysis results with enhanced visualization.

    Expects the results dictionary to contain keys:
    "original_signal", "processed_signal", "peaks", and "hrv_metrics".
    """
    try:
        required_keys = ["original_signal", "processed_signal", "peaks", "hrv_metrics"]
        for key in required_keys:
            if key not in results:
                raise ValueError(f"Missing required key: {key}")

        # Convert signals if torch tensors.
        results["original_signal"] = _to_numpy(results["original_signal"])
        results["processed_signal"] = _to_numpy(results["processed_signal"])

        fig = plt.figure(figsize=figsize, dpi=DEFAULT_DPI)
        gs = plt.GridSpec(4, 2, figure=fig)

        # 1. Signal Comparison
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(results["original_signal"], "b-", label="Original", alpha=0.6)
        ax1.plot(results["processed_signal"], "g-", label="Processed")
        ax1.set_title("Signal Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. QRS Detection
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(results["processed_signal"], "b-")
        ax2.plot(
            results["peaks"],
            results["processed_signal"][results["peaks"]],
            "ro",
            label="R-Peaks",
        )
        ax2.set_title("QRS Detection")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. HRV Analysis
        ax3 = fig.add_subplot(gs[2, 0])
        rr_intervals = results["hrv_metrics"]["rr_intervals"]
        ax3.plot(rr_intervals, "b-")
        ax3.set_title("RR Intervals")
        ax3.set_xlabel("Beat number")
        ax3.set_ylabel("RR interval (ms)")
        ax3.grid(True, alpha=0.3)

        # 4. Poincaré Plot
        ax4 = fig.add_subplot(gs[2, 1])
        rr_n = rr_intervals[:-1]
        rr_n1 = rr_intervals[1:]
        ax4.plot(rr_n, rr_n1, "bo", alpha=0.5, markersize=2)
        ax4.set_title("Poincaré Plot")
        ax4.set_xlabel("RR_n (ms)")
        ax4.set_ylabel("RR_n+1 (ms)")
        ax4.grid(True, alpha=0.3)

        if "sd1" in results["hrv_metrics"] and "sd2" in results["hrv_metrics"]:
            from matplotlib.patches import Ellipse
            sd1 = results["hrv_metrics"]["sd1"]
            sd2 = results["hrv_metrics"]["sd2"]
            mean_rr = np.mean(rr_intervals)
            ellipse = Ellipse(
                (mean_rr, mean_rr),
                width=2 * sd2,
                height=2 * sd1,
                angle=45,
                fill=False,
                linestyle="--",
                color="r",
            )
            ax4.add_patch(ellipse)
            ax4.text(
                0.02,
                0.98,
                f"SD1: {sd1:.1f}ms\nSD2: {sd2:.1f}ms",
                transform=ax4.transAxes,
                verticalalignment="top",
            )

        # 5. Power Spectral Density
        if "psd" in results and "frequencies" in results:
            ax5 = fig.add_subplot(gs[3, :])
            ax5.semilogy(results["frequencies"], results["psd"])
            ax5.set_title("Power Spectral Density")
            ax5.set_xlabel("Frequency (Hz)")
            ax5.set_ylabel("Power (dB/Hz)")
            ax5.grid(True, alpha=0.3)

            vlf_mask = results["frequencies"] <= 0.04
            lf_mask = (results["frequencies"] > 0.04) & (results["frequencies"] <= 0.15)
            hf_mask = (results["frequencies"] > 0.15) & (results["frequencies"] <= 0.4)

            ax5.fill_between(results["frequencies"][vlf_mask],
                             results["psd"][vlf_mask],
                             alpha=0.3,
                             label="VLF")
            ax5.fill_between(results["frequencies"][lf_mask],
                             results["psd"][lf_mask],
                             alpha=0.3,
                             label="LF")
            ax5.fill_between(results["frequencies"][hf_mask],
                             results["psd"][hf_mask],
                             alpha=0.3,
                             label="HF")
            ax5.legend()

            if all(k in results["hrv_metrics"] for k in ["lf_hf_ratio", "total_power"]):
                ax5.text(
                    0.02,
                    0.98,
                    f"LF/HF: {results['hrv_metrics']['lf_hf_ratio']:.2f}\nTotal Power: {results['hrv_metrics']['total_power']:.1f}ms²",
                    transform=ax5.transAxes,
                    verticalalignment="top",
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")

        plt.show()

    except Exception as e:
        logger.error(f"Error plotting comprehensive analysis: {str(e)}")
        raise

def plot_advanced_analysis(data, **kwargs):
    """
    Plot an advanced analysis overview of the ECG data.
    This example simply plots a histogram.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=30, color="skyblue", edgecolor="black")
    ax.set_title("Advanced Analysis")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig

def plot_beat_template(
    template: np.ndarray,
    fs: float,
    confidence_interval: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot beat template with optional 95% confidence interval.
    Accepts template and confidence_interval as numpy arrays or torch tensors.
    """
    try:
        template = _to_numpy(template)
        if confidence_interval is not None:
            confidence_interval = _to_numpy(confidence_interval)
            if confidence_interval.shape[0] != 2 or confidence_interval.shape[1] != len(template):
                raise ValueError("Confidence interval must be 2xN array")
        if fs <= 0:
            raise ValueError("Sampling frequency must be positive")

        time = np.arange(len(template)) / fs * 1000
        fig, ax = plt.subplots(figsize=figsize, dpi=DEFAULT_DPI)
        if confidence_interval is not None:
            ax.fill_between(time,
                            confidence_interval[0],
                            confidence_interval[1],
                            alpha=0.3,
                            color="gray",
                            label="95% CI")
        ax.plot(time, template, "b-", label="Template", linewidth=2)
        ax.set_title("Beat Template")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Identify key points (P-wave, R-peak, T-wave)
        p_wave_idx = np.argmax(template[: int(0.3 * len(template))])
        qrs_idx = np.argmax(np.abs(template))
        t_wave_idx = len(template) - 1 - np.argmax(template[int(0.6 * len(template)) :][::-1])
        ax.plot(time[p_wave_idx], template[p_wave_idx], "go", label="P-wave")
        ax.plot(time[qrs_idx], template[qrs_idx], "ro", label="R-peak")
        ax.plot(time[t_wave_idx], template[t_wave_idx], "mo", label="T-wave")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        plt.show()

    except Exception as e:
        logger.error(f"Error plotting beat template: {str(e)}")
        raise

def plot_quality_metrics(
    signal: np.ndarray,
    fs: float,
    window_size: float = 5.0,
    overlap: float = 0.5,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot signal quality metrics (SNR, baseline wander, power line noise) over time.
    Accepts signal as a numpy array or torch tensor.
    """
    try:
        signal = _to_numpy(signal)
        if not isinstance(signal, np.ndarray):
            raise ValueError("Signal must be a numpy array or torch tensor")
        if not np.isfinite(signal).all():
            raise ValueError("Signal contains non-finite values")
        if fs <= 0:
            raise ValueError("Sampling frequency must be positive")
        if not 0 <= overlap < 1:
            raise ValueError("Overlap must be between 0 and 1")

        window_samples = int(window_size * fs)
        step = int(window_samples * (1 - overlap))
        n_windows = (len(signal) - window_samples) // step + 1

        time_arr = np.arange(n_windows) * step / fs
        snr = np.zeros(n_windows)
        baseline_wander = np.zeros(n_windows)
        power_line_noise = np.zeros(n_windows)

        for i in range(n_windows):
            window = signal[i * step : i * step + window_samples]
            snr[i] = _calculate_snr(window, fs)["SNR"]
            baseline_wander[i] = _assess_baseline_wander(window, fs)["baseline_wander_severity"]
            power_line_noise[i] = _calculate_power_line_noise(window, fs)["power_line_noise"]

        fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)
        axs[0].plot(time_arr, snr)
        axs[0].set_ylabel("SNR (dB)")
        axs[1].plot(time_arr, baseline_wander)
        axs[1].set_ylabel("Baseline Wander")
        axs[2].plot(time_arr, power_line_noise)
        axs[2].set_ylabel("Power Line Noise")
        axs[2].set_xlabel("Time (s)")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    except Exception as e:
        logger.error(f"Error plotting quality metrics: {str(e)}")
        raise

def generate_analysis_report(
    signal: np.ndarray,
    fs: float,
    features: Dict,
    quality_metrics: Dict,
    waves: Optional[Dict] = None,
) -> str:
    """
    Generate a comprehensive textual analysis report of the ECG signal.
    """
    try:
        quality_section = ["Signal Quality Assessment", "----"]
        quality_section.extend([
            f"Overall Signal Quality: {quality_metrics.get('overall_quality', 0):.2f}",
            f"Signal-to-Noise Ratio: {quality_metrics.get('SNR', 0):.1f} dB",
            f"Baseline Stability: {'Stable' if quality_metrics.get('baseline_stable', False) else 'Unstable'}",
            f"Power Line Interference: {'Present' if quality_metrics.get('powerline_interference_present', True) else 'Not Detected'}",
        ])

        hrv_section = ["Heart Rate Variability Analysis", "----"]
        hr = features.get("Mean_HR", 0)
        hrv_section.extend([
            f"Mean Heart Rate: {hr:.1f} bpm",
            f"SDNN: {features.get('SDNN', 0):.2f} ms",
            f"RMSSD: {features.get('RMSSD', 0):.2f} ms",
            f"pNN50: {features.get('pNN50', 0):.1f}%",
            f"LF/HF Ratio: {features.get('LF_HF_ratio', 0):.2f}",
        ])

        morph_section = ["Morphological Analysis", "----"]
        morph_section.extend([
            f"QT Interval: {features.get('QT_interval', 0):.1f} ms",
            f"QTc (Bazett): {features.get('QTc_Bazett', 0):.1f} ms",
            f"PR Interval: {features.get('PR_interval', 0):.1f} ms",
            f"QRS Duration: {features.get('QRS_duration', 0):.1f} ms",
        ])

        stats_section = ["Statistical Summary", "----"]
        signal_stats = {
            "Mean": np.mean(signal),
            "Std": np.std(signal),
            "Max": np.max(signal),
            "Min": np.min(signal),
            "Range": np.ptp(signal),
        }
        stats_section.extend([
            f"Signal Mean: {signal_stats['Mean']:.2f}",
            f"Signal Std: {signal_stats['Std']:.2f}",
            f"Signal Range: {signal_stats['Range']:.2f}",
        ])

        clinical_section = ["Clinical Indicators", "----"]
        hr_status = "Normal"
        if hr > 100:
            hr_status = "Tachycardia"
        elif hr < 60:
            hr_status = "Bradycardia"
        st_status = "Normal"
        if features.get("ST_elevation", False):
            st_status = "ST Elevation"
        elif features.get("ST_depression", False):
            st_status = "ST Depression"
        clinical_section.extend([
            f"Heart Rate Status: {hr_status}",
            f"ST Segment Status: {st_status}",
            f"T-Wave Alternans: {'Present' if features.get('TWA_present', False) else 'Not Detected'}",
        ])

        recommend_section = ["Recommendations", "----"]
        recommendations = []
        if quality_metrics.get("overall_quality", 0) < 0.6:
            recommendations.append("- Improve signal quality for more reliable analysis")
        if quality_metrics.get("baseline_wander_severity", 0) > 0.3:
            recommendations.append("- Reduce patient movement to minimize baseline wander")
        if quality_metrics.get("powerline_interference_present", False):
            recommendations.append("- Check electrode connections and power line isolation")
        if hr_status != "Normal":
            recommendations.append(f"- Monitor heart rate ({hr_status} detected)")
        if st_status != "Normal":
            recommendations.append("- Further investigation of ST segment changes recommended")
        if recommendations:
            recommend_section.extend(recommendations)
        else:
            recommend_section.append("- No specific recommendations")

        sections = [
            "\n".join(quality_section),
            "\n".join(hrv_section),
            "\n".join(morph_section),
            "\n".join(stats_section),
            "\n".join(clinical_section),
            "\n".join(recommend_section),
        ]
        report = "\n\n".join(sections)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"ECG Analysis Report\nGenerated: {timestamp}\n{'=' * 50}\n\n"
        return header + report

    except Exception as e:
        logger.error(f"Error generating analysis report: {str(e)}")
        return f"Error generating report: {str(e)}"


def test_plot_quality_metrics():
    t = np.linspace(0, 10, 1000)
    test_signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.random.randn(len(t))
    fs = 100

    try:
        plot_quality_metrics(test_signal, fs)
        print("Test passed: Valid inputs")
    except Exception as e:
        print(f"Test failed: Valid inputs - {str(e)}")

    try:
        plot_quality_metrics([1, 2, 3], fs)
        print("Test failed: Invalid signal type")
    except ValueError:
        print("Test passed: Invalid signal type")

    try:
        plot_quality_metrics(test_signal, -1)
        print("Test failed: Invalid fs")
    except ValueError:
        print("Test passed: Invalid fs")

    try:
        plot_quality_metrics(test_signal, fs, overlap=1.5)
        print("Test failed: Invalid overlap")
    except ValueError:
        print("Test passed: Invalid overlap")

    try:
        plot_quality_metrics(test_signal, fs, save_path="test_plot.png")
        print("Test passed: Save plot")
    except Exception as e:
        print(f"Test failed: Save plot - {str(e)}")

if __name__ == "__main__":
    test_plot_quality_metrics()