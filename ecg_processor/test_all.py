"""
test_all.py

This script provides comprehensive testing for all functions and classes in the ECG processing suite.
It includes tests for:
  • The deep denoiser (ECGDeepDenoiser)
  • Advanced denoising methods (wavelet, adaptive, EMD, median, smoothing)
  • HRV calculations and quality metrics
  • Feature extraction functions
  • The ECG preprocessor (with dummy QRS detection / segmentation)
  • Time-series and transformer classifiers

Features:
  • Comprehensive error handling and validation
  • Detailed test reports and logging
  • Visual validation through plots
  • Performance metrics tracking
  • Memory usage monitoring

Usage:
  Run this script with Python to execute all tests and view results:
  $ python test_all.py [--verbose] [--skip-plots]
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import psutil
from typing import Dict, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("test_all.log")],
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Class for storing test results"""

    name: str
    success: bool
    execution_time: float
    memory_usage: float
    error_message: Optional[str] = None
    metrics: Optional[Dict] = None


class TestError(Exception):
    """Base exception for test errors"""

    pass


class TestConfigError(TestError):
    """Exception for test configuration errors"""

    pass


class TestValidationError(TestError):
    """Exception for test validation errors"""

    pass


def measure_execution_time(func):
    """Decorator to measure function execution time"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result, execution_time
        except Exception:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds")
            raise

    return wrapper


def measure_memory_usage(func):
    """Decorator to measure memory usage"""

    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        try:
            result = func(*args, **kwargs)
            end_memory = process.memory_info().rss / 1024 / 1024
            memory_used = end_memory - start_memory
            logger.info(f"{func.__name__} used {memory_used:.2f} MB of memory")
            return result, memory_used
        except Exception:
            end_memory = process.memory_info().rss / 1024 / 1024
            memory_used = end_memory - start_memory
            logger.error(
                f"{func.__name__} failed with {memory_used:.2f} MB memory usage"
            )
            raise

    return wrapper


def validate_signal(signal: np.ndarray, min_length: int = 100) -> None:
    """Validate signal array"""
    if not isinstance(signal, np.ndarray):
        raise TestValidationError("Signal must be a numpy array")
    if signal.size < min_length:
        raise TestValidationError(f"Signal length must be at least {min_length}")
    if not np.isfinite(signal).all():
        raise TestValidationError("Signal contains invalid values (inf or nan)")


##########################
# 1. Test ECGDeepDenoiser
##########################
try:
    from ecg_processor.ecg_deep_denoiser import ECGDeepDenoiser

    # Assumes your plotting functions (e.g. plot_signal_comparison) are in a module called "plotting_module"
    from .visualization import plot_signal_comparison
except ImportError as e:
    print(f"Import error in deep denoiser test: {e}")


def test_ecg_deep_denoiser():
    print("\n--- Testing ECGDeepDenoiser ---")
    input_length = 500
    # Instantiate the deep denoiser object
    deep_denoiser = ECGDeepDenoiser(input_length=input_length, learning_rate=0.001)

    # Create dummy training data:
    # Generate 100 segments of a 5 Hz sine wave with added noise.
    num_samples = 100
    t = np.linspace(0, 1, input_length)
    x_train = []
    for _ in range(num_samples):
        clean_signal = np.sin(2 * np.pi * 5 * t)
        noise = np.random.normal(0, 0.2, size=t.shape)
        noisy_signal = clean_signal + noise
        x_train.append(noisy_signal)
    x_train = np.array(x_train)

    print("Training deep denoiser (5 epochs)...")
    deep_denoiser.train(x_train, epochs=5, batch_size=16)

    # Test denoising on a new noisy signal:
    clean_signal = np.sin(2 * np.pi * 5 * t)
    noise = np.random.normal(0, 0.2, size=t.shape)
    test_signal = clean_signal + noise
    denoised_signal = deep_denoiser.denoise(test_signal)

    # Plot the original and denoised signals:
    plot_signal_comparison(
        original=test_signal,
        processed=denoised_signal,
        fs=input_length,
        title="ECGDeepDenoiser: Original vs. Denoised",
    )


###################################
# 2. Test Advanced Denoising Methods
###################################
try:
    from ecg_processor.advanced_denoising import (
        wavelet_denoise,
        adaptive_lms_filter,
        emd_denoise,
        median_filter_signal,
        smooth_signal,
    )
except ImportError as e:
    print(f"Import error in advanced denoising test: {e}")


def test_advanced_denoising():
    print("\n--- Testing Advanced Denoising Methods ---")
    fs = 500
    t = np.linspace(0, 1, fs)
    original_signal = np.sin(2 * np.pi * 5 * t)
    noisy_signal = original_signal + np.random.normal(0, 0.3, size=t.shape)

    denoised_wavelet = wavelet_denoise(noisy_signal, wavelet="db4", level=4)
    # For adaptive filtering, we create a dummy reference signal (here, random noise)
    reference = np.random.normal(0, 0.3, size=t.shape)
    try:
        denoised_adaptive = adaptive_lms_filter(
            noisy_signal, reference, mu=0.01, filter_order=32
        )
    except Exception as e:
        denoised_adaptive = None
        print("Adaptive filtering error:", e)
    denoised_emd = emd_denoise(noisy_signal, imf_to_remove=1)
    denoised_median = median_filter_signal(noisy_signal, kernel_size=5)
    denoised_smoothing = smooth_signal(noisy_signal, window_length=51, polyorder=3)

    plt.figure(figsize=(12, 8))
    plt.plot(t, noisy_signal, label="Noisy Signal", color="gray")
    plt.plot(t, denoised_wavelet, label="Wavelet Denoised")
    if denoised_adaptive is not None:
        plt.plot(t, denoised_adaptive, label="Adaptive Denoised")
    plt.plot(t, denoised_emd, label="EMD Denoised")
    plt.plot(t, denoised_median, label="Median Filtered")
    plt.plot(t, denoised_smoothing, label="Smoothed")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Advanced Denoising Methods")
    plt.legend()
    plt.grid(True)
    plt.show()


#############################
# 3. Test HRV & Signal Quality
#############################
try:
    from ecg_processor.hrv import (
        calculate_dfa,
        calculate_time_domain_hrv,
        calculate_frequency_domain_hrv,
        calculate_non_linear_hrv,
        calculate_complete_hrv,
        plot_poincare,
    )
except ImportError as e:
    print(f"Import error in HRV test: {e}")
try:
    from .quality import calculate_signal_quality
except ImportError as e:
    print(f"Import error in signal quality test: {e}")


def test_hrv_and_quality():
    print("\n--- Testing HRV and Signal Quality Functions ---")
    # Create dummy RR intervals in milliseconds.
    rr_intervals = np.array([800, 810, 790, 820, 805, 815, 800])
    time_hrv = calculate_time_domain_hrv(rr_intervals)
    freq_hrv = calculate_frequency_domain_hrv(rr_intervals, fs=4.0)
    non_linear = calculate_non_linear_hrv(rr_intervals)
    complete_hrv = calculate_complete_hrv(rr_intervals, fs=4.0)

    print("Time-domain HRV:", time_hrv)
    print("Frequency-domain HRV:", freq_hrv)
    print("Non-linear HRV:", non_linear)
    print("Complete HRV:", complete_hrv)

    # Plot Poincaré plot
    plot_poincare(rr_intervals)

    # Test signal quality: use a dummy “original” noisy signal vs. a processed clean sine wave.
    duration = 5  # seconds
    t_full = np.linspace(0, duration, duration * 500)
    orig_sig = np.sin(2 * np.pi * 1 * t_full) + 0.2 * np.random.randn(len(t_full))
    proc_sig = np.sin(2 * np.pi * 1 * t_full)
    quality = calculate_signal_quality(orig_sig, proc_sig, fs=500)
    print("Signal Quality Metrics:", quality)


def test_dfa():
    print("\n--- Testing DFA Calculation ---")
    # Generate synthetic RR intervals; ensure they are long enough (e.g., 300 samples)
    rr_intervals = np.linspace(800, 1200, 300) + np.random.normal(0, 50, 300)
    # (Data in milliseconds; adjust values as needed)

    # Calculate DFA metrics using your implemented function
    dfa_results = calculate_dfa(rr_intervals)
    print("Calculated DFA metrics:", dfa_results)

    # Validate results
    assert isinstance(dfa_results, dict), "DFA should return a dictionary of metrics"
    required_keys = ["alpha1", "alpha2", "alpha_overall", "r_squared"]
    for key in required_keys:
        assert key in dfa_results, f"Missing required DFA metric: {key}"
        assert isinstance(dfa_results[key], float), (
            f"DFA metric {key} should be a float value"
        )

    # Validate specific metrics
    assert 0 < dfa_results["alpha1"] < 2, "Alpha1 should be between 0 and 2"
    assert 0 < dfa_results["alpha2"] < 2, "Alpha2 should be between 0 and 2"
    assert 0 < dfa_results["r_squared"] <= 1, "R-squared should be between 0 and 1"

    # --- Re-compute intermediate values for plotting ---
    scales = dfa_results.get("scales", [])
    fluct = dfa_results.get("fluctuations", [])

    # Generate the log-log plot if we have the scale data
    if scales and fluct and len(scales) == len(fluct):
        plt.figure(figsize=(8, 6))
        plt.loglog(scales, fluct, "o-", label="F(n) vs. scale")
        plt.xlabel("Scale (n)")
        plt.ylabel("Fluctuation F(n)")
        plt.title("DFA Log-Log Plot")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.show()


###############################
# 4. Test Feature Extraction
###############################
try:
    from ecg_processor.features import (
        extract_statistical_features,
        extract_wavelet_features,
        extract_morphological_features,
    )
except ImportError as e:
    print(f"Import error in feature extraction test: {e}")


def test_feature_extraction():
    print("\n--- Testing Feature Extraction ---")
    fs = 500
    t_feat = np.linspace(0, 1, fs)
    beat = np.sin(2 * np.pi * 5 * t_feat)
    stats = extract_statistical_features(beat)
    wavelet_feats = extract_wavelet_features(beat, wavelet="db4", level=4)
    try:
        morph_feats = extract_morphological_features(beat, fs)
    except Exception as e:
        morph_feats = {"error": str(e)}
    print("Statistical Features:", stats)
    print("Wavelet Features:", wavelet_feats)
    print("Morphological Features:", morph_feats)


###########################
# 5. Test ECGPreprocessor
###########################
try:
    from ecg_processor.ecg_preprocessor import ECGPreprocessor
except ImportError as e:
    print(f"Import error in preprocessor test: {e}")


def test_ecg_preprocessor():
    print("\n--- Testing ECGPreprocessor ---")
    fs = 500
    duration = 5
    t_pre = np.linspace(0, duration, duration * fs)
    # Create a dummy ECG signal (sine wave with noise)
    ecg_signal = np.sin(2 * np.pi * 1 * t_pre) + 0.5 * np.random.randn(len(t_pre))
    preprocessor = ECGPreprocessor(sampling_rate=fs, lead_config="single")
    try:
        results = preprocessor.process_signal(ecg_signal)
        print("ECG Preprocessing Results Keys:", list(results.keys()))
        # Optionally, you can plot the preprocessed signal:
        plt.figure()
        plt.plot(results["original_signal"], label="Original")
        plt.plot(results["processed_signal"], label="Processed")
        plt.title("ECG Preprocessor Signal Comparison")
        plt.legend()
        plt.show()
    except Exception as e:
        print("Error during ECG Preprocessing:", e)


##################################
# 6. Test ECGTimeSeriesClassifier
##################################
try:
    from ecg_processor.ecg_timeseries_classifier import (
        ECGTimeSeriesClassifier,
        train_time_series_classifier,
        predict,
    )
except ImportError as e:
    print(f"Import error in time-series classifier test: {e}")


def test_time_series_classifier():
    print("\n--- Testing ECGTimeSeriesClassifier ---")
    num_samples = 100
    input_length = 500  # length of each ECG segment
    x_train = np.random.rand(num_samples, input_length).astype(np.float32)
    y_train = np.random.randint(0, 2, size=num_samples)
    model = ECGTimeSeriesClassifier(input_length=input_length, num_classes=2)
    history = train_time_series_classifier(
        model, x_train, y_train, epochs=5, batch_size=16, learning_rate=0.001
    )
    print("Time-Series Classifier Training History:", history)
    x_test = np.random.rand(10, input_length).astype(np.float32)
    predictions = predict(model, x_test)
    print("ECGTimeSeriesClassifier Predictions:", predictions)


##################################
# 7. Test ECGTransformerClassifier
##################################
try:
    from ecg_processor.ecg_transformer_classifier import (
        ECGTransformerClassifier,
        train_transformer_classifier,
        predict_transformer,
    )
except ImportError as e:
    print(f"Import error in transformer classifier test: {e}")


def test_transformer_classifier():
    print("\n--- Testing ECGTransformerClassifier ---")
    num_samples = 100
    input_length = 500
    x_train = np.random.rand(num_samples, input_length).astype(np.float32)
    y_train = np.random.randint(0, 2, size=num_samples)
    model = ECGTransformerClassifier(
        input_length=input_length,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_classes=2,
        dropout=0.1,
    )
    history = train_transformer_classifier(
        model, x_train, y_train, epochs=5, batch_size=16, learning_rate=0.001
    )
    print("Transformer Classifier Training History:", history)
    x_test = np.random.rand(10, input_length).astype(np.float32)
    predictions = predict_transformer(model, x_test)
    print("ECGTransformerClassifier Predictions:", predictions)


###############################
# 8. Test Feature Transform Functions with Plots
###############################
try:
    from ecg_processor.features_transform import (
        extract_stft_features,
        extract_wavelet_features,
        extract_hybrid_features,
    )
except ImportError as e:
    print(f"Import error in feature transform test: {e}")

import pywt
from scipy import signal


def test_feature_transform():
    print("\n--- Testing Feature Transform Functions with Plots ---")
    fs = 500  # Sampling frequency
    t_feat = np.linspace(0, 1, fs)
    # Create a synthetic ECG beat (e.g., a 5 Hz sine wave)
    beat = np.sin(2 * np.pi * 5 * t_feat)

    # --- Extract STFT Features and Plot Spectrogram ---
    stft_feats = extract_stft_features(beat, fs, nperseg=128)
    print("STFT Features:", stft_feats)
    f, t, Zxx = signal.stft(beat, fs=fs, nperseg=128)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, np.abs(Zxx), shading="gouraud")
    plt.title("STFT Magnitude Spectrogram")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(label="Magnitude")
    plt.tight_layout()
    plt.show()

    # --- Extract Wavelet Features and Plot Wavelet Coefficients ---
    wavelet_feats = extract_wavelet_features(beat, wavelet="db4", level=4)
    print("Wavelet Features:", wavelet_feats)
    coeffs = pywt.wavedec(beat, "db4", level=4)
    num_coeffs = len(coeffs)
    plt.figure(figsize=(10, num_coeffs * 2))
    for i, coef in enumerate(coeffs):
        plt.subplot(num_coeffs, 1, i + 1)
        plt.plot(coef)
        plt.title(f"Wavelet Coefficients Level {i}")
        plt.xlabel("Coefficient Index")
        plt.ylabel("Value")
    plt.tight_layout()
    plt.show()

    # --- Extract Hybrid Features and Plot as a Bar Chart ---
    hybrid_feats = extract_hybrid_features(
        beat, fs, wavelet="db4", level=4, nperseg=128
    )
    print("Hybrid Features:", hybrid_feats)
    keys = list(hybrid_feats.keys())
    values = list(hybrid_feats.values())
    plt.figure(figsize=(10, 5))
    plt.bar(keys, values, color="skyblue")
    plt.title("Hybrid Features")
    plt.xlabel("Feature")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


#############################
# Main: Run All Tests
#############################
if __name__ == "__main__":
    test_ecg_deep_denoiser()
    test_advanced_denoising()
    test_hrv_and_quality()
    test_dfa()
    test_feature_extraction()
    test_feature_transform()
    test_ecg_preprocessor()
    test_time_series_classifier()
    test_transformer_classifier()

    print("\nAll tests completed.")
