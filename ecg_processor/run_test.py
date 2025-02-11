#!/usr/bin/env python3
"""
A complete pipeline for ECG preprocessing.

This script:
  • Generates a synthetic ECG signal using NeuroKit2.
  • Initializes the ECG preprocessor for your codebase.
  • Processes the signal to remove noise, detect QRS complexes, segment beats,
    compute HRV and quality metrics, and extract beat features.
  • Prints some key outputs and plots the original and processed signals.
"""

import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from ecg_processor.ecg_preprocessor import ECGPreprocessor


def generate_ecg_signal(duration=10, sampling_rate=500):
    """
    Generate a synthetic ECG signal using NeuroKit2.

    Parameters:
      duration (int): Duration of the ECG signal in seconds.
      sampling_rate (int): Number of samples per second.

    Returns:
      np.ndarray: Simulated ECG signal.
    """
    # Adjust noise level as needed (here, using a low noise setting)
    ecg_signal = nk.ecg_simulate(
        duration=duration, sampling_rate=sampling_rate, noise=0.05
    )
    return ecg_signal


def run_preprocessing_pipeline(ecg_signal, sampling_rate=500):
    """
    Run the full ECG preprocessing pipeline.

    Parameters:
      ecg_signal (np.ndarray): Input ECG signal.
      sampling_rate (int): Sampling rate used to generate the signal.

    Returns:
      dict: Dictionary of processed outputs including:
            'original_signal', 'processed_signal', 'peaks',
            'beats', 'hrv_metrics', 'beat_features', and 'quality_metrics'.
    """
    # Instantiate the preprocessor; set debug=True for more logging (optional)
    preprocessor = ECGPreprocessor(sampling_rate=sampling_rate, debug=True)

    # Run the complete processing pipeline on the input signal.
    results = preprocessor.process_signal(ecg_signal)

    return results


def plot_results(original, processed, sampling_rate=500):
    """
    Plot the original and processed ECG signals side by side.

    Parameters:
      original (np.ndarray): Original ECG signal.
      processed (np.ndarray): Processed (denoised and filtered) ECG signal.
      sampling_rate (int): Sampling rate (samples per second).
    """
    t = np.linspace(0, len(original) / sampling_rate, len(original))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t, original, color="blue", label="Original ECG")
    plt.title("Original ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, processed, color="green", label="Processed ECG")
    plt.title("Processed ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    # Define parameters for the simulation
    sampling_rate = 500  # samples per second
    duration = 10  # seconds

    # Step 1: Generate a synthetic ECG signal.
    ecg_signal = generate_ecg_signal(duration=duration, sampling_rate=sampling_rate)

    # Step 2: Run the complete preprocessing pipeline.
    results = run_preprocessing_pipeline(ecg_signal, sampling_rate=sampling_rate)

    # Step 3: Print output summaries.
    print("\n--- Preprocessing Outputs ---")
    print("Number of detected peaks:", len(results.get("peaks", [])))
    print("HRV Metrics:", results.get("hrv_metrics", {}))
    print("Quality Metrics:", results.get("quality_metrics", {}))

    # Step 4: Plot the original and processed signals.
    if "processed_signal" in results:
        plot_results(
            ecg_signal, results["processed_signal"], sampling_rate=sampling_rate
        )
    else:
        print("Processed signal not found in the results.")


if __name__ == "__main__":
    main()
