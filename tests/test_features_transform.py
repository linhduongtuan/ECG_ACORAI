# test_features_transform.py
import unittest
import numpy as np
import pywt
from scipy import signal
from ecg_processor.features_transform import (
    extract_stft_features,
    extract_wavelet_features,
    extract_hybrid_features,
    _spectral_centroid,
    _spectral_bandwidth,
    _spectral_rolloff,
    _spectral_flatness,
    _count_zero_crossings,
    _energy_ratio,
    _calculate_cross_domain_features,
)


class TestFeatureTransform(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic ECG signal for testing
        self.fs = 500  # Sampling frequency in Hz
        self.duration = 1  # second
        t = np.linspace(0, self.duration, int(self.duration * self.fs))
        self.valid_signal = np.sin(
            2 * np.pi * 1 * np.linspace(0, 1, self.fs)
        )  # 1-second sine wave
        self.short_signal = np.sin(
            2 * np.pi * 1 * np.linspace(0, 0.5, self.fs // 2)
        )  # 0.5-second sine wave

        # Create synthetic ECG-like signal
        self.clean_signal = (
            1.0 * np.sin(2 * np.pi * 1 * t)  # Basic rhythm
            + 0.5 * np.sin(2 * np.pi * 5 * t)  # QRS complex
            + 0.3 * np.sin(2 * np.pi * 10 * t)  # P wave
            + 0.2 * np.sin(2 * np.pi * 15 * t)  # T wave
        )
        self.noisy_signal = self.clean_signal + 0.1 * np.random.randn(len(t))

    def test_extract_stft_features(self):
        """Test STFT feature extraction."""
        # Test with valid input
        features = extract_stft_features(self.clean_signal, self.fs)
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)

        # Check specific features
        self.assertIn("stft_mean_energy", features)
        self.assertIn("stft_spectral_centroid", features)
        self.assertIn("stft_power_low", features)

        # Test with invalid inputs
        with self.assertRaises(ValueError):
            extract_stft_features([], self.fs)  # Empty signal
        with self.assertRaises(ValueError):
            extract_stft_features(self.clean_signal, -1)  # Invalid sampling rate
        with self.assertRaises(ValueError):
            extract_stft_features(np.array([[1, 2], [3, 4]]), self.fs)  # 2D array

    def test_extract_stft_features_valid(self):
        """Test STFT feature extraction with valid inputs."""
        features = extract_stft_features(self.valid_signal, self.fs)
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)

    def test_extract_stft_features_invalid(self):
        """Test STFT feature extraction with invalid inputs."""
        with self.assertRaises(ValueError):
            extract_stft_features([], self.fs)  # Non-NumPy array
        with self.assertRaises(ValueError):
            extract_stft_features(
                self.valid_signal, -1
            )  # Non-positive sampling frequency
        with self.assertRaises(ValueError):
            extract_stft_features(np.array([[1, 2], [3, 4]]), self.fs)  # Non-1D array

    def test_extract_wavelet_features_valid(self):
        """Test wavelet feature extraction with valid inputs."""
        features = extract_wavelet_features(self.valid_signal, wavelet="db4", level=4)
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)

    def test_extract_wavelet_features_invalid(self):
        """Test wavelet feature extraction with invalid inputs."""
        with self.assertRaises(ValueError):
            extract_wavelet_features([], wavelet="db4", level=4)  # Non-NumPy array
        with self.assertRaises(ValueError):
            extract_wavelet_features(
                self.short_signal, wavelet="db4", level=10
            )  # Signal too short for level

    def test_extract_hybrid_features_valid(self):
        """Test hybrid feature extraction with valid inputs."""
        features = extract_hybrid_features(
            self.valid_signal, self.fs, wavelet="db4", level=4, nperseg=128
        )
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)

    def test_extract_hybrid_features_invalid(self):
        """Test hybrid feature extraction with invalid inputs."""
        with self.assertRaises(ValueError):
            extract_hybrid_features([], self.fs)  # Non-NumPy array
        with self.assertRaises(ValueError):
            extract_hybrid_features(
                self.valid_signal, -1
            )  # Non-positive sampling frequency

    def test_extract_wavelet_features(self):
        """Test wavelet feature extraction."""
        # Test with valid input
        features = extract_wavelet_features(self.clean_signal)
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)

        # Check specific features
        self.assertIn("wavelet_energy_0", features)
        self.assertIn("wavelet_entropy_0", features)
        self.assertIn("wavelet_total_energy", features)

        # Test with different wavelet and level
        features_db2 = extract_wavelet_features(
            self.clean_signal, wavelet="db2", level=3
        )
        self.assertIsInstance(features_db2, dict)

        # Test with invalid inputs
        with self.assertRaises(ValueError):
            extract_wavelet_features([])  # Empty signal
        with self.assertRaises(ValueError):
            extract_wavelet_features(self.clean_signal, level=10)  # Too many levels

    def test_extract_hybrid_features(self):
        """Test hybrid feature extraction."""
        # Test with valid input
        features = extract_hybrid_features(self.clean_signal, self.fs)
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)

        # Check that it contains both STFT and wavelet features
        self.assertIn("stft_mean_energy", features)
        self.assertIn("wavelet_energy_0", features)

        # Test with custom frequency bands
        custom_bands = {"very_low": (0, 1), "very_high": (40, 100)}
        features_custom = extract_hybrid_features(
            self.clean_signal, self.fs, freq_bands=custom_bands
        )
        self.assertIn("stft_power_very_low", features_custom)

        # Test with invalid inputs
        with self.assertRaises(ValueError):
            extract_hybrid_features([], self.fs)
        with self.assertRaises(ValueError):
            extract_hybrid_features(self.clean_signal, -1)

    def test_spectral_helper_functions(self):
        """Test spectral analysis helper functions."""
        # Create test spectrogram
        f, t, Zxx = signal.stft(self.clean_signal, fs=self.fs)
        magnitude = np.abs(Zxx)

        # Test spectral centroid
        centroid = _spectral_centroid(f, magnitude)
        self.assertIsInstance(centroid, float)
        self.assertGreaterEqual(centroid, 0)

        # Test spectral bandwidth
        bandwidth = _spectral_bandwidth(f, magnitude)
        self.assertIsInstance(bandwidth, float)
        self.assertGreaterEqual(bandwidth, 0)

        # Test spectral rolloff
        rolloff = _spectral_rolloff(magnitude)
        self.assertIsInstance(rolloff, float)
        self.assertGreaterEqual(rolloff, 0)

        # Test spectral flatness
        flatness = _spectral_flatness(magnitude)
        self.assertIsInstance(flatness, float)
        self.assertGreaterEqual(flatness, 0)
        self.assertLessEqual(flatness, 1)

    def test_wavelet_helper_functions(self):
        """Test wavelet analysis helper functions."""
        # Get wavelet coefficients
        coeffs = pywt.wavedec(self.clean_signal, "db4", level=4)

        # Test zero crossings
        crossings = _count_zero_crossings(coeffs)
        self.assertIsInstance(crossings, int)
        self.assertGreaterEqual(crossings, 0)

        # Test energy ratio
        energy_ratio = _energy_ratio(coeffs)
        self.assertIsInstance(energy_ratio, float)
        self.assertGreaterEqual(energy_ratio, 0)

        # Test with empty coefficients
        self.assertEqual(_energy_ratio([]), 0.0)

    def test_cross_domain_features(self):
        """Test cross-domain feature calculation."""
        # Create sample feature dictionaries
        stft_feats = {
            "stft_total_energy": 1.0,
            "stft_energy_1": 0.5,
            "stft_energy_2": 0.5,
        }
        wavelet_feats = {
            "wavelet_total_energy": 2.0,
            "wavelet_energy_1": 1.0,
            "wavelet_energy_2": 1.0,
        }

        # Test feature calculation
        cross_features = _calculate_cross_domain_features(stft_feats, wavelet_feats)
        self.assertIsInstance(cross_features, dict)
        self.assertIn("stft_to_wavelet_energy_ratio", cross_features)
        self.assertIn("energy_distribution_correlation", cross_features)

        # Test with empty dictionaries
        empty_features = _calculate_cross_domain_features({}, {})
        self.assertEqual(len(empty_features), 0)


if __name__ == "__main__":
    unittest.main()
