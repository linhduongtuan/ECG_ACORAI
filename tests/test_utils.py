# test_utils.py
import unittest
import numpy as np
import os
import tempfile
from ecg_processor.utils import (
    create_bandpass_filter,
    create_notch_filter,
    normalize_signal,
    load_data,
    resample_signal,
    segment_signal,
    detect_outliers,
    interpolate_missing,
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create sample signals for testing
        self.fs = 250.0  # Hz
        t = np.linspace(0, 1, int(self.fs))
        self.clean_signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
        self.noisy_signal = self.clean_signal + 0.1 * np.random.randn(len(t))

    def test_create_bandpass_filter(self):
        """Test bandpass filter creation."""
        # Test valid parameters
        b, a = create_bandpass_filter(0.5, 40.0, self.fs)
        self.assertIsInstance(b, np.ndarray)
        self.assertIsInstance(a, np.ndarray)

        # Test invalid parameters
        with self.assertRaises(ValueError):
            create_bandpass_filter(-1, 40.0, self.fs)  # negative lowcut
        with self.assertRaises(ValueError):
            create_bandpass_filter(0.5, 200.0, self.fs)  # highcut > nyquist
        with self.assertRaises(ValueError):
            create_bandpass_filter(40.0, 0.5, self.fs)  # lowcut > highcut

    def test_create_notch_filter(self):
        """Test notch filter creation."""
        # Test valid parameters
        b, a = create_notch_filter(50.0, 30.0, self.fs)
        self.assertIsInstance(b, np.ndarray)
        self.assertIsInstance(a, np.ndarray)

        # Test invalid parameters
        with self.assertRaises(ValueError):
            create_notch_filter(-50.0, 30.0, self.fs)  # negative frequency
        with self.assertRaises(ValueError):
            create_notch_filter(50.0, -30.0, self.fs)  # negative Q
        with self.assertRaises(ValueError):
            create_notch_filter(200.0, 30.0, self.fs)  # freq > nyquist

    def test_normalize_signal(self):
        """Test signal normalization."""
        # Test different normalization methods
        methods = ["minmax", "zscore", "robust", "l2"]
        for method in methods:
            normalized = normalize_signal(self.noisy_signal, method=method)
            self.assertEqual(normalized.shape, self.noisy_signal.shape)
            self.assertTrue(np.isfinite(normalized).all())

        # Test invalid inputs
        with self.assertRaises(ValueError):
            normalize_signal(np.array([]), method="minmax")  # empty array
        with self.assertRaises(ValueError):
            normalize_signal(self.noisy_signal, method="invalid")  # invalid method
        with self.assertRaises(ValueError):
            normalize_signal(np.array([np.nan]))  # contains NaN

    def test_load_data(self):
        """Test data loading functionality."""
        # Test CSV loading
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            np.savetxt(f, self.clean_signal, delimiter=",")

        loaded_data = load_data(f.name)
        np.testing.assert_array_almost_equal(loaded_data, self.clean_signal)
        os.remove(f.name)

        # Test invalid file extension
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False, mode="w") as f:
            f.write("dummy data")
            invalid_file_path = f.name

        with self.assertRaises(ValueError):
            load_data(invalid_file_path)

        os.remove(invalid_file_path)

        # Test nonexistent file
        with self.assertRaises(FileNotFoundError):
            load_data("nonexistent_file.csv")

    def test_resample_signal(self):
        """Test signal resampling."""
        # Test upsampling
        fs_out = 2 * self.fs
        upsampled = resample_signal(self.clean_signal, self.fs, fs_out)
        self.assertEqual(len(upsampled), 2 * len(self.clean_signal))

        # Test downsampling
        fs_out = self.fs / 2
        downsampled = resample_signal(self.clean_signal, self.fs, fs_out)
        self.assertEqual(len(downsampled), len(self.clean_signal) // 2)

        # Test invalid inputs
        with self.assertRaises(ValueError):
            resample_signal(self.clean_signal, -1, self.fs)  # negative input fs
        with self.assertRaises(ValueError):
            resample_signal(self.clean_signal, self.fs, -1)  # negative output fs

    def test_segment_signal(self):
        """Test signal segmentation."""
        segment_length = 50
        overlap = 0.5
        segments = segment_signal(self.clean_signal, segment_length, overlap)

        # Check segment dimensions
        self.assertEqual(segments.shape[1], segment_length)

        # Test invalid inputs
        with self.assertRaises(ValueError):
            segment_signal(self.clean_signal, -1)  # negative segment length
        with self.assertRaises(ValueError):
            segment_signal(self.clean_signal, 50, 1.5)  # invalid overlap

    def test_detect_outliers(self):
        """Test outlier detection."""
        # Create signal with known outliers
        signal = np.zeros(100)
        signal[50] = 10  # Add outlier

        # Test different methods
        methods = ["zscore", "iqr", "mad"]
        for method in methods:
            outliers = detect_outliers(signal, threshold=3.0, method=method)
            self.assertTrue(outliers[50])  # Known outlier should be detected
            self.assertFalse(outliers[0])  # Non-outlier should not be detected

        # Test invalid inputs
        with self.assertRaises(ValueError):
            detect_outliers(signal, threshold=-1)  # negative threshold
        with self.assertRaises(ValueError):
            detect_outliers(signal, method="invalid")  # invalid method

    def test_interpolate_missing(self):
        """Test interpolation of missing values."""
        # Create signal with missing values
        signal = np.linspace(0, 10, 100)
        mask = np.zeros_like(signal, dtype=bool)
        mask[40:60] = True  # Mark values as missing

        # Test different interpolation methods
        methods = ["linear", "cubic", "nearest"]
        for method in methods:
            interpolated = interpolate_missing(signal, mask, method=method)
            self.assertEqual(len(interpolated), len(signal))
            self.assertTrue(np.all(np.isfinite(interpolated)))

        # Test invalid inputs
        with self.assertRaises(ValueError):
            interpolate_missing(signal, mask[:-1])  # mismatched shapes
        with self.assertRaises(ValueError):
            interpolate_missing(signal, mask.astype(float))  # non-boolean mask


if __name__ == "__main__":
    unittest.main()
