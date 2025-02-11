# test_advanced_denoising.py

import unittest
import numpy as np
from ecg_processor.advanced_denoising import (
    validate_signal,
    wavelet_denoise,
    adaptive_lms_filter,
    emd_denoise,
    median_filter_signal,
    smooth_signal,
    remove_respiratory_noise,
    remove_emg_noise,
    remove_eda_noise,
    advanced_denoise_pipeline,
)


class TestAdvancedDenoising(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample signals for testing
        self.fs = 500  # Sampling frequency (Hz)
        self.t = np.linspace(0, 1, self.fs, endpoint=False)
        self.clean_signal = np.sin(2 * np.pi * 5 * self.t)  # 5 Hz sine wave
        self.noise = np.random.normal(0, 0.2, size=self.clean_signal.shape)
        self.noisy_signal = self.clean_signal + self.noise

    def test_validate_signal(self):
        """Test signal validation function."""
        # Test valid signal
        valid_signal = np.array([1.0, 2.0, 3.0])
        validate_signal(valid_signal)  # Should not raise any exception

        # Test invalid signals
        with self.assertRaises(TypeError):
            validate_signal([1, 2, 3])  # List instead of numpy array

        with self.assertRaises(ValueError):
            validate_signal(np.array([[1, 2], [3, 4]]))  # 2D array

        with self.assertRaises(ValueError):
            validate_signal(np.array([]))  # Empty array

        with self.assertRaises(ValueError):
            validate_signal(np.array([np.nan, 1, 2]))  # Contains NaN

    def test_wavelet_denoise(self):
        """Test wavelet denoising function."""
        denoised = wavelet_denoise(self.noisy_signal)

        # Check output properties
        self.assertEqual(denoised.shape, self.noisy_signal.shape)
        self.assertTrue(np.all(np.isfinite(denoised)))

        # Check if denoising reduced noise
        original_noise = np.std(self.noisy_signal - self.clean_signal)
        denoised_noise = np.std(denoised - self.clean_signal)
        self.assertLess(denoised_noise, original_noise)

    def test_adaptive_lms_filter(self):
        """Test adaptive LMS filter."""
        noise_reference = np.random.normal(0, 0.2, size=self.noisy_signal.shape)
        denoised = adaptive_lms_filter(self.noisy_signal, noise_reference)

        self.assertEqual(denoised.shape, self.noisy_signal.shape)
        self.assertTrue(np.all(np.isfinite(denoised)))

    def test_emd_denoise(self):
        """Test EMD denoising."""
        denoised = emd_denoise(self.noisy_signal)

        self.assertEqual(denoised.shape, self.noisy_signal.shape)
        self.assertTrue(np.all(np.isfinite(denoised)))

    def test_median_filter_signal(self):
        """Test median filtering."""
        # Test with different kernel sizes
        kernel_sizes = [3, 5, 7]
        for kernel_size in kernel_sizes:
            denoised = median_filter_signal(self.noisy_signal, kernel_size)
            self.assertEqual(denoised.shape, self.noisy_signal.shape)
            self.assertTrue(np.all(np.isfinite(denoised)))

        # Test invalid kernel size
        with self.assertRaises(ValueError):
            median_filter_signal(self.noisy_signal, -1)

    def test_smooth_signal(self):
        """Test Savitzky-Golay smoothing."""
        denoised = smooth_signal(self.noisy_signal)

        self.assertEqual(denoised.shape, self.noisy_signal.shape)
        self.assertTrue(np.all(np.isfinite(denoised)))

        # Test invalid parameters
        with self.assertRaises(ValueError):
            smooth_signal(self.noisy_signal, window_length=-1)
        with self.assertRaises(ValueError):
            smooth_signal(self.noisy_signal, polyorder=-1)

    def test_remove_respiratory_noise(self):
        """Test respiratory noise removal."""
        denoised = remove_respiratory_noise(self.noisy_signal, self.fs)

        self.assertEqual(denoised.shape, self.noisy_signal.shape)
        self.assertTrue(np.all(np.isfinite(denoised)))

    def test_remove_emg_noise(self):
        """Test EMG noise removal."""
        denoised = remove_emg_noise(self.noisy_signal, self.fs)

        self.assertEqual(denoised.shape, self.noisy_signal.shape)
        self.assertTrue(np.all(np.isfinite(denoised)))

    def test_remove_eda_noise(self):
        """Test EDA noise removal."""
        denoised = remove_eda_noise(self.noisy_signal, self.fs)

        self.assertEqual(denoised.shape, self.noisy_signal.shape)
        self.assertTrue(np.all(np.isfinite(denoised)))

    def test_advanced_denoise_pipeline(self):
        """Test the complete denoising pipeline."""
        result = advanced_denoise_pipeline(self.noisy_signal, self.fs)

        # Check if all expected keys are present
        expected_keys = [
            "original",
            "denoised",
            "resp_removed",
            "emg_removed",
            "eda_removed",
        ]
        for key in expected_keys:
            self.assertIn(key, result)
            self.assertEqual(result[key].shape, self.noisy_signal.shape)
            self.assertTrue(np.all(np.isfinite(result[key])))

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        try:
            # Test with very short signal
            short_signal = np.array([1.0])  # Single value signal
            with self.assertRaises(ValueError):
                wavelet_denoise(short_signal)

            # Test with non-numpy array input
            with self.assertRaises(TypeError):
                wavelet_denoise([1, 2, 3])  # List instead of numpy array

            # Test with 2D array input
            with self.assertRaises(ValueError):
                wavelet_denoise(np.array([[1, 2], [3, 4]]))

            # Test with NaN values
            invalid_signal = np.array([1.0, np.nan, 3.0])
            with self.assertRaises(ValueError):
                wavelet_denoise(invalid_signal)

            # Test with infinite values
            invalid_signal = np.array([1.0, np.inf, 3.0])
            with self.assertRaises(ValueError):
                wavelet_denoise(invalid_signal)

            # Test median filter with invalid kernel size
            with self.assertRaises(ValueError):
                median_filter_signal(self.noisy_signal, kernel_size=-1)

            # Test smooth signal with invalid window length
            with self.assertRaises(ValueError):
                smooth_signal(self.noisy_signal, window_length=-1)

            # Test smooth signal with invalid polynomial order
            with self.assertRaises(ValueError):
                smooth_signal(self.noisy_signal, polyorder=-1)

            # Test with constant signal
            constant_signal = np.ones(100)
            denoised = wavelet_denoise(constant_signal)
            self.assertTrue(np.allclose(denoised, constant_signal, atol=1e-10))

        except Exception as e:
            self.fail(f"Test failed with unexpected error: {str(e)}")


if __name__ == "__main__":
    unittest.main()
