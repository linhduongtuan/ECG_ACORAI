# test_ecg_preprocessor.py

import unittest
import numpy as np
from unittest.mock import patch

from ecg_processor.ecg_preprocessor import ECGPreprocessor


class TestECGPreprocessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 500
        self.preprocessor = ECGPreprocessor(sampling_rate=self.sampling_rate)

        # Create a more realistic synthetic ECG signal
        duration = 10  # seconds
        t = np.linspace(0, duration, int(duration * self.sampling_rate))

        # Create synthetic ECG with clear QRS complexes
        self.test_signal = np.zeros_like(t)
        for i in range(10):  # Create 10 beats
            peak_loc = int((i + 0.5) * self.sampling_rate)  # One beat per second
            if peak_loc < len(self.test_signal):
                # Create QRS complex
                qrs_width = int(0.1 * self.sampling_rate)  # 100ms QRS width
                t_local = np.linspace(-np.pi / 2, np.pi / 2, qrs_width)
                self.test_signal[
                    peak_loc - qrs_width // 2 : peak_loc + qrs_width // 2
                ] = np.sin(t_local)

    def test_initialization(self):
        """Test initialization of ECGPreprocessor."""
        self.assertEqual(self.preprocessor.fs, self.sampling_rate)
        self.assertEqual(self.preprocessor.lead_config, "single")
        self.assertIsNotNone(self.preprocessor.bp_b)
        self.assertIsNotNone(self.preprocessor.bp_a)
        self.assertIsNotNone(self.preprocessor.notch_b)
        self.assertIsNotNone(self.preprocessor.notch_a)

    def test_invalid_initialization(self):
        """Test initialization with invalid parameters."""
        with self.assertRaises(ValueError):
            ECGPreprocessor(sampling_rate=-1)
        with self.assertRaises(ValueError):
            ECGPreprocessor(lead_config="invalid")

    @patch("ecg_processor.ecg_data_loader.ECGDataLoader.load_data")
    def test_load_and_process(self, mock_load_data):
        """Test loading and processing ECG data from file."""
        # Mock the data loader
        mock_load_data.return_value = {
            "signal": self.test_signal,
            "metadata": {"test": "metadata"},
            "annotations": {"test": "annotations"},
            "sampling_rate": self.sampling_rate,
        }

        result = self.preprocessor.load_and_process("test.dat")

        self.assertIn("original_signal", result)
        self.assertIn("processed_signal", result)
        self.assertIn("metadata", result)
        self.assertIn("annotations", result)
        mock_load_data.assert_called_once_with("test.dat")

    def test_process_signal_basic(self):
        """Test basic signal processing functionality."""
        result = self.preprocessor.process_signal(self.test_signal)

        self.assertIn("original_signal", result)
        self.assertIn("processed_signal", result)
        self.assertIn("peaks", result)
        self.assertIn("beats", result)
        self.assertIn("hrv_metrics", result)
        self.assertIn("beat_features", result)
        self.assertIn("quality_metrics", result)

    def test_process_signal_invalid_input(self):
        """Test processing with invalid input signals."""
        with self.assertRaises(ValueError):
            self.preprocessor.process_signal([1, 2, 3])  # Not numpy array

        with self.assertRaises(ValueError):
            self.preprocessor.process_signal(np.array([]))  # Empty array

        with self.assertRaises(ValueError):
            self.preprocessor.process_signal(np.array([1, np.nan, 3]))  # Contains NaN

    def test_apply_filters(self):
        """Test filter application."""
        filtered_signal = self.preprocessor._apply_filters(self.test_signal)
        self.assertEqual(len(filtered_signal), len(self.test_signal))
        self.assertTrue(np.all(np.isfinite(filtered_signal)))

    def test_apply_denoising(self):
        """Test different denoising methods."""
        # Test wavelet denoising
        denoised = self.preprocessor._apply_denoising(
            self.test_signal, method="wavelet"
        )
        self.assertEqual(len(denoised), len(self.test_signal))

        # Test invalid method
        with self.assertRaises(ValueError):
            self.preprocessor._apply_denoising(self.test_signal, method="invalid")

    def test_detect_qrs_peaks(self):
        """Test QRS peak detection."""
        # Create a more realistic ECG signal for testing
        duration = 10  # seconds
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        # Create synthetic ECG with clear R peaks
        signal = np.zeros_like(t)
        for i in range(10):  # Create 10 peaks
            peak_loc = int((i + 0.5) * self.sampling_rate)  # One peak every second
            if peak_loc < len(signal):
                signal[peak_loc - 10 : peak_loc + 10] = np.sin(
                    np.linspace(-np.pi / 2, np.pi / 2, 20)
                )

        try:
            peaks = self.preprocessor._detect_qrs_peaks(signal)
            self.assertIsInstance(peaks, np.ndarray)
            self.assertTrue(len(peaks) > 0)
        except Exception as e:
            self.fail(f"QRS detection failed: {str(e)}")

    def test_segment_beats(self):
        """Test beat segmentation."""
        # Create sample peaks
        peaks = np.array([100, 300, 500])
        beats = self.preprocessor._segment_beats(self.test_signal, peaks)

        self.assertIsInstance(beats, list)
        self.assertTrue(all(isinstance(beat, np.ndarray) for beat in beats))

    def test_calculate_hrv(self):
        """Test HRV calculation."""
        # Create sample peaks
        peaks = np.array([100, 300, 500, 700, 900])
        hrv_metrics = self.preprocessor._calculate_hrv(peaks)

        self.assertIsInstance(hrv_metrics, dict)
        expected_keys = {"mean_hr", "sdnn", "rmssd", "pnn50"}
        for key in expected_keys:
            self.assertIn(key, hrv_metrics)

    def test_extract_features(self):
        """Test feature extraction."""
        # Create sample beats with sufficient length
        beats = []
        for _ in range(3):
            # Create a synthetic beat with proper length (at least 128 samples)
            t = np.linspace(0, 1, 500)  # 500 samples per beat
            beat = np.sin(2 * np.pi * 2 * t) * np.exp(-2 * t)
            beats.append(beat)

        features = self.preprocessor._extract_features(beats)

        self.assertIsInstance(features, dict)
        self.assertTrue(len(features) > 0)
        # Check if features were extracted for each beat
        for i in range(len(beats)):
            self.assertIn(f"beat_{i}", features)
            self.assertTrue(len(features[f"beat_{i}"]) > 0)

    def test_classifier_workflow(self):
        """Test the complete classifier workflow."""
        # Create sample beats with sufficient length
        beats = []
        n_beats = 10
        for _ in range(n_beats):
            t = np.linspace(0, 1, 500)  # 500 samples per beat
            beat = np.sin(2 * np.pi * 2 * t) * np.exp(-2 * t)
            beats.append(beat)

        # Create binary labels
        labels = np.random.randint(0, 2, size=n_beats)

        # Test training
        success = self.preprocessor.train_classifier(beats, labels)
        self.assertTrue(success)

        # Test classification
        results = self.preprocessor.classify_beats(beats)
        self.assertIsNotNone(results)
        self.assertIn("classifications", results)
        self.assertIn("probabilities", results)
        self.assertEqual(len(results["classifications"]), n_beats)


if __name__ == "__main__":
    unittest.main()
