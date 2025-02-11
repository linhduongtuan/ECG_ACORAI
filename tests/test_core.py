import unittest
import numpy as np
from unittest.mock import patch

from ecg_processor.core import (
    ValidationTracker,
    ECGPreprocessor,
    RealTimeECGProcessor,
)


class TestValidationTracker(unittest.TestCase):
    """Test cases for ValidationTracker class."""

    def setUp(self):
        self.tracker = ValidationTracker()

    def test_validate_signal_valid(self):
        """Test validation of valid signal."""
        signal = np.array([1.0, 2.0, 3.0])
        self.assertTrue(self.tracker.validate_signal(signal, "test"))
        self.assertEqual(len(self.tracker.errors), 0)

    def test_validate_signal_invalid_type(self):
        """Test validation with invalid signal type."""
        signal = [1.0, 2.0, 3.0]  # List instead of numpy array
        self.assertFalse(self.tracker.validate_signal(signal, "test"))
        self.assertEqual(len(self.tracker.errors), 1)

    def test_validate_signal_2d(self):
        """Test validation with 2D signal."""
        signal = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertFalse(self.tracker.validate_signal(signal, "test"))
        self.assertEqual(len(self.tracker.errors), 1)

    def test_validate_signal_too_short(self):
        """Test validation with signal that's too short."""
        signal = np.array([1.0])
        self.assertFalse(self.tracker.validate_signal(signal, "test"))
        self.assertEqual(len(self.tracker.errors), 1)

    def test_validate_signal_non_finite(self):
        """Test validation with non-finite values."""
        signal = np.array([1.0, np.nan, 3.0])
        self.assertFalse(self.tracker.validate_signal(signal, "test"))
        self.assertEqual(len(self.tracker.errors), 1)

    def test_validate_signal_constant(self):
        """Test validation with constant signal."""
        signal = np.array([1.0, 1.0, 1.0])
        self.assertFalse(self.tracker.validate_signal(signal, "test"))
        self.assertEqual(len(self.tracker.errors), 1)

    def test_add_warning(self):
        """Test adding warnings."""
        self.tracker.add_warning("test", "warning message")
        self.assertEqual(len(self.tracker.warnings), 1)
        self.assertEqual(self.tracker.warnings[0], ("test", "warning message"))

    def test_get_report(self):
        """Test getting validation report."""
        self.tracker.add_warning("test", "warning")
        signal = [1.0, 2.0]  # Invalid type
        self.tracker.validate_signal(signal, "test")

        report = self.tracker.get_report()
        self.assertIn("warnings", report)
        self.assertIn("errors", report)
        self.assertEqual(len(report["warnings"]), 1)
        self.assertEqual(len(report["errors"]), 1)


class TestECGPreprocessor(unittest.TestCase):
    """Test cases for ECGPreprocessor class."""

    def setUp(self):
        self.preprocessor = ECGPreprocessor(sampling_rate=500)
        self.test_signal = np.sin(2 * np.pi * 1.0 * np.arange(1000) / 500)

    def test_init_valid(self):
        """Test initialization with valid parameters."""
        preprocessor = ECGPreprocessor(sampling_rate=500, lead_config="single")
        self.assertEqual(preprocessor.fs, 500)
        self.assertEqual(preprocessor.lead_config, "single")

    def test_init_invalid_sampling_rate(self):
        """Test initialization with invalid sampling rate."""
        with self.assertRaises(ValueError):
            ECGPreprocessor(sampling_rate=-1)

    def test_init_invalid_lead_config(self):
        """Test initialization with invalid lead configuration."""
        with self.assertRaises(ValueError):
            ECGPreprocessor(lead_config="invalid")

    def test_process_signal(self):
        """Test signal processing pipeline."""
        result = self.preprocessor.process_signal(self.test_signal)
        self.assertIn("original_signal", result)
        self.assertIn("processed_signal", result)
        self.assertIn("peaks", result)
        self.assertIn("beats", result)
        self.assertIn("hrv_metrics", result)
        self.assertIn("beat_features", result)
        self.assertIn("quality_metrics", result)

    def test_apply_filters(self):
        """Test filter application."""
        filtered = self.preprocessor._apply_filters(self.test_signal)
        self.assertEqual(len(filtered), len(self.test_signal))
        self.assertTrue(np.all(np.isfinite(filtered)))

    def test_detect_qrs_peaks(self):
        """Test QRS peak detection."""
        peaks = self.preprocessor._detect_qrs_peaks(self.test_signal)
        self.assertIsInstance(peaks, np.ndarray)
        self.assertTrue(len(peaks) > 0)

    def test_segment_beats(self):
        """Test beat segmentation."""
        peaks = np.array([100, 300, 500])  # Mock peaks
        beats = self.preprocessor._segment_beats(self.test_signal, peaks)
        self.assertIsInstance(beats, list)
        self.assertTrue(all(isinstance(beat, np.ndarray) for beat in beats))

    @patch("joblib.dump")
    def test_train_classifier(self, mock_dump):
        """Test classifier training."""
        beats = [np.random.rand(300) for _ in range(10)]
        labels = np.array([0, 1] * 5)
        success = self.preprocessor.train_classifier(beats, labels)
        self.assertTrue(success)
        self.assertIsNotNone(self.preprocessor.classifier)


class TestRealTimeECGProcessor(unittest.TestCase):
    """Test cases for RealTimeECGProcessor class."""

    def setUp(self):
        """Set up test fixtures."""

        self.processor = RealTimeECGProcessor(
            sampling_rate=250, buffer_size=1000, overlap=200
        )

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.processor.sampling_rate, 250)
        self.assertEqual(self.processor.buffer_size, 1000)
        self.assertEqual(self.processor.overlap, 200)
        self.assertEqual(len(self.processor.signal_buffer), 0)

    def test_update_online_statistics(self):
        """Test online statistics update."""
        initial_mean = self.processor.signal_mean
        initial_std = self.processor.signal_std

        # Add samples with increasing values to ensure statistics change
        test_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in test_values:
            self.processor.update_online_statistics(value)

        # Check that statistics have changed
        self.assertNotEqual(self.processor.signal_mean, initial_mean)
        self.assertNotEqual(self.processor.signal_std, initial_std)

        # Verify the calculated statistics
        expected_mean = np.mean(test_values)
        expected_std = np.std(
            test_values, ddof=1
        )  # ddof=1 for sample standard deviation

        self.assertAlmostEqual(self.processor.signal_mean, expected_mean, places=5)
        self.assertAlmostEqual(self.processor.signal_std, expected_std, places=5)

    def test_process_sample(self):
        """Test single sample processing."""
        # Test with buffer not full
        result = self.processor.process_sample(1.0)
        self.assertEqual(result, {})

        # Fill buffer and test processing
        for _ in range(self.processor.buffer_size):
            result = self.processor.process_sample(np.random.rand())

        self.assertIsInstance(result, dict)

    def test_process_window(self):
        """Test window processing."""
        # Create a synthetic signal
        t = np.linspace(0, 10, 1000)
        signal = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz sine wave

        # Mock the assess_signal_quality function
        def mock_assess_quality(*args, **kwargs):
            return {"overall_quality": 0.9}

        # Replace the actual function with our mock
        self.processor.preprocessor.calculate_signal_quality = mock_assess_quality

        result = self.processor._process_window(signal)

        self.assertIsInstance(result, dict)
        self.assertIn("quality", result)
        self.assertIn("overall_quality", result["quality"])

    def test_generate_alerts(self):
        """Test alert generation."""
        features = {"Heart_Rate": 120, "ST_elevation": True, "TWA_present": False}
        quality = {"overall_quality": 0.5}

        alerts = self.processor._generate_alerts(features, quality)

        self.assertIsInstance(alerts, list)
        self.assertGreater(len(alerts), 0)
        self.assertIn("Tachycardia detected (HR: 120 bpm)", alerts)
        self.assertIn("ST elevation detected", alerts)


if __name__ == "__main__":
    unittest.main()
