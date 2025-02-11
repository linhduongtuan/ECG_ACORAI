# tests/test_preprocessor.py
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ecg_processor import ECGPreprocessor
from ecg_processor.exceptions import ProcessingError
from .conftest import create_sample_ecg


class TestECGPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across test methods."""
        cls.sampling_rate = 500
        cls.duration = 10
        cls.sample_ecg = create_sample_ecg(
            sampling_rate=cls.sampling_rate, duration=cls.duration
        )
        cls.preprocessor = ECGPreprocessor(sampling_rate=cls.sampling_rate, debug=True)

    def setUp(self):
        """Set up test fixtures before each test method."""
        np.random.seed(42)

    def test_preprocessor_initialization(self):
        """Test ECG preprocessor initialization."""
        processor = ECGPreprocessor(sampling_rate=500)
        self.assertEqual(processor.fs, 500)
        self.assertEqual(processor.lead_config, "single")

        with self.assertRaises(ValueError):
            ECGPreprocessor(sampling_rate=-1)

        with self.assertRaises(ValueError):
            ECGPreprocessor(sampling_rate=500, lead_config="invalid")

    def test_signal_processing(self):
        """Test basic signal processing pipeline."""
        results = self.preprocessor.process_signal(self.sample_ecg)

        expected_keys = [
            "original_signal",
            "processed_signal",
            "peaks",
            "beats",
            "hrv_metrics",
            "beat_features",
            "quality_metrics",
            "processing_report",
        ]
        for key in expected_keys:
            self.assertIn(key, results)

        self.assertEqual(len(results["original_signal"]), len(self.sample_ecg))
        self.assertEqual(len(results["processed_signal"]), len(self.sample_ecg))
        self.assertGreater(len(results["peaks"]), 0)
        self.assertGreater(len(results["beats"]), 0)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        with self.assertRaises(ProcessingError):
            self.preprocessor.process_signal(np.array([]))

        with self.assertRaises(ProcessingError):
            self.preprocessor.process_signal(np.array([1.0, np.nan, 3.0]))

        with self.assertRaises(ProcessingError):
            self.preprocessor.process_signal([1, 2, 3])

    def test_classifier_training(self):
        """Test classifier training and prediction."""
        results = self.preprocessor.process_signal(self.sample_ecg)
        beats = results["beats"]
        labels = np.random.choice([0, 1], size=len(beats))

        self.assertTrue(self.preprocessor.train_classifier(beats, labels))

        predictions = self.preprocessor.classify_beats(beats)
        self.assertIsNotNone(predictions)
        self.assertIn("classifications", predictions)
        self.assertIn("probabilities", predictions)
        self.assertEqual(len(predictions["classifications"]), len(beats))

    def test_beat_segmentation(self):
        """Test beat segmentation functionality."""
        results = self.preprocessor.process_signal(self.sample_ecg)
        expected_length = int(0.6 * self.preprocessor.fs)

        for beat in results["beats"]:
            self.assertEqual(len(beat), expected_length)
            peak_idx = np.argmax(np.abs(beat))
            self.assertGreaterEqual(peak_idx, 0.15 * self.preprocessor.fs)
            self.assertLessEqual(peak_idx, 0.25 * self.preprocessor.fs)

    def test_feature_extraction(self):
        """Test feature extraction from beats."""
        results = self.preprocessor.process_signal(self.sample_ecg)
        features = results["beat_features"]

        self.assertEqual(len(features), len(results["beats"]))

        first_beat_features = next(iter(features.values()))
        essential_features = ["mean", "std", "skewness", "kurtosis"]
        for feat in essential_features:
            self.assertIn(feat, first_beat_features)


if __name__ == "__main__":
    unittest.main(verbosity=2)
