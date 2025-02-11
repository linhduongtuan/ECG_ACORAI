# tests/test_ecg_processor.py
import unittest
import numpy as np
import neurokit2 as nk
from ecg_processor import ECGPreprocessor, ProcessingError


class TestECGPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across test methods."""
        cls.sampling_rate = 500
        cls.duration = 10
        cls.sample_ecg = nk.ecg_simulate(
            duration=cls.duration, sampling_rate=cls.sampling_rate, noise=0.1
        )
        cls.preprocessor = ECGPreprocessor(sampling_rate=cls.sampling_rate, debug=True)

    def setUp(self):
        """Set up test fixtures before each test method."""
        np.random.seed(42)

    def test_initialization(self):
        """Test ECG preprocessor initialization."""
        self.assertEqual(self.preprocessor.fs, self.sampling_rate)
        self.assertEqual(self.preprocessor.lead_config, "single")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
