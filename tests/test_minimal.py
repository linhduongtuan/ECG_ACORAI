# tests/test_minimal.py
import unittest
import numpy as np
import neurokit2 as nk
from ecg_processor import ECGPreprocessor


class TestMinimal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across test methods."""
        cls.sampling_rate = 500
        cls.duration = 10
        # Generate a realistic ECG signal using neurokit2
        cls.test_signal = nk.ecg_simulate(
            duration=cls.duration,
            sampling_rate=cls.sampling_rate,
            heart_rate=70,
            noise=0.01,
        )
        cls.processor = ECGPreprocessor(sampling_rate=cls.sampling_rate)

    def test_import(self):
        """Test basic initialization of ECGPreprocessor."""
        processor = ECGPreprocessor(sampling_rate=500)
        self.assertEqual(processor.fs, 500)
        self.assertEqual(processor.lead_config, "single")

    def test_basic_processing(self):
        """Test basic signal processing functionality."""
        try:
            results = self.processor.process_signal(self.test_signal)

            # Check if all expected keys are present in the results
            expected_keys = {
                "original_signal",
                "processed_signal",
                "peaks",
                "beats",
                "hrv_metrics",
                "beat_features",
                "quality_metrics",
            }
            self.assertEqual(set(results.keys()), expected_keys)

            # Check signal lengths
            self.assertEqual(len(results["original_signal"]), len(self.test_signal))
            self.assertEqual(len(results["processed_signal"]), len(self.test_signal))

            # Check if peaks were detected
            self.assertGreater(len(results["peaks"]), 0)

            # Check if beats were segmented
            self.assertGreater(len(results["beats"]), 0)

            # Check if HRV metrics were calculated
            self.assertIsInstance(results["hrv_metrics"], dict)
            self.assertGreater(len(results["hrv_metrics"]), 0)

            # Check if quality metrics were calculated
            self.assertIsInstance(results["quality_metrics"], dict)
            self.assertGreater(len(results["quality_metrics"]), 0)

        except Exception as e:
            self.fail(f"Processing failed with error: {str(e)}")

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test empty signal
        with self.assertRaises(ValueError):
            self.processor.process_signal(np.array([]))

        # Test signal with NaN values
        with self.assertRaises(ValueError):
            self.processor.process_signal(np.array([1.0, np.nan, 3.0]))

        # Test signal with wrong type
        with self.assertRaises(ValueError):
            self.processor.process_signal([1, 2, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
