import unittest
import numpy as np
from unittest.mock import patch
from ecg_processor.realtime import RealTimeECGProcessor
from ecg_processor.exceptions import ProcessingError


class TestRealTimeECGProcessor(unittest.TestCase):
    def setUp(self):
        self.sampling_rate = 250
        self.buffer_size = 2000
        self.overlap = 500
        self.processor = RealTimeECGProcessor(
            sampling_rate=self.sampling_rate,
            buffer_size=self.buffer_size,
            overlap=self.overlap,
        )

    def test_initialization(self):
        self.assertEqual(self.processor.sampling_rate, self.sampling_rate)
        self.assertEqual(self.processor.buffer_size, self.buffer_size)
        self.assertEqual(self.processor.overlap, self.overlap)
        self.assertEqual(len(self.processor.signal_buffer), 0)
        self.assertEqual(len(self.processor.feature_buffer), 0)
        self.assertEqual(len(self.processor.quality_buffer), 0)
        self.assertIsNotNone(self.processor.preprocessor)
        self.assertEqual(self.processor.quality_threshold, 0.6)

    def test_process_sample_buffering(self):
        sample = 1.0
        # Fill up to buffer_size - 1 samples (no processing yet)
        for i in range(self.buffer_size - 1):
            result = self.processor.process_sample(sample)
            self.assertEqual(len(self.processor.signal_buffer), i + 1)
            self.assertEqual(result, {})
        # Next call should trigger processing
        self.processor.process_sample(sample)
        self.assertEqual(len(self.processor.signal_buffer), self.overlap)

    @patch("ecg_processor.ecg_preprocessor.ECGPreprocessor.process_signal")
    def test_process_sample_processing(self, mock_process_signal):
        # Setup mock to return a sample result
        mock_result = {"processed_signal": np.array([1, 2, 3])}
        mock_process_signal.return_value = mock_result

        # Fill buffer with exactly buffer_size - 1 samples
        for _ in range(self.buffer_size - 1):
            self.processor.process_sample(1.0)

        # Call one more time to trigger processing
        result = self.processor.process_sample(1.0)
        self.assertIn("processed_signal", result)
        mock_process_signal.assert_called_once()

    def test_process_sample_overlap(self):
        # Fill buffer with exactly buffer_size - 1 samples
        for _ in range(self.buffer_size - 1):
            self.processor.process_sample(1.0)

        # Trigger processing with one more sample
        self.processor.process_sample(1.0)
        self.assertEqual(len(self.processor.signal_buffer), self.overlap)

    def test_process_sample_error_handling(self):
        with patch(
            "ecg_processor.ecg_preprocessor.ECGPreprocessor.process_signal",
            side_effect=Exception("Test error"),
        ):
            for _ in range(self.buffer_size - 1):
                self.processor.process_sample(1.0)
            with self.assertRaises(ProcessingError) as context:
                self.processor.process_sample(1.0)
            self.assertEqual(
                str(context.exception), "Error processing sample: Test error"
            )


if __name__ == "__main__":
    unittest.main()
