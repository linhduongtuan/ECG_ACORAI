# tests/base.py
import unittest
import numpy as np
import neurokit2 as nk
from ecg_processor import ECGPreprocessor, RealTimeECGProcessor


class BaseECGTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across test methods."""
        cls.sampling_rate = 500
        cls.duration = 10
        cls.sample_ecg = nk.ecg_simulate(
            duration=cls.duration, sampling_rate=cls.sampling_rate, noise=0.1
        )
        cls.preprocessor = ECGPreprocessor(sampling_rate=cls.sampling_rate, debug=True)
        cls.realtime_processor = RealTimeECGProcessor(sampling_rate=cls.sampling_rate)

    def setUp(self):
        """Set up test fixtures before each test method."""
        np.random.seed(42)  # Ensure reproducibility
