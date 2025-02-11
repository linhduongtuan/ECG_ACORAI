import unittest
import numpy as np
from ecg_processor.hrv import (
    calculate_advanced_hrv,
    calculate_time_domain_hrv,
    calculate_frequency_domain_hrv,
    calculate_non_linear_hrv,
    calculate_sample_entropy,
    calculate_dfa,
    calculate_poincare_metrics,
)


class TestAdvancedHRV(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        # Create synthetic RR intervals for testing
        np.random.seed(42)  # For reproducibility

        # Normal RR intervals (around 800ms with some variation)
        self.normal_rr = 800 + 50 * np.random.randn(100)

        # Very short sequence
        self.short_rr = np.array([800, 820])

        # Invalid RR intervals
        self.invalid_rr = np.array([-800, 800, 820])

        # RR intervals with extreme values
        self.extreme_rr = np.array([200, 800, 820, 4000])

        # Empty array
        self.empty_rr = np.array([])

        # Standard sampling frequency
        self.fs = 1000.0

    def test_normal_calculation(self):
        """Test HRV calculation with normal RR intervals."""
        try:
            results = calculate_advanced_hrv(self.normal_rr, self.fs)

            # Check if all expected metrics are present
            expected_metrics = {
                "SDNN",
                "RMSSD",
                "pNN50",
                "SDANN",
                "SDNN_index",
                "Mean_HR",
                "STD_HR",
                "VLF_power",
                "LF_power",
                "HF_power",
                "LF_HF_ratio",
                "Total_power",
                "SD1",
                "SD2",
                "SampEn",
                "ApEn",
                "DFA_alpha1",
                "DFA_alpha2",
            }

            for metric in expected_metrics:
                self.assertIn(metric, results)
                self.assertTrue(np.isfinite(results[metric]))

            # Check if values are within physiological ranges
            self.assertGreater(results["SDNN"], 0)
            self.assertGreater(results["RMSSD"], 0)
            self.assertGreaterEqual(results["pNN50"], 0)
            self.assertLessEqual(results["pNN50"], 100)
            self.assertGreater(results["Mean_HR"], 0)
            self.assertLess(results["Mean_HR"], 200)

        except Exception as e:
            self.fail(f"Test failed with exception: {str(e)}")

    def test_time_domain_metrics(self):
        """Test time domain HRV metrics calculation."""
        try:
            results = calculate_time_domain_hrv(self.normal_rr)
            self.assertIsInstance(results, dict)
            self.assertGreater(results["sdnn"], 0)
            self.assertGreater(results["rmssd"], 0)
        except Exception as e:
            self.fail(f"Time domain test failed: {str(e)}")

    def test_frequency_domain_metrics(self):
        """Test frequency domain HRV metrics calculation."""
        try:
            results = calculate_frequency_domain_hrv(self.normal_rr)
            self.assertIsInstance(results, dict)
            self.assertGreaterEqual(results["lf_hf_ratio"], 0)
        except Exception as e:
            self.fail(f"Frequency domain test failed: {str(e)}")

    def test_non_linear_metrics(self):
        """Test non-linear HRV metrics calculation."""
        try:
            results = calculate_non_linear_hrv(self.normal_rr)
            self.assertIsInstance(results, dict)
            self.assertTrue("sd1" in results)
            self.assertTrue("sd2" in results)
        except Exception as e:
            self.fail(f"Non-linear metrics test failed: {str(e)}")

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test empty array
        with self.assertRaises(ValueError):
            calculate_advanced_hrv(self.empty_rr, self.fs)

        # Test negative intervals
        with self.assertRaises(ValueError):
            calculate_advanced_hrv(self.invalid_rr, self.fs)

        # Test invalid sampling frequency
        with self.assertRaises(ValueError):
            calculate_advanced_hrv(self.normal_rr, -1.0)

        # Test NaN values
        nan_rr = np.array([800, np.nan, 820])
        with self.assertRaises(ValueError):
            calculate_advanced_hrv(nan_rr, self.fs)

        # Test infinite values
        inf_rr = np.array([800, np.inf, 820])
        with self.assertRaises(ValueError):
            calculate_advanced_hrv(inf_rr, self.fs)

    def test_calculate_dfa(self):
        """Test Detrended Fluctuation Analysis calculation."""
        # Test normal calculation
        dfa_results = calculate_dfa(self.normal_rr)
        self.assertIn("alpha1", dfa_results)
        self.assertIn("alpha2", dfa_results)

        # Test with invalid scale_min
        with self.assertRaises(ValueError):
            calculate_dfa(self.normal_rr, scale_min=1)

        # Test with too short signal
        with self.assertRaises(ValueError):
            calculate_dfa(self.short_rr)

        # Test with negative values
        with self.assertRaises(ValueError):
            calculate_dfa(self.invalid_rr)

        # Test with empty array
        with self.assertRaises(ValueError):
            calculate_dfa(self.empty_rr)

    def test_entropy_calculations(self):
        """Test entropy calculations."""
        # Test normal calculation
        sampen = calculate_sample_entropy(self.normal_rr)
        self.assertIsInstance(sampen, float)
        self.assertGreater(sampen, 0)

        # Test with too short sequence
        with self.assertRaises(ValueError):
            calculate_sample_entropy(self.short_rr)

        # Test with negative values
        with self.assertRaises(ValueError):
            calculate_sample_entropy(self.invalid_rr)

        # Test with empty array
        with self.assertRaises(ValueError):
            calculate_sample_entropy(self.empty_rr)

    def test_poincare_metrics(self):
        """Test Poincaré plot metrics calculation."""
        # Test normal calculation
        metrics = calculate_poincare_metrics(self.normal_rr)
        required_metrics = ["sd1", "sd2", "sd1_sd2_ratio", "ellipse_area"]
        for metric in required_metrics:
            self.assertIn(metric, metrics)

        # Test with too short sequence
        with self.assertRaises(ValueError):
            calculate_poincare_metrics(self.short_rr)

        # Test with negative values
        with self.assertRaises(ValueError):
            calculate_poincare_metrics(self.invalid_rr)

        # Test with empty array
        with self.assertRaises(ValueError):
            calculate_poincare_metrics(self.empty_rr)

    def test_error_handling(self):
        """Test error handling for various edge cases."""
        # Test with empty array
        with self.assertRaises(ValueError):
            calculate_non_linear_hrv(self.empty_rr)

        # Test with invalid sampling frequency
        with self.assertRaises(ValueError):
            calculate_frequency_domain_hrv(self.normal_rr, fs=-1)

        # Test with negative values
        with self.assertRaises(ValueError):
            calculate_time_domain_hrv(self.invalid_rr)

        # Test with too short sequence
        with self.assertRaises(ValueError):
            calculate_time_domain_hrv(np.array([800]))


def test_edge_cases(self):
    """Test edge cases that should return results with warnings."""
    # Test very short sequence (but valid)
    short_rr = np.array([800, 820, 840])
    with self.assertWarns(Warning):
        results = calculate_advanced_hrv(short_rr, self.fs)
        self.assertTrue(np.isfinite(results["SDNN"]))
        self.assertTrue(np.isnan(results["DFA_alpha1"]))

    # Test extreme values
    with self.assertWarns(Warning):
        results = calculate_advanced_hrv(self.extreme_rr, self.fs)
        self.assertTrue(np.isfinite(results["SDNN"]))

    def test_extreme_values(self):
        """Test handling of extreme RR interval values."""
        with self.assertWarns(Warning):
            results = calculate_advanced_hrv(self.extreme_rr, self.fs)
            # Check if results contain valid metrics
            self.assertTrue(np.isfinite(results["SDNN"]))
            # Some metrics might be NaN for extreme values, which is acceptable
            self.assertTrue("DFA_alpha1" in results)
            self.assertTrue("DFA_alpha2" in results)


def test_time_domain_hrv():
    """Test time domain HRV calculations."""
    # Create synthetic RR intervals (around 60 bpm with some variation)
    rr_intervals = 1000 + np.random.normal(0, 50, 100)  # in milliseconds
    metrics = calculate_time_domain_hrv(rr_intervals)

    # Check essential metrics
    assert "SDNN" in metrics
    assert "RMSSD" in metrics
    assert "pNN50" in metrics

    # Check reasonable values
    assert metrics["SDNN"] > 0
    assert metrics["RMSSD"] > 0
    assert 0 <= metrics["pNN50"] <= 100


def test_frequency_domain_hrv():
    """Test frequency domain HRV calculations."""
    # Create synthetic RR intervals
    rr_intervals = 1000 + 50 * np.sin(2 * np.pi * 0.1 * np.arange(100))
    metrics = calculate_frequency_domain_hrv(rr_intervals)

    # Check essential metrics
    assert "LF_power" in metrics
    assert "HF_power" in metrics
    assert "LF_HF_ratio" in metrics

    # Check reasonable values
    assert metrics["LF_power"] > 0
    assert metrics["HF_power"] > 0
    assert metrics["LF_HF_ratio"] > 0


if __name__ == "__main__":
    unittest.main()
