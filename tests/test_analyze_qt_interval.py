import unittest
import neurokit2 as nk
from ecg_processor.features import analyze_qt_interval


class TestQTAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        # Generate synthetic ECG data using neurokit2
        self.fs = 500  # sampling rate
        self.duration = 2  # seconds
        self.ecg = nk.ecg_simulate(duration=self.duration, sampling_rate=self.fs)

        # Create dummy waves dictionary with realistic timing
        self.normal_waves = {
            "Q_start": 100,
            "T_end": 300,
            "R_start": 150,
            "R_end": 200,
            "J_point": 220,
        }

        # Create waves dictionary with missing values
        self.incomplete_waves = {
            "Q_start": None,
            "T_end": 300,
            "R_start": 150,
            "R_end": 200,
        }

        # Create waves dictionary with physiologically impossible values
        self.invalid_waves = {
            "Q_start": 300,  # Q after T
            "T_end": 200,
            "R_start": 400,
            "R_end": 350,
        }

    def test_normal_qt_calculation(self):
        """Test QT interval calculation with normal input."""
        # Create more realistic wave timings
        # Normal QT interval is typically 350-440ms at normal heart rates
        # RR interval for normal heart rate (60-100 bpm) is 600-1000ms
        waves = {
            "Q_start": int(0.2 * self.fs),  # 200ms
            "T_end": int(0.6 * self.fs),  # 600ms (giving ~400ms QT interval)
            "R_start": int(0.3 * self.fs),  # 300ms
            "R_end": int(
                1.1 * self.fs
            ),  # 1100ms (giving ~800ms RR interval, or 75 bpm)
            "J_point": int(0.4 * self.fs),  # 400ms
        }

        results = analyze_qt_interval(self.ecg, waves, self.fs)

        # Check if all expected keys are present
        expected_keys = {
            "QT_interval",
            "QTc_Bazett",
            "QTc_Fridericia",
            "QTc_Framingham",
            "JT_interval",
            "JTc",
            "Heart_Rate",
        }
        self.assertEqual(set(results.keys()), expected_keys)

        # Print actual values for debugging
        print("\nActual values:")
        for key, value in results.items():
            print(f"{key}: {value}")

        # Check if values are within physiological ranges
        self.assertTrue(
            300 <= results["QT_interval"] <= 450,
            f"QT interval {results['QT_interval']} not in range 300-450ms",
        )
        self.assertTrue(
            350 <= results["QTc_Bazett"] <= 450,
            f"QTc Bazett {results['QTc_Bazett']} not in range 350-450ms",
        )
        self.assertTrue(
            60 <= results["Heart_Rate"] <= 100,
            f"Heart rate {results['Heart_Rate']} not in range 60-100 bpm",
        )

        # Additional checks for other QTc formulas
        self.assertTrue(
            350 <= results["QTc_Fridericia"] <= 450,
            f"QTc Fridericia {results['QTc_Fridericia']} not in range 350-450ms",
        )
        self.assertTrue(
            350 <= results["QTc_Framingham"] <= 450,
            f"QTc Framingham {results['QTc_Framingham']} not in range 350-450ms",
        )

        # Check JT interval if available
        if results["JT_interval"] is not None:
            self.assertTrue(
                200 <= results["JT_interval"] <= 300,
                f"JT interval {results['JT_interval']} not in range 200-300ms",
            )


def test_invalid_sampling_rate(self):
    """Test behavior with invalid sampling rate."""
    with self.assertRaises(ValueError):
        analyze_qt_interval(self.ecg, self.normal_waves, -1)

    with self.assertRaises(ValueError):
        analyze_qt_interval(self.ecg, self.normal_waves, 0)


def test_different_heart_rates(self):
    """Test QTc calculations at different heart rates."""
    test_cases = [
        {
            "waves": {
                "Q_start": 100,
                "T_end": 300,
                "R_start": 150,
                "R_end": 350,  # 60 bpm
                "J_point": 220,
            },
            "expected_hr": 60,
        },
        {
            "waves": {
                "Q_start": 100,
                "T_end": 250,
                "R_start": 150,
                "R_end": 250,  # 100 bpm
                "J_point": 220,
            },
            "expected_hr": 100,
        },
    ]

    for case in test_cases:
        results = analyze_qt_interval(self.ecg, case["waves"], self.fs)
        self.assertIsNotNone(results["QTc_Bazett"])
        self.assertIsNotNone(results["QTc_Fridericia"])
        self.assertIsNotNone(results["QTc_Framingham"])
        self.assertAlmostEqual(
            results["Heart_Rate"], case["expected_hr"], delta=5
        )  # Allow 5 bpm difference

    def test_incomplete_waves(self):
        """Test behavior with missing wave points."""
        results = analyze_qt_interval(self.ecg, self.incomplete_waves, self.fs)
        self.assertEqual(
            results, {}
        )  # Should return empty dict when essential points are missing

    def test_invalid_waves(self):
        """Test behavior with physiologically impossible wave points."""
        results = analyze_qt_interval(self.ecg, self.invalid_waves, self.fs)
        self.assertEqual(results, {})  # Should return empty dict for invalid timing

    def test_different_heart_rates(self):
        """Test QTc calculations at different heart rates."""
        # Test cases with different RR intervals
        test_cases = [
            {"rr": 1.0, "qt": 0.4},  # 60 bpm
            {"rr": 0.6, "qt": 0.35},  # 100 bpm
            {"rr": 1.2, "qt": 0.44},  # 50 bpm
        ]

        for case in test_cases:
            waves = {
                "Q_start": 100,
                "T_end": int(100 + case["qt"] * self.fs),
                "R_start": 150,
                "R_end": int(150 + case["rr"] * self.fs),
                "J_point": 220,
            }

            results = analyze_qt_interval(self.ecg, waves, self.fs)

            # Check if QTc is calculated correctly
            self.assertIsNotNone(results["QTc_Bazett"])
            self.assertIsNotNone(results["QTc_Fridericia"])
            self.assertIsNotNone(results["QTc_Framingham"])

    def test_extreme_values(self):
        """Test behavior with extreme but valid values."""
        extreme_waves = {
            "Q_start": 50,
            "T_end": 450,  # Long QT
            "R_start": 100,
            "R_end": 600,  # Very slow heart rate
            "J_point": 150,
        }

        results = analyze_qt_interval(self.ecg, extreme_waves, self.fs)

        # Should still calculate values but might be outside normal ranges
        self.assertIsNotNone(results["QT_interval"])
        self.assertIsNotNone(results["QTc_Bazett"])

    def test_invalid_sampling_rate(self):
        """Test behavior with invalid sampling rate."""
        with self.assertRaises(Exception):
            analyze_qt_interval(self.ecg, self.normal_waves, -1)

    def test_qtc_consistency(self):
        """Test consistency between different QTc formulas."""
        results = analyze_qt_interval(self.ecg, self.normal_waves, self.fs)

        # Bazett's should be larger than Fridericia's at high heart rates
        if results["Heart_Rate"] > 60:
            self.assertGreater(results["QTc_Bazett"], results["QTc_Fridericia"])


def run_qt_tests():
    """Run all QT interval analysis tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestQTAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == "__main__":
    run_qt_tests()
