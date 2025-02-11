# tests/test_quality.py

import unittest
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from ecg_processor.quality import (
    calculate_signal_quality,
    plot_signal_comparison,
    assess_beat_quality,
    assess_signal_quality,
    _calculate_power_band,
    _calculate_snr,
    _assess_baseline_wander,
    _detect_powerline_interference,
    _calculate_signal_complexity,
    _assess_rpeak_quality,
    _calculate_chunk_metrics,
    _average_chunk_metrics,
    _normalize_metrics,
    _calculate_overall_quality,
)

# Configure matplotlib for non-interactive backend
plt.switch_backend("agg")


class TestSignalQualityAssessment(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.fs = 500
        self.duration = 10
        self.n_samples = int(self.fs * self.duration)

        # Generate clean ECG
        self.clean_ecg = nk.ecg_simulate(
            duration=self.duration, sampling_rate=self.fs, noise=0, heart_rate=60
        )

        # Generate noisy signals
        self.noisy_ecg = self._generate_noisy_ecg()
        self.baseline_wander_ecg = self._add_baseline_wander()
        self.powerline_ecg = self._add_powerline_interference()

        # Get wave delineation for clean ECG
        _, info = nk.ecg_peaks(self.clean_ecg, sampling_rate=self.fs)
        self.clean_waves = {"R_peaks": info["ECG_R_Peaks"]}

        # Generate single beat for beat quality tests
        self.single_beat = self.clean_ecg[0 : int(self.fs)]  # 1 second

        # Create invalid signals
        self.short_signal = np.array([1, 2, 3])
        self.invalid_signal = np.array([np.nan, np.inf, 1, 2, 3])
        self.empty_signal = np.array([])

    def _generate_noisy_ecg(self) -> np.ndarray:
        """Generate ECG with significant noise."""
        noise = 0.5 * np.random.randn(self.n_samples)
        return self.clean_ecg + noise

    def _add_baseline_wander(self) -> np.ndarray:
        """Add significant baseline wander to ECG."""
        t = np.arange(self.n_samples) / self.fs
        baseline = 1.0 * np.sin(2 * np.pi * 0.2 * t)
        return self.clean_ecg + baseline

    def _add_powerline_interference(self) -> np.ndarray:
        """Add significant powerline interference to ECG."""
        t = np.arange(self.n_samples) / self.fs
        powerline = 0.5 * np.sin(2 * np.pi * 50 * t)
        return self.clean_ecg + powerline

    def test_calculate_signal_quality(self):
        """Test calculate_signal_quality function."""
        # Test normal case
        metrics = calculate_signal_quality(self.noisy_ecg, self.clean_ecg, self.fs)
        self.assertIsInstance(metrics, dict)
        self.assertIn("snr", metrics)
        self.assertIn("signal_rms", metrics)

        # Test error cases
        with self.assertRaises(ValueError):
            calculate_signal_quality(self.short_signal, self.clean_ecg, self.fs)
        with self.assertRaises(ValueError):
            calculate_signal_quality(self.invalid_signal, self.clean_ecg, self.fs)
        with self.assertRaises(ValueError):
            calculate_signal_quality(self.noisy_ecg, self.clean_ecg, -1)

    def test_plot_signal_comparison(self):
        """Test plot_signal_comparison function."""
        # Test normal case without metrics
        fig = plot_signal_comparison(
            self.noisy_ecg, self.clean_ecg, self.fs, show_metrics=False
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

        # Test error cases
        with self.assertRaises(ValueError):
            plot_signal_comparison(self.short_signal, self.clean_ecg, self.fs)
        with self.assertRaises(ValueError):
            plot_signal_comparison(self.invalid_signal, self.clean_ecg, self.fs)

    def test_assess_beat_quality(self):
        """Test assess_beat_quality function."""
        # Test normal case with clean beat
        is_good, metrics = assess_beat_quality(self.single_beat, self.fs)
        self.assertIsInstance(is_good, bool)
        self.assertIsInstance(metrics, dict)

        # Test with noisy beat
        noisy_beat = self.single_beat + 0.5 * np.random.randn(len(self.single_beat))
        is_good, metrics = assess_beat_quality(noisy_beat, self.fs)
        self.assertIsInstance(is_good, bool)
        self.assertIsInstance(metrics, dict)

        # Test error cases
        with self.assertRaises(ValueError):
            assess_beat_quality(self.invalid_signal, self.fs)
        with self.assertRaises(ValueError):
            assess_beat_quality(self.single_beat, -1)

    def test_assess_signal_quality(self):
        """Test assess_signal_quality function."""
        # Test normal case with waves
        metrics = assess_signal_quality(self.clean_ecg, self.fs, self.clean_waves)
        self.assertIsInstance(metrics, dict)
        self.assertIn("overall_quality", metrics)
        self.assertTrue(0 <= metrics["overall_quality"] <= 1)

        # Test without waves
        metrics = assess_signal_quality(self.clean_ecg, self.fs)
        self.assertIsInstance(metrics, dict)
        self.assertIn("overall_quality", metrics)
        self.assertTrue(0 <= metrics["overall_quality"] <= 1)

        # Test error cases
        with self.assertRaises(ValueError):
            assess_signal_quality(self.short_signal, self.fs)
        with self.assertRaises(ValueError):
            assess_signal_quality(self.invalid_signal, self.fs)
        with self.assertRaises(ValueError):
            assess_signal_quality(self.clean_ecg, -1)

    def test_calculate_power_band(self):
        """Test _calculate_power_band function."""
        # Create test frequency domain data
        freqs = np.linspace(0, 100, 1000)
        psd = np.ones_like(freqs)

        # Test normal case
        power = _calculate_power_band(freqs, psd, 0, 50)
        self.assertIsInstance(power, float)
        self.assertGreater(power, 0)

        # Test error cases
        with self.assertRaises(ValueError):
            _calculate_power_band(freqs, psd, 50, 0)  # Invalid frequency range

    def test_calculate_snr(self):
        """Test _calculate_snr function."""
        # Test normal case with clean signal
        snr_metrics = _calculate_snr(self.clean_ecg, self.fs)
        self.assertIsInstance(snr_metrics, dict)
        self.assertIn("SNR", snr_metrics)
        self.assertGreater(snr_metrics["SNR"], 0)

        # Test with noisy signal
        snr_noisy = _calculate_snr(self.noisy_ecg, self.fs)
        self.assertIsInstance(snr_noisy, dict)
        self.assertIn("SNR", snr_noisy)
        self.assertGreater(snr_noisy["SNR"], 0)
        self.assertLess(
            snr_noisy["SNR"], snr_metrics["SNR"]
        )  # Noisy signal should have lower SNR

        # Test error cases
        with self.assertRaises(ValueError):
            _calculate_snr(None, self.fs)
        with self.assertRaises(ValueError):
            _calculate_snr(self.clean_ecg, -1)

    def test_assess_baseline_wander(self):
        """Test _assess_baseline_wander function."""
        # Test normal case with clean signal
        wander_metrics = _assess_baseline_wander(self.clean_ecg, self.fs)
        self.assertIsInstance(wander_metrics, dict)
        self.assertIn("baseline_wander_severity", wander_metrics)
        self.assertIn("baseline_stable", wander_metrics)
        self.assertTrue(0 <= wander_metrics["baseline_wander_severity"] <= 1)

        # Test with baseline wander
        wander_metrics_bad = _assess_baseline_wander(self.baseline_wander_ecg, self.fs)
        self.assertIsInstance(wander_metrics_bad, dict)
        self.assertGreater(wander_metrics_bad["baseline_wander_severity"], 0.5)

        # Test error cases
        with self.assertRaises(ValueError):
            _assess_baseline_wander(None, self.fs)
        with self.assertRaises(ValueError):
            _assess_baseline_wander(self.clean_ecg, -1)

    def test_detect_powerline_interference(self):
        """Test _detect_powerline_interference function."""
        # Test normal case with clean signal
        interference_metrics = _detect_powerline_interference(self.clean_ecg, self.fs)
        self.assertIsInstance(interference_metrics, dict)
        self.assertIn("powerline_interference_ratio", interference_metrics)
        self.assertIn("powerline_interference_present", interference_metrics)
        self.assertTrue(0 <= interference_metrics["powerline_interference_ratio"] <= 1)

        # Test with powerline interference
        interference_metrics_bad = _detect_powerline_interference(
            self.powerline_ecg, self.fs
        )
        self.assertIsInstance(interference_metrics_bad, dict)
        self.assertGreater(
            interference_metrics_bad["powerline_interference_ratio"],
            interference_metrics["powerline_interference_ratio"],
        )

        # Test error cases
        with self.assertRaises(ValueError):
            _detect_powerline_interference(self.short_signal, self.fs)
        with self.assertRaises(ValueError):
            _detect_powerline_interference(self.invalid_signal, self.fs)
        with self.assertRaises(ValueError):
            _detect_powerline_interference(self.clean_ecg, -1)

    def test_calculate_signal_complexity(self):
        """Test _calculate_signal_complexity function."""
        # Test normal case with clean signal
        complexity = _calculate_signal_complexity(self.clean_ecg)
        self.assertIsInstance(complexity, dict)
        self.assertIn("signal_complexity", complexity)
        self.assertTrue(0 <= complexity["signal_complexity"] <= 1)

        # Test with noisy signal
        complexity_noisy = _calculate_signal_complexity(self.noisy_ecg)
        self.assertIsInstance(complexity_noisy, dict)
        self.assertIn("signal_complexity", complexity_noisy)
        self.assertTrue(0 <= complexity_noisy["signal_complexity"] <= 1)

        # Test error cases
        with self.assertRaises(ValueError):
            _calculate_signal_complexity(self.short_signal)
        with self.assertRaises(ValueError):
            _calculate_signal_complexity(self.invalid_signal)

    def test_assess_rpeak_quality(self):
        """Test _assess_rpeak_quality function."""
        # Test normal case with clean signal
        quality_metrics = _assess_rpeak_quality(self.clean_ecg, self.clean_waves)
        self.assertIsInstance(quality_metrics, dict)
        self.assertIn("rpeak_detection_quality", quality_metrics)
        self.assertTrue(0 <= quality_metrics["rpeak_detection_quality"] <= 1)

        # Test with noisy signal
        quality_metrics_noisy = _assess_rpeak_quality(self.noisy_ecg, self.clean_waves)
        self.assertIsInstance(quality_metrics_noisy, dict)
        self.assertTrue(0 <= quality_metrics_noisy["rpeak_detection_quality"] <= 1)

        # Test error cases
        with self.assertRaises(ValueError):
            _assess_rpeak_quality(self.short_signal, self.clean_waves)
        with self.assertRaises(ValueError):
            _assess_rpeak_quality(self.invalid_signal, self.clean_waves)

    def test_calculate_chunk_metrics(self):
        """Test _calculate_chunk_metrics function."""
        # Test normal case with clean signal chunk
        metrics = _calculate_chunk_metrics(self.single_beat, self.fs)
        self.assertIsInstance(metrics, dict)
        self.assertTrue(all(np.isfinite(list(metrics.values()))))

        # Test with noisy chunk
        noisy_chunk = self.single_beat + 0.5 * np.random.randn(len(self.single_beat))
        metrics_noisy = _calculate_chunk_metrics(noisy_chunk, self.fs)
        self.assertIsInstance(metrics_noisy, dict)
        self.assertTrue(all(np.isfinite(list(metrics_noisy.values()))))

        # Test error cases
        with self.assertRaises(ValueError):
            _calculate_chunk_metrics(None, self.fs)
        with self.assertRaises(ValueError):
            _calculate_chunk_metrics(self.single_beat, -1)

    def test_average_chunk_metrics(self):
        """Test _average_chunk_metrics function."""
        # Create test metrics
        metrics_list = [
            {"snr": 10.0, "quality": 0.8},
            {"snr": 20.0, "quality": 0.9},
            {"snr": 15.0, "quality": 0.85},
        ]

        # Test normal case
        avg_metrics = _average_chunk_metrics(metrics_list)
        self.assertIsInstance(avg_metrics, dict)
        self.assertAlmostEqual(avg_metrics["snr"], 15.0, places=5)
        self.assertAlmostEqual(avg_metrics["quality"], 0.85, places=5)

        # Test error cases
        with self.assertRaises(ValueError):
            _average_chunk_metrics([])  # Empty list
        with self.assertRaises(ValueError):
            _average_chunk_metrics([{"invalid": np.nan}])  # Invalid metrics

    def test_normalize_metrics(self):
        """Test _normalize_metrics function."""
        # Create test metrics
        metrics = {
            "SNR": 100.0,
            "baseline_wander_severity": 2.5,
            "powerline_interference_ratio": -0.5,
            "signal_complexity": 0.7,
        }

        # Test normal case
        normalized = _normalize_metrics(metrics)
        self.assertIsInstance(normalized, dict)
        for k, v in normalized.items():
            if k == "SNR":
                self.assertLessEqual(v, 40)  # SNR is clipped at 40
            else:
                self.assertGreaterEqual(v, 0)
                self.assertLessEqual(v, 1)

        # Test error cases
        with self.assertRaises(ValueError):
            _normalize_metrics({"invalid": np.nan})
        with self.assertRaises(ValueError):
            _normalize_metrics({"invalid": np.inf})

    def test_calculate_overall_quality(self):
        """Test _calculate_overall_quality function."""
        # Create test metrics
        metrics = {
            "SNR": 20.0,
            "baseline_wander_severity": 0.3,
            "powerline_interference_ratio": 0.1,
            "signal_complexity": 0.4,
            "rpeak_detection_quality": 0.9,
        }

        # Test normal case
        quality_score = _calculate_overall_quality(metrics)
        self.assertIsInstance(quality_score, float)
        self.assertTrue(0 <= quality_score <= 1)

        # Test error cases
        with self.assertRaises(ValueError):
            _calculate_overall_quality({})  # Empty metrics
        with self.assertRaises(ValueError):
            _calculate_overall_quality({"invalid": -1})  # Invalid metric value


if __name__ == "__main__":
    unittest.main(verbosity=2)
