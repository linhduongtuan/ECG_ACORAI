# tests/test_features.py

import unittest
import numpy as np
import neurokit2 as nk
from ecg_processor.features import (
    extract_statistical_features,
    extract_morphological_features,
    analyze_qt_interval,
    detect_t_wave_alternans,
    analyze_st_segment,
    _calculate_k_score,
    _classify_st_shape,
)


class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic ECG data for testing
        self.fs = 500  # sampling rate
        self.duration = 10  # seconds

        # Generate synthetic ECG signal using neurokit2
        self.ecg = nk.ecg_simulate(
            duration=self.duration, sampling_rate=self.fs, noise=0.05
        )

        # Process ECG to get peaks and waves
        signals, info = nk.ecg_process(self.ecg, sampling_rate=self.fs)

        # Get R-peaks and delineate waves
        self.rpeaks = info["ECG_R_Peaks"]
        _, waves_dict = nk.ecg_delineate(
            self.ecg, self.rpeaks, sampling_rate=self.fs, method="peak"
        )

        # Create waves dictionary with proper structure and ensure integer indices
        self.waves = {
            "Q_start": int(self.rpeaks[0] - 0.05 * self.fs)
            if len(self.rpeaks) > 0
            else 0,
            "Q_Peak": int(self.rpeaks[0] - 0.025 * self.fs)
            if len(self.rpeaks) > 0
            else 0,
            "R_Peak": int(self.rpeaks[0]) if len(self.rpeaks) > 0 else 0,
            "S_Peak": int(self.rpeaks[0] + 0.025 * self.fs)
            if len(self.rpeaks) > 0
            else 0,
            "T_Peak": int(self.rpeaks[0] + 0.2 * self.fs)
            if len(self.rpeaks) > 0
            else 0,
            "T_end": int(self.rpeaks[0] + 0.4 * self.fs) if len(self.rpeaks) > 0 else 0,
            "P_Peak": int(self.rpeaks[0] - 0.2 * self.fs)
            if len(self.rpeaks) > 0
            else 0,
            "P_start": int(self.rpeaks[0] - 0.3 * self.fs)
            if len(self.rpeaks) > 0
            else 0,
            "P_end": int(self.rpeaks[0] - 0.1 * self.fs) if len(self.rpeaks) > 0 else 0,
            "QRS_onset": int(self.rpeaks[0] - 0.05 * self.fs)
            if len(self.rpeaks) > 0
            else 0,
            "QRS_offset": int(self.rpeaks[0] + 0.05 * self.fs)
            if len(self.rpeaks) > 0
            else 0,
            "T_onset": int(self.rpeaks[0] + 0.15 * self.fs)
            if len(self.rpeaks) > 0
            else 0,
            "T_offset": int(self.rpeaks[0] + 0.4 * self.fs)
            if len(self.rpeaks) > 0
            else 0,
            "J_point": int(self.rpeaks[0] + 0.08 * self.fs)
            if len(self.rpeaks) > 0
            else 0,
        }

    def test_extract_statistical_features(self):
        """Test statistical feature extraction."""
        features = extract_statistical_features(self.ecg)

        expected_features = ["mean", "std", "max", "min", "median", "mad"]
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], float)

    def test_analyze_qt_interval(self):
        """Test QT interval analysis."""
        # Create a sample beat with clear QT interval
        beat_start = self.waves["Q_start"]
        beat_end = self.waves["T_end"]

        # Ensure indices are within bounds
        beat_start = max(0, min(beat_start, len(self.ecg) - 1))
        beat_end = max(beat_start + 1, min(beat_end, len(self.ecg)))

        beat = self.ecg[beat_start:beat_end]

        # Create wave indices relative to the extracted beat
        beat_waves = {
            "Q_start": 0,
            "T_end": len(beat) - 1,
            "R_start": self.waves["R_Peak"] - beat_start,
            "R_end": self.waves["R_Peak"] - beat_start + int(0.1 * self.fs),
        }

        results = analyze_qt_interval(beat, beat_waves, self.fs)

        expected_metrics = ["QT_interval", "Heart_Rate"]
        for metric in expected_metrics:
            self.assertIn(metric, results)
            self.assertIsInstance(results[metric], (float, type(None)))

    def test_detect_t_wave_alternans(self):
        """Test T-wave alternans detection."""
        # Create synthetic T-waves with alternating amplitudes
        t = np.linspace(0, 2 * np.pi, 100)
        even_beats = np.sin(t)
        odd_beats = 1.2 * np.sin(t)  # Slightly larger amplitude

        beats = []
        waves_list = []
        for i in range(4):  # Create 4 alternating beats
            beats.append(even_beats if i % 2 == 0 else odd_beats)
            waves_list.append({"T_start": 0, "T_end": len(t) - 1})

        results = detect_t_wave_alternans(beats, waves_list, self.fs)

        expected_metrics = ["TWA_magnitude", "TWA_ratio", "K_score"]
        for metric in expected_metrics:
            self.assertIn(metric, results)
            self.assertIsInstance(results[metric], float)

    def test_analyze_st_segment(self):
        """Test ST segment analysis."""
        # Create a sample beat with clear ST segment
        beat_start = self.waves["QRS_offset"]
        beat_end = self.waves["T_onset"]

        # Ensure indices are within bounds
        beat_start = max(0, min(beat_start, len(self.ecg) - 1))
        beat_end = max(beat_start + 1, min(beat_end, len(self.ecg)))

        st_segment = self.ecg[beat_start:beat_end]

        # Create wave indices relative to the extracted segment
        segment_waves = {
            "J_point": 0,
            "T_start": len(st_segment) - 1,
            "P_start": 0,
            "P_end": int(0.1 * self.fs),
        }

        results = analyze_st_segment(st_segment, segment_waves, self.fs)

        expected_metrics = ["ST_level", "ST_slope", "ST_integral"]
        for metric in expected_metrics:
            self.assertIn(metric, results)
            self.assertIsInstance(results[metric], (float, type(None)))

    def test_extract_morphological_features(self):
        """Test morphological feature extraction."""
        # Create a clean beat with clear QRS complex
        beat_start = max(0, self.waves["QRS_onset"] - int(0.1 * self.fs))
        beat_end = min(len(self.ecg), self.waves["QRS_offset"] + int(0.1 * self.fs))
        beat = self.ecg[beat_start:beat_end]

        features = extract_morphological_features(beat, self.fs)

        expected_features = ["qrs_duration", "r_amplitude", "s_amplitude"]
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (float, type(None)))

    def test_helper_functions(self):
        """Test helper functions."""
        # Test K-score calculation with proper float arrays
        even_beats = np.array([1.0, 2.0, 3.0], dtype=float)
        odd_beats = np.array([1.5, 2.5, 3.5], dtype=float)
        k_score = _calculate_k_score(even_beats, odd_beats)
        self.assertIsInstance(k_score, float)
        self.assertGreater(k_score, 0)

        # Test with more realistic T-wave shapes
        t = np.linspace(0, 2 * np.pi, 100)
        even_beats = np.sin(t)
        odd_beats = 1.2 * np.sin(t)  # 20% larger amplitude
        k_score = _calculate_k_score(even_beats, odd_beats)
        self.assertIsInstance(k_score, float)
        self.assertGreater(k_score, 0)

        # Test ST shape classification
        st_segment = np.linspace(0, 1, 100)  # Upsloping
        shape = _classify_st_shape(st_segment, 0.1)
        self.assertIn(shape, ["horizontal", "upsloping", "downsloping"])


if __name__ == "__main__":
    unittest.main()
