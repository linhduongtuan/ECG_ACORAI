# core_torch.py

from typing import Dict, List, Optional
import os
import collections
import logging

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from scipy import signal
from scipy.signal import filtfilt
from .config import ECGConfig
from .utils import (
    create_bandpass_filter,
    create_notch_filter,
    normalize_signal,
)
from .features import (
    extract_statistical_features,
    extract_wavelet_features,
    extract_morphological_features,
)
from .hrv import (
    calculate_time_domain_hrv,
    calculate_frequency_domain_hrv,
    calculate_advanced_hrv,
)
from .quality import calculate_signal_quality, assess_signal_quality
import neurokit2 as nk

logger = logging.getLogger(__name__)


# ---------------------------
# Exception and Tracker Classes
# ---------------------------
class ProcessingError(Exception):
    """Custom exception for ECG signal processing errors."""

    pass


class ValidationTracker:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def validate_signal(self, signal_input, stage: str) -> bool:
        """
        Validate the signal at different processing stages.
        Accepts both NumPy arrays and torch.Tensors.
        """
        try:
            # Convert torch.Tensor to numpy array for validation.
            if torch.is_tensor(signal_input):
                signal_np = signal_input.detach().cpu().numpy()
            elif isinstance(signal_input, np.ndarray):
                signal_np = signal_input
            else:
                raise ValueError(
                    f"{stage}: Signal must be a numpy array or torch tensor"
                )

            if signal_np.ndim != 1:
                raise ValueError(f"{stage}: Signal must be 1-dimensional")
            if len(signal_np) < 2:
                raise ValueError(f"{stage}: Signal is too short")
            if not np.isfinite(signal_np).all():
                raise ValueError(f"{stage}: Signal contains non-finite values")
            if np.all(signal_np == signal_np[0]):
                raise ValueError(f"{stage}: Signal is constant")
            return True
        except ValueError as e:
            self.errors.append((stage, str(e)))
            return False

    def add_warning(self, stage: str, message: str) -> None:
        self.warnings.append((stage, message))

    def get_report(self) -> Dict:
        return {"warnings": self.warnings, "errors": self.errors}


# ---------------------------
# Decorator for Debugging/Profiling
# ---------------------------
def debug_and_profile(func):
    def wrapper(self, *args, **kwargs):
        if self.debug:
            logger.debug(f"Entering {func.__name__}")
            result = func(self, *args, **kwargs)
            logger.debug(f"Exiting {func.__name__}")
            return result
        else:
            return func(self, *args, **kwargs)

    return wrapper


# ---------------------------
# ECG Preprocessor using PyTorch for internal data handling
# ---------------------------
class ECGPreprocessor:
    def __init__(
        self,
        sampling_rate: int = ECGConfig.DEFAULT_SAMPLING_RATE,
        lead_config: str = "single",
        debug: bool = False,
        advanced_denoising: bool = False,
        remove_respiratory: bool = False,
        remove_emg: bool = False,
        remove_eda: bool = False,
    ):
        """Initialize the ECG preprocessor with the given parameters."""
        self.debug = debug
        if self.debug:
            logger.setLevel(logging.DEBUG)
        self.tracker = ValidationTracker()
        self.fs = sampling_rate
        self.lead_config = lead_config

        self.scaler = StandardScaler()
        self.classifier = None
        self._feature_cache = {}
        self.advanced_denoising = advanced_denoising
        self.remove_respiratory = remove_respiratory
        self.remove_emg = remove_emg
        self.remove_eda = remove_eda

        # Basic parameter validation
        if not isinstance(sampling_rate, int) or sampling_rate <= 0:
            raise ValueError(
                f"Sampling rate must be positive integer, got {sampling_rate}"
            )
        if lead_config not in ["single", "multi"]:
            raise ValueError(
                f"Invalid lead config '{lead_config}'. Must be 'single' or 'multi'"
            )

        try:
            # Create filter coefficients via our torch-based utils.
            self.bp_b, self.bp_a = create_bandpass_filter(
                ECGConfig.FILTER_LOWCUT,
                ECGConfig.FILTER_HIGHCUT,
                self.fs,
                ECGConfig.FILTER_ORDER,
            )
            self.notch_b, self.notch_a = create_notch_filter(
                ECGConfig.NOTCH_FREQ, ECGConfig.NOTCH_Q, self.fs
            )
            logger.debug("Filters initialized successfully")
            # Save coefficients as numpy arrays for use with filtfilt.
            self.bp_b_np = self.bp_b.detach().cpu().numpy()
            self.bp_a_np = self.bp_a.detach().cpu().numpy()
            self.notch_b_np = self.notch_b.detach().cpu().numpy()
            self.notch_a_np = self.notch_a.detach().cpu().numpy()
        except Exception as e:
            logger.error(f"Filter initialization failed: {str(e)}")
            raise

    @debug_and_profile
    def process_signal(self, signal_input) -> Dict:
        """
        Process the ECG signal with validation, filtering, normalization,
        QRS detection, beat segmentation, HRV and feature extraction.
        """
        try:
            # Ensure signal is a numpy array for processing.
            if torch.is_tensor(signal_input):
                signal_np = signal_input.detach().cpu().numpy()
            else:
                signal_np = signal_input.copy()
            # Validate input signal.
            if not self.tracker.validate_signal(signal_np, "input_validation"):
                raise ProcessingError("Input signal validation failed")
            original_signal = signal_np.copy()

            # Apply filtering.
            signal_filtered = self._apply_filters(signal_np)
            if not self.tracker.validate_signal(signal_filtered, "filtering"):
                raise ProcessingError("Signal filtering failed")

            # Normalize signal using our torch utils.
            signal_norm = normalize_signal(signal_filtered)
            if not self.tracker.validate_signal(signal_norm, "normalization"):
                raise ProcessingError("Signal normalization failed")

            # QRS detection and beat segmentation.
            peaks = self._detect_qrs_peaks(signal_norm)
            if len(peaks) == 0:
                self.tracker.add_warning("qrs_detection", "No QRS peaks detected")
            beats = self._segment_beats(signal_norm, peaks)
            if len(beats) == 0:
                self.tracker.add_warning("beat_segmentation", "No beats segmented")

            # HRV calculation.
            hrv_metrics = self._calculate_hrv(peaks)
            # Feature extraction.
            beat_features = self._extract_features(beats)
            # Signal quality metrics.
            quality_metrics = calculate_signal_quality(
                original_signal, signal_norm, self.fs
            )

            results = {
                "original_signal": original_signal,
                "processed_signal": signal_norm,
                "peaks": peaks,
                "beats": beats,
                "hrv_metrics": hrv_metrics,
                "beat_features": beat_features,
                "quality_metrics": quality_metrics,
                "processing_report": self.tracker.get_report(),
            }
            logger.info("Signal processing completed successfully")
            return results

        except Exception as e:
            logger.error(f"Signal processing failed: {str(e)}")
            raise

    def _apply_filters(self, signal_np: np.ndarray) -> np.ndarray:
        """Apply bandpass and notch filters using filtfilt."""
        try:
            signal_bp = filtfilt(self.bp_b_np, self.bp_a_np, signal_np)
            signal_filt = filtfilt(self.notch_b_np, self.notch_a_np, signal_bp)
            # If advanced denoising is enabled, apply it.
            if self.advanced_denoising:
                from .advanced_denoising import advanced_denoise_pipeline

                denoise_res = advanced_denoise_pipeline(
                    signal_filt,
                    self.fs,
                    remove_resp=self.remove_respiratory,
                    remove_emg=self.remove_emg,
                    remove_eda=self.remove_eda,
                )
                signal_filt = denoise_res["denoised"]
                if self.debug:
                    logger.debug("Advanced denoising applied")
            return signal_filt
        except Exception as e:
            logger.error(f"Error in filter application: {str(e)}")
            raise ProcessingError(f"Filter application failed: {str(e)}")

    def _detect_qrs_peaks(self, signal_np: np.ndarray) -> np.ndarray:
        """
        Detect QRS complexes using a Pan–Tompkins–like algorithm.
        Uses numpy operations.
        """
        # Basic input checks already done.
        try:
            derivative = np.gradient(signal_np)
            squared = derivative**2
            window = int(0.15 * self.fs)
            if window % 2 == 0:
                window += 1
            ma = np.convolve(squared, np.ones(window) / window, mode="same")
            threshold = 0.3 * np.mean(ma)
            from scipy.signal import find_peaks

            peaks, _ = find_peaks(ma, height=threshold, distance=int(0.2 * self.fs))
            return peaks
        except Exception as e:
            logger.error(f"Error in QRS detection: {str(e)}")
            raise

    def _segment_beats(
        self, signal_np: np.ndarray, peaks: np.ndarray
    ) -> List[np.ndarray]:
        """
        Segment the ECG signal into beats (600ms windows: 200ms before and 400ms after each R-peak).
        """
        if len(peaks) == 0:
            logger.warning("No peaks provided for segmentation")
            return []
        if peaks.max() >= len(signal_np):
            raise ValueError("Peak indices exceed signal length")
        try:
            beats = []
            pre_window = int(0.2 * self.fs)
            post_window = int(0.4 * self.fs)
            for peak in peaks:
                start = max(0, peak - pre_window)
                end = min(len(signal_np), peak + post_window)
                if end - start == pre_window + post_window:
                    beat = signal_np[start:end]
                    if np.isfinite(beat).all():
                        beats.append(beat)
                else:
                    logger.debug(f"Skipping incomplete beat at peak {peak}")
            return beats
        except Exception as e:
            logger.error(f"Error in beat segmentation: {str(e)}")
            raise

    def _calculate_hrv(self, peaks: np.ndarray) -> Dict:
        """Calculate HRV metrics using differences between peak indices."""
        if len(peaks) < 2:
            return {}
        rr_intervals = np.diff(peaks) / self.fs * 1000
        time_m = calculate_time_domain_hrv(rr_intervals)
        freq_m = calculate_frequency_domain_hrv(rr_intervals)
        return {**time_m, **freq_m}

    def _extract_features(self, beats: List[np.ndarray]) -> Dict:
        """
        Extract features from each beat (statistical, wavelet, morphological).
        Uses a feature cache keyed by a hash of the beat.
        """
        if not beats:
            logger.warning("No beats provided for feature extraction")
            return {}
        try:
            features = {}
            for i, beat in enumerate(beats):
                beat_hash = hash(beat.tobytes())
                if beat_hash in self._feature_cache:
                    features[f"beat_{i}"] = self._feature_cache[beat_hash]
                    continue
                try:
                    beat_feats = extract_statistical_features(beat)
                    try:
                        wave_feats = extract_wavelet_features(beat)
                        beat_feats.update(wave_feats)
                    except Exception as e:
                        logger.warning(
                            f"Wavelet feature extraction failed for beat {i}: {str(e)}"
                        )
                    try:
                        morph_feats = extract_morphological_features(beat, self.fs)
                        beat_feats.update(morph_feats)
                    except Exception as e:
                        logger.warning(
                            f"Morphological feature extraction failed for beat {i}: {str(e)}"
                        )
                    beat_feats = {k: v for k, v in beat_feats.items() if v is not None}
                    if beat_feats:
                        features[f"beat_{i}"] = beat_feats
                        self._feature_cache[beat_hash] = beat_feats
                except Exception as e:
                    logger.warning(f"Error extracting features for beat {i}: {str(e)}")
            return features
        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            raise

    def train_classifier(self, beats: List[np.ndarray], labels: np.ndarray) -> bool:
        """Train a RandomForest classifier on extracted beat features."""
        try:
            X = self._prepare_features_for_classification(beats)
            if X.size == 0:
                raise ValueError("No valid features for training")
            X = self.scaler.fit_transform(X)
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.classifier.fit(X, labels)
            ECGConfig.ensure_model_dir()
            try:
                joblib.dump(self.classifier, ECGConfig.MODEL_PATH)
                joblib.dump(self.scaler, ECGConfig.SCALER_PATH)
                logger.info(f"Models saved successfully to {ECGConfig.MODEL_DIR}")
            except Exception as e:
                logger.warning(f"Could not save models: {str(e)}")
            return True
        except Exception as e:
            logger.error(f"Error training classifier: {str(e)}")
            return False

    def classify_beats(self, beats: List[np.ndarray]) -> Optional[Dict]:
        """Classify beats using the trained classifier."""
        try:
            if self.classifier is None:
                self.load_classifier()
            X = self._prepare_features_for_classification(beats)
            X = self.scaler.transform(X)
            predictions = self.classifier.predict(X)
            probabilities = self.classifier.predict_proba(X)
            return {
                "classifications": predictions.tolist(),
                "probabilities": probabilities.tolist(),
            }
        except Exception as e:
            logger.error(f"Error classifying beats: {str(e)}")
            return None

    def load_classifier(self) -> None:
        """Load the classifier and scaler from disk."""
        try:
            if not ECGConfig.MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model file not found at {ECGConfig.MODEL_PATH}"
                )
            if not ECGConfig.SCALER_PATH.exists():
                raise FileNotFoundError(
                    f"Scaler file not found at {ECGConfig.SCALER_PATH}"
                )
            self.classifier = joblib.load(ECGConfig.MODEL_PATH)
            self.scaler = joblib.load(ECGConfig.SCALER_PATH)
            if not isinstance(self.classifier, RandomForestClassifier):
                raise ValueError("Loaded model is not a RandomForestClassifier")
            if not isinstance(self.scaler, StandardScaler):
                raise ValueError("Loaded scaler is not a StandardScaler")
        except Exception as e:
            logger.error(f"Error loading classifier: {str(e)}")
            raise

    def _prepare_features_for_classification(
        self, beats: List[np.ndarray]
    ) -> np.ndarray:
        """Prepare a feature matrix from beat features."""
        try:
            features = self._extract_features(beats)
            if not features:
                raise ValueError("No valid features extracted from beats")
            first_beat_features = next(iter(features.values()))
            feature_names = list(first_beat_features.keys())
            X = []
            for beat_features in features.values():
                feature_vector = [
                    beat_features.get(fname, 0.0) for fname in feature_names
                ]
                X.append(feature_vector)
            X = np.array(X)
            if X.size == 0:
                raise ValueError("No valid feature vectors created")
            logger.debug(f"Created feature matrix with shape {X.shape}")
            return X
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise


# ---------------------------
# Real-Time ECG Processor (simplified version)
# ---------------------------
class RealTimeECGProcessor:
    """
    Real-time ECG signal processor with streaming capabilities.
    This version uses online buffering and filtering; most filtering still relies
    on SciPy (with coefficients that could be produced via our torch-based utils).
    """

    def __init__(
        self, sampling_rate: int = 250, buffer_size: int = 2000, overlap: int = 500
    ):
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.overlap = overlap

        self.signal_buffer = collections.deque(maxlen=buffer_size)
        self.feature_buffer = collections.deque(maxlen=100)
        self.quality_buffer = collections.deque(maxlen=100)

        self.initialize_filters()
        self.preprocessor = ECGPreprocessor(
            sampling_rate=sampling_rate, advanced_denoising=True
        )
        self.quality_threshold = 0.6
        self.initialize_feature_extractors()

        # Online standardization statistics
        self.signal_mean = 0
        self.signal_std = 1
        self.alpha = 0.01

    def initialize_filters(self):
        """Initialize online filters (using SciPy for design)."""
        nyq = self.sampling_rate / 2
        self.bp_b, self.bp_a = signal.butter(3, [0.5 / nyq, 40 / nyq], btype="band")
        self.notch_b, self.notch_a = signal.iirnotch(50 / nyq, 30)
        self.bp_state = signal.lfilter_zi(self.bp_b, self.bp_a)
        self.notch_state = signal.lfilter_zi(self.notch_b, self.notch_a)

    def initialize_feature_extractors(self):
        """Initialize R-peak detection via neurokit2."""
        self.r_peak_detector = nk.ecg_peaks
        self.wave_delineator = nk.ecg_delineate

    def update_online_statistics(self, new_value: float):
        """Update online mean and standard deviation using Welford's algorithm."""
        if not hasattr(self, "_n_samples"):
            self._n_samples = 0
        self._n_samples += 1
        delta = new_value - self.signal_mean
        self.signal_mean += delta / self._n_samples
        delta2 = new_value - self.signal_mean
        if self._n_samples > 1:
            self.signal_std = np.sqrt(
                ((self._n_samples - 2) * (self.signal_std**2) + delta * delta2)
                / (self._n_samples - 1)
            )

    def process_sample(self, sample: float) -> Dict:
        """
        Process a single ECG sample in real time.
        Updates online standardization, buffers new sample and processes when full.
        """
        try:
            self.update_online_statistics(sample)
            if self.signal_std > 0:
                sample_normalized = (sample - self.signal_mean) / self.signal_std
            else:
                sample_normalized = sample - self.signal_mean
            self.signal_buffer.append(sample_normalized)

            if len(self.signal_buffer) < self.buffer_size:
                return {}

            signal_array = np.array(list(self.signal_buffer))
            filtered, self.bp_state = signal.lfilter(
                self.bp_b, self.bp_a, signal_array, zi=self.bp_state * signal_array[0]
            )
            filtered, self.notch_state = signal.lfilter(
                self.notch_b, self.notch_a, filtered, zi=self.notch_state * filtered[0]
            )

            result = self._process_window(filtered)
            for _ in range(self.buffer_size - self.overlap):
                self.signal_buffer.popleft()
            return result
        except Exception as e:
            logger.error(f"Error in real-time processing: {str(e)}")
            return {}

    def _process_window(self, signal_win: np.ndarray) -> Dict:
        """Process a window block: signal quality, R-peak detection, feature extraction."""
        try:
            quality_metrics = assess_signal_quality(signal_win, self.sampling_rate)
            self.quality_buffer.append(quality_metrics.get("overall_quality", 1))
            if quality_metrics.get("overall_quality", 1) < self.quality_threshold:
                return {
                    "quality": quality_metrics,
                    "alert": "Poor signal quality detected",
                }
            # Detect R-peaks via neurokit2.
            peaks = self.r_peak_detector(signal_win, self.sampling_rate)[1][
                "ECG_R_Peaks"
            ]
            waves = None
            if len(peaks) > 0:
                waves = self.wave_delineator(signal_win, peaks, self.sampling_rate)
            features = {}
            if waves is not None:
                features.update(extract_morphological_features(signal_win, waves))
                if len(peaks) >= 2:
                    rr_intervals = np.diff(peaks) / self.sampling_rate * 1000
                    features.update(calculate_advanced_hrv(rr_intervals))
            self.feature_buffer.append(features)
            alerts = self._generate_alerts(features, quality_metrics)
            return {
                "filtered_signal": signal_win,
                "r_peaks": peaks,
                "waves": waves,
                "features": features,
                "quality": quality_metrics,
                "alerts": alerts,
            }
        except Exception as e:
            logger.error(f"Error in window processing: {str(e)}")
            return {}

    def _generate_alerts(self, features: Dict, quality: Dict) -> List[str]:
        """Generate alerts based on features and quality metrics."""
        alerts = []
        if quality.get("overall_quality", 1) < self.quality_threshold:
            alerts.append("Poor signal quality")
        if quality.get("powerline_interference_present", False):
            alerts.append("Power line interference detected")
        if features:
            hr = features.get("Heart_Rate", 0)
            if hr > 100:
                alerts.append(f"Tachycardia detected (HR: {hr:.0f} bpm)")
            elif hr < 60:
                alerts.append(f"Bradycardia detected (HR: {hr:.0f} bpm)")
            if features.get("ST_elevation", False):
                alerts.append("ST elevation detected")
            elif features.get("ST_depression", False):
                alerts.append("ST depression detected")
            if features.get("TWA_present", False):
                alerts.append("T-wave alternans detected")
        return alerts


# ---------------------------
# End of Core Processing Module
# ---------------------------
if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
        from scipy.signal import find_peaks

        sampling_rate = ECGConfig.DEFAULT_SAMPLING_RATE
        duration = 10  # seconds
        ecg_signal = nk.ecg_simulate(
            duration=duration, sampling_rate=sampling_rate, noise=0.1
        )

        # Use the ECGPreprocessor for offline processing.
        preprocessor = ECGPreprocessor(debug=True)

        # Optionally, override QRS detection and segmentation methods.
        def improved_detect_qrs_peaks(self, signal: np.ndarray) -> np.ndarray:
            try:
                _, peaks = nk.ecg_peaks(signal, sampling_rate=self.fs)
                return peaks["ECG_R_Peaks"]
            except Exception as e:
                logger.error(f"Error in improved QRS detection: {str(e)}")
                from scipy.signal import find_peaks

                peaks, _ = find_peaks(signal, distance=int(0.5 * self.fs))
                return peaks

        def improved_segment_beats(
            self, signal: np.ndarray, peaks: np.ndarray
        ) -> List[np.ndarray]:
            beats = []
            pre_window = int(0.2 * self.fs)
            post_window = int(0.4 * self.fs)
            for peak in peaks:
                if peak - pre_window >= 0 and peak + post_window < len(signal):
                    beat = signal[peak - pre_window : peak + post_window]
                    if len(beat) == pre_window + post_window:
                        beats.append(beat)
            return beats

        preprocessor._detect_qrs_peaks = improved_detect_qrs_peaks.__get__(
            preprocessor, ECGPreprocessor
        )
        preprocessor._segment_beats = improved_segment_beats.__get__(
            preprocessor, ECGPreprocessor
        )

        print("Processing ECG signal with ECGPreprocessor...")
        results = preprocessor.process_signal(ecg_signal)

        print("\nProcessing Results:")
        print(f"Original signal length: {len(results['original_signal'])}")
        print(f"Processed signal length: {len(results['processed_signal'])}")
        print(f"Number of QRS peaks detected: {len(results['peaks'])}")
        print(f"Number of beats extracted: {len(results['beats'])}")

        t = np.arange(len(ecg_signal)) / sampling_rate
        plt.figure(figsize=(15, 10))
        plt.subplot(211)
        plt.plot(t, results["original_signal"], label="Original", alpha=0.7)
        plt.plot(t, results["processed_signal"], label="Processed", alpha=0.7)
        plt.plot(
            results["peaks"] / sampling_rate,
            results["processed_signal"][results["peaks"]],
            "ro",
            label="QRS Peaks",
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("ECG Signal Processing Results")
        plt.legend()
        plt.grid(True)
        if results["beats"]:
            plt.subplot(212)
            beat_time = np.arange(len(results["beats"][0])) / sampling_rate
            for i, beat in enumerate(results["beats"][:5]):
                plt.plot(beat_time, beat, label=f"Beat {i + 1}", alpha=0.7)
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.title("Individual Beats")
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Optionally, train and test classifier if enough beats are present.
        if len(results["beats"]) >= 10:
            print("\nTraining classifier...")
            n_beats = len(results["beats"])
            labels = np.random.choice([0, 1], size=n_beats, p=[0.8, 0.2])
            try:
                ECGConfig.ensure_model_dir()
                training_success = preprocessor.train_classifier(
                    results["beats"], labels
                )
                if training_success:
                    print("Classifier trained successfully")
                    print(f"Models saved to {ECGConfig.MODEL_DIR}")
                    classification_results = preprocessor.classify_beats(
                        results["beats"]
                    )
                    if classification_results:
                        print("\nClassification Results:")
                        print(
                            f"Number of beats classified: {len(classification_results['classifications'])}"
                        )
                        unique, counts = np.unique(
                            classification_results["classifications"],
                            return_counts=True,
                        )
                        print("Class distribution:", dict(zip(unique, counts)))
                else:
                    print("Classifier training failed")
            except Exception as e:
                print(f"Error in classifier training/testing: {str(e)}")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
