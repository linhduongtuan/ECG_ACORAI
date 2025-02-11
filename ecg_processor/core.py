from typing import Dict, List, Optional
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from scipy import signal
from scipy.signal import filtfilt
from .config import ECGConfig
from .utils import create_bandpass_filter, create_notch_filter, normalize_signal
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
import collections
import neurokit2 as nk

logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Custom exception for ECG signal processing errors."""

    pass


class ValidationTracker:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def validate_signal(self, signal: np.ndarray, stage: str) -> bool:
        """Validate the signal at different processing stages."""
        try:
            if not isinstance(signal, np.ndarray):
                raise ValueError(f"{stage}: Signal must be a numpy array")

            if signal.ndim != 1:
                raise ValueError(f"{stage}: Signal must be 1-dimensional")

            if len(signal) < 2:  # Minimum length for meaningful processing
                raise ValueError(f"{stage}: Signal is too short")

            if not np.isfinite(signal).all():
                raise ValueError(f"{stage}: Signal contains non-finite values")

            if np.all(signal == signal[0]):
                raise ValueError(f"{stage}: Signal is constant")

            return True
        except ValueError as e:
            self.errors.append((stage, str(e)))
            return False

    def add_warning(self, stage: str, message: str) -> None:
        self.warnings.append((stage, message))

    def get_report(self) -> Dict:
        return {"warnings": self.warnings, "errors": self.errors}


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
        """Initialize ECG preprocessor with specified sampling rate and lead configuration."""
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

        # Validate initialization parameters
        if not isinstance(sampling_rate, int) or sampling_rate <= 0:
            raise ValueError(
                f"Sampling rate must be positive integer, got {sampling_rate}"
            )

        if lead_config not in ["single", "multi"]:
            raise ValueError(
                f"Invalid lead config '{lead_config}'. Must be 'single' or 'multi'"
            )

        try:
            # Create filters
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

        except Exception as e:
            logger.error(f"Filter initialization failed: {str(e)}")
            raise

    @debug_and_profile
    def process_signal(self, signal: np.ndarray) -> Dict:
        """Process ECG signal with comprehensive validation and debugging."""
        try:
            # Validate input signal
            if not self.tracker.validate_signal(signal, "input_validation"):
                raise ProcessingError("Input signal validation failed")

            # Store original signal
            original_signal = signal.copy()

            # Apply filters with validation
            signal = self._apply_filters(signal)
            if not self.tracker.validate_signal(signal, "filtering"):
                raise ProcessingError("Signal filtering failed")

            signal = normalize_signal(signal)
            if not self.tracker.validate_signal(signal, "normalization"):
                raise ProcessingError("Signal normalization failed")

            # QRS detection and segmentation
            peaks = self._detect_qrs_peaks(signal)
            if len(peaks) == 0:
                self.tracker.add_warning("qrs_detection", "No QRS peaks detected")

            beats = self._segment_beats(signal, peaks)
            if len(beats) == 0:
                self.tracker.add_warning("beat_segmentation", "No beats segmented")

            # Analysis
            hrv_metrics = self._calculate_hrv(peaks)
            beat_features = self._extract_features(beats)
            quality_metrics = calculate_signal_quality(original_signal, signal, self.fs)

            # Compile results
            results = {
                "original_signal": original_signal,
                "processed_signal": signal,
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

    def _apply_filters(self, signal: np.ndarray) -> np.ndarray:
        """Apply all filters to the signal with comprehensive denoising."""
        try:
            # Basic filtering (bandpass and notch)
            signal_filtered = filtfilt(self.bp_b, self.bp_a, signal)
            signal_filtered = filtfilt(self.notch_b, self.notch_a, signal_filtered)

            # Advanced denoising if enabled
            if hasattr(self, "advanced_denoising") and self.advanced_denoising:
                from .advanced_denoising import advanced_denoise_pipeline

                denoising_results = advanced_denoise_pipeline(
                    signal_filtered,
                    self.fs,
                    remove_resp=self.remove_respiratory,
                    remove_emg=self.remove_emg,
                    remove_eda=self.remove_eda,
                )
                signal_filtered = denoising_results["denoised"]

                if self.debug:
                    logger.debug("Advanced denoising applied:")
                    for stage, signal_stage in denoising_results.items():
                        if stage != "original":
                            snr = calculate_signal_quality(
                                denoising_results["original"], signal_stage
                            )
                            logger.debug(f"- {stage}: SNR = {snr:.2f} dB")

            return signal_filtered

        except Exception as e:
            logger.error(f"Error in filter application: {str(e)}")
            raise ProcessingError(f"Filter application failed: {str(e)}")

    def _detect_qrs_peaks(self, signal: np.ndarray) -> np.ndarray:
        """
        Detect QRS complexes in the ECG signal using Pan-Tompkins algorithm.

        Parameters
        ----------
        signal : np.ndarray
            Pre-processed ECG signal

        Returns
        -------
        np.ndarray
            Array of QRS peak indices

        Raises
        ------
        ValueError
            If signal is invalid or too short
        """
        if not isinstance(signal, np.ndarray):
            raise TypeError("Signal must be a numpy array")

        if signal.ndim != 1:
            raise ValueError(
                f"Signal must be 1-dimensional, got {signal.ndim} dimensions"
            )

        if len(signal) < self.fs:  # At least 1 second of data
            raise ValueError(f"Signal too short. Must be at least {self.fs} samples")

        try:
            # Implementation of Pan-Tompkins algorithm
            # 1. Apply bandpass filter (already done in process_signal)
            # 2. Derivative
            derivative = np.gradient(signal)

            # 3. Square
            squared = derivative**2

            # 4. Moving average
            window = int(0.15 * self.fs)  # 150ms window
            if window % 2 == 0:
                window += 1
            ma = np.convolve(squared, np.ones(window) / window, mode="same")

            # 5. Adaptive thresholding
            threshold = 0.3 * np.mean(ma)

            # 6. Find peaks
            from scipy.signal import find_peaks

            peaks, _ = find_peaks(
                ma, height=threshold, distance=int(0.2 * self.fs)
            )  # Min 200ms between peaks

            return peaks

        except Exception as e:
            logger.error(f"Error in QRS detection: {str(e)}")
            raise

    def _segment_beats(self, signal: np.ndarray, peaks: np.ndarray) -> List[np.ndarray]:
        """
        Segment ECG signal into individual beats centered around QRS peaks.

        Parameters
        ----------
        signal : np.ndarray
            Pre-processed ECG signal
        peaks : np.ndarray
            Array of QRS peak indices

        Returns
        -------
        List[np.ndarray]
            List of segmented beats

        Raises
        ------
        ValueError
            If inputs are invalid or incompatible
        """
        if not isinstance(signal, np.ndarray) or not isinstance(peaks, np.ndarray):
            raise TypeError("Signal and peaks must be numpy arrays")

        if len(peaks) == 0:
            logger.warning("No peaks provided for segmentation")
            return []

        if peaks.max() >= len(signal):
            raise ValueError("Peak indices exceed signal length")

        try:
            beats = []
            # Window size: 600ms total (200ms before peak, 400ms after)
            pre_window = int(0.2 * self.fs)
            post_window = int(0.4 * self.fs)

            for peak in peaks:
                start = max(0, peak - pre_window)
                end = min(len(signal), peak + post_window)

                # Only include beats with full window size
                if end - start == pre_window + post_window:
                    beat = signal[start:end]
                    # Basic quality check
                    if not np.isfinite(beat).all():
                        logger.warning(f"Invalid values in beat at peak {peak}")
                        continue
                    beats.append(beat)
                else:
                    logger.debug(f"Skipping incomplete beat at peak {peak}")

            return beats

        except Exception as e:
            logger.error(f"Error in beat segmentation: {str(e)}")
            raise

    def _calculate_hrv(self, peaks: np.ndarray) -> Dict:
        """Calculate HRV metrics"""
        rr_intervals = np.diff(peaks) / self.fs * 1000

        time_metrics = calculate_time_domain_hrv(rr_intervals)
        freq_metrics = calculate_frequency_domain_hrv(rr_intervals)

        return {**time_metrics, **freq_metrics}

    def _extract_features(self, beats: List[np.ndarray]) -> Dict:
        """
        Extract multiple types of features from each beat.

        Parameters
        ----------
        beats : List[np.ndarray]
            List of segmented beats

        Returns
        -------
        Dict
            Dictionary of features for each beat

        Raises
        ------
        ValueError
            If beats are invalid or feature extraction fails
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
                    # Start with statistical features which are more robust
                    beat_features = extract_statistical_features(beat)

                    # Try to add wavelet features
                    try:
                        wavelet_features = extract_wavelet_features(beat)
                        beat_features.update(wavelet_features)
                    except Exception as e:
                        logger.warning(
                            f"Wavelet feature extraction failed for beat {i}: {str(e)}"
                        )

                    # Try to add morphological features with proper error handling
                    try:
                        morph_features = extract_morphological_features(beat, self.fs)
                        beat_features.update(morph_features)
                    except Exception as e:
                        logger.warning(
                            f"Morphological feature extraction failed for beat {i}: {str(e)}"
                        )

                    # Validate features
                    beat_features = {
                        k: v for k, v in beat_features.items() if v is not None
                    }

                    if beat_features:  # Only add if we have valid features
                        features[f"beat_{i}"] = beat_features
                        self._feature_cache[beat_hash] = beat_features

                except Exception as e:
                    logger.warning(f"Error extracting features for beat {i}: {str(e)}")
                    continue

            return features

        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            raise

    def train_classifier(self, beats: List[np.ndarray], labels: np.ndarray) -> bool:
        """Train beat classifier"""
        try:
            X = self._prepare_features_for_classification(beats)
            if X.size == 0:
                raise ValueError("No valid features for training")

            X = self.scaler.fit_transform(X)

            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.classifier.fit(X, labels)

            # Ensure model directory exists and save models
            ECGConfig.ensure_model_dir()

            try:
                joblib.dump(self.classifier, ECGConfig.MODEL_PATH)
                joblib.dump(self.scaler, ECGConfig.SCALER_PATH)
                logger.info(f"Models saved successfully to {ECGConfig.MODEL_DIR}")
            except Exception as e:
                logger.warning(f"Could not save models: {str(e)}")
                # Continue even if saving fails

            return True

        except Exception as e:
            logger.error(f"Error training classifier: {str(e)}")
            return False

    def classify_beats(self, beats: List[np.ndarray]) -> Optional[Dict]:
        """Classify beats using trained model"""
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
        """
        Load trained classifier and scaler from disk.

        Raises
        ------
        FileNotFoundError
            If model or scaler files don't exist
        ValueError
            If loaded models are invalid
        """
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

            # Validate loaded models
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
        """Prepare features for classification with validation."""
        try:
            features = self._extract_features(beats)
            if not features:
                raise ValueError("No valid features extracted from beats")

            # Get the feature names from the first beat to ensure consistency
            first_beat_features = next(iter(features.values()))
            feature_names = list(first_beat_features.keys())

            # Prepare feature matrix
            X = []
            for beat_id, beat_features in features.items():
                # Ensure all beats have the same features
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


class RealTimeECGProcessor:
    """
    Real-time ECG signal processor with streaming capabilities.

    This class implements a real-time processing pipeline for ECG signals,
    including buffering, online filtering, and feature extraction.
    """

    def __init__(
        self, sampling_rate: int = 250, buffer_size: int = 2000, overlap: int = 500
    ):
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.overlap = overlap

        # Initialize buffers
        self.signal_buffer = collections.deque(maxlen=buffer_size)
        self.feature_buffer = collections.deque(maxlen=100)
        self.quality_buffer = collections.deque(maxlen=100)

        # Initialize filters
        self.initialize_filters()

        # Initialize preprocessor
        self.preprocessor = ECGPreprocessor(
            sampling_rate=sampling_rate, advanced_denoising=True
        )

        # Initialize quality assessor
        self.quality_threshold = 0.6

        # Initialize feature extractors
        self.initialize_feature_extractors()

        # Statistics for online standardization
        self.signal_mean = 0
        self.signal_std = 1
        self.alpha = 0.01  # Update rate for online statistics

    def initialize_filters(self):
        """Initialize online filters."""
        nyq = self.sampling_rate / 2

        # Bandpass filter (0.5-40 Hz)
        self.bp_b, self.bp_a = signal.butter(3, [0.5 / nyq, 40 / nyq], btype="band")

        # Notch filter (50 Hz)
        self.notch_b, self.notch_a = signal.iirnotch(50 / nyq, 30)

        # Initialize filter states
        self.bp_state = signal.lfilter_zi(self.bp_b, self.bp_a)
        self.notch_state = signal.lfilter_zi(self.notch_b, self.notch_a)

    def initialize_feature_extractors(self):
        """Initialize feature extraction components."""
        self.r_peak_detector = nk.ecg_peaks
        self.wave_delineator = nk.ecg_delineate

    def update_online_statistics(self, new_value: float):
        """
        Update online mean and standard deviation using Welford's online algorithm.

        Parameters
        ----------
        new_value : float
            New sample value
        """
        # Count number of samples processed
        if not hasattr(self, "_n_samples"):
            self._n_samples = 0
        self._n_samples += 1

        # Update mean and variance using Welford's online algorithm
        delta = new_value - self.signal_mean
        self.signal_mean += delta / self._n_samples

        # Update variance
        delta2 = new_value - self.signal_mean
        if self._n_samples > 1:
            self.signal_std = np.sqrt(
                ((self._n_samples - 2) * (self.signal_std**2) + delta * delta2)
                / (self._n_samples - 1)
            )

    def process_sample(self, sample: float) -> Dict:
        """
        Process a single ECG sample in real-time.

        Parameters
        ----------
        sample : float
            New ECG sample

        Returns
        -------
        Dict
            Processing results including:
            - Filtered signal
            - Current quality assessment
            - Detected features
            - Alerts if any
        """
        try:
            # Update online statistics
            self.update_online_statistics(sample)

            # Normalize sample
            if self.signal_std > 0:
                sample_normalized = (sample - self.signal_mean) / self.signal_std
            else:
                sample_normalized = sample - self.signal_mean

            # Add to buffer
            self.signal_buffer.append(sample_normalized)

            # Only process when buffer is full
            if len(self.signal_buffer) < self.buffer_size:
                return {}

            # Convert buffer to numpy array
            signal_array = np.array(list(self.signal_buffer))

            # Apply filters using scipy.signal.lfilter
            filtered, self.bp_state = signal.lfilter(
                self.bp_b, self.bp_a, signal_array, zi=self.bp_state * signal_array[0]
            )
            filtered, self.notch_state = signal.lfilter(
                self.notch_b, self.notch_a, filtered, zi=self.notch_state * filtered[0]
            )

            # Process the current window
            result = self._process_window(filtered)

            # Remove overlap samples from buffer
            for _ in range(self.buffer_size - self.overlap):
                self.signal_buffer.popleft()

            return result

        except Exception as e:
            logger.error(f"Error in real-time processing: {str(e)}")
            return {}

    def _process_window(self, signal: np.ndarray) -> Dict:
        """Process a window of ECG signal."""
        try:
            # Assess signal quality
            quality_metrics = assess_signal_quality(signal, self.sampling_rate)
            self.quality_buffer.append(quality_metrics["overall_quality"])

            # Only process if signal quality is good
            if quality_metrics["overall_quality"] < self.quality_threshold:
                return {
                    "quality": quality_metrics,
                    "alert": "Poor signal quality detected",
                }

            # Detect R-peaks
            r_peaks = self.r_peak_detector(signal, self.sampling_rate)[1]["ECG_R_Peaks"]

            # Delineate waves if R-peaks found
            waves = None
            if len(r_peaks) > 0:
                waves = self.wave_delineator(signal, r_peaks, self.sampling_rate)

            # Extract features
            features = {}
            if waves is not None:
                features.update(extract_morphological_features(signal, waves))

                # Extract HRV features if enough R-peaks
                if len(r_peaks) >= 2:
                    rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000
                    features.update(calculate_advanced_hrv(rr_intervals))

            # Store features
            self.feature_buffer.append(features)

            # Generate alerts
            alerts = self._generate_alerts(features, quality_metrics)

            return {
                "filtered_signal": signal,
                "r_peaks": r_peaks,
                "waves": waves,
                "features": features,
                "quality": quality_metrics,
                "alerts": alerts,
            }

        except Exception as e:
            logger.error(f"Error in window processing: {str(e)}")
            return {}

    def _generate_alerts(self, features: Dict, quality: Dict) -> List[str]:
        """Generate alerts based on features and signal quality."""
        alerts = []

        # Quality alerts
        if quality.get("overall_quality", 1) < self.quality_threshold:
            alerts.append("Poor signal quality")

        if quality.get("powerline_interference_present", False):
            alerts.append("Power line interference detected")

        # Feature-based alerts
        if features:
            # Heart rate alerts
            hr = features.get("Heart_Rate", 0)
            if hr > 100:
                alerts.append(f"Tachycardia detected (HR: {hr:.0f} bpm)")
            elif hr < 60:
                alerts.append(f"Bradycardia detected (HR: {hr:.0f} bpm)")

            # ST segment alerts
            if features.get("ST_elevation", False):
                alerts.append("ST elevation detected")
            elif features.get("ST_depression", False):
                alerts.append("ST depression detected")

            # T-wave alternans alert
            if features.get("TWA_present", False):
                alerts.append("T-wave alternans detected")

        return alerts


if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
        from scipy.signal import find_peaks
        import neurokit2 as nk

        # Generate a more realistic ECG signal using neurokit2
        sampling_rate = ECGConfig.DEFAULT_SAMPLING_RATE
        duration = 10  # seconds
        ecg_signal = nk.ecg_simulate(
            duration=duration, sampling_rate=sampling_rate, noise=0.1
        )

        # Instantiate the ECGPreprocessor
        preprocessor = ECGPreprocessor(debug=True)

        def improved_detect_qrs_peaks(self, signal: np.ndarray) -> np.ndarray:
            """Improved QRS detection using neurokit2."""
            try:
                # Use neurokit2's QRS detection
                _, peaks = nk.ecg_peaks(signal, sampling_rate=self.fs)
                return peaks["ECG_R_Peaks"]
            except Exception as e:
                logger.error(f"Error in QRS detection: {str(e)}")
                # Fallback to basic peak detection
                peaks, _ = find_peaks(signal, distance=int(0.5 * self.fs))
                return peaks

        def improved_segment_beats(
            self, signal: np.ndarray, peaks: np.ndarray
        ) -> List[np.ndarray]:
            """Improved beat segmentation with proper window sizes."""
            beats = []
            # Use 600ms window (200ms before and 400ms after R peak)
            pre_window = int(0.2 * self.fs)
            post_window = int(0.4 * self.fs)

            for peak in peaks:
                if peak - pre_window >= 0 and peak + post_window < len(signal):
                    beat = signal[peak - pre_window : peak + post_window]
                    if (
                        len(beat) == pre_window + post_window
                    ):  # Ensure consistent length
                        beats.append(beat)
            return beats

        # Override the methods
        preprocessor._detect_qrs_peaks = improved_detect_qrs_peaks.__get__(
            preprocessor, ECGPreprocessor
        )
        preprocessor._segment_beats = improved_segment_beats.__get__(
            preprocessor, ECGPreprocessor
        )

        # Process the ECG signal
        print("Processing ECG signal...")
        results = preprocessor.process_signal(ecg_signal)

        # Print results
        print("\nProcessing Results:")
        print(f"Original signal length: {len(results['original_signal'])}")
        print(f"Processed signal length: {len(results['processed_signal'])}")
        print(f"Number of QRS peaks detected: {len(results['peaks'])}")
        print(f"Number of beats extracted: {len(results['beats'])}")

        # Plot results
        t = np.arange(len(ecg_signal)) / sampling_rate

        plt.figure(figsize=(15, 10))

        # Original vs Processed Signal
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

        # Plot individual beats
        if results["beats"]:
            plt.subplot(212)
            beat_time = np.arange(len(results["beats"][0])) / sampling_rate
            for i, beat in enumerate(results["beats"][:5]):  # Plot first 5 beats
                plt.plot(beat_time, beat, label=f"Beat {i + 1}", alpha=0.7)
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.title("Individual Beats")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Train and test classifier if we have enough beats
        if len(results["beats"]) >= 10:
            print("\nTraining classifier...")
            # Create synthetic labels (0: Normal, 1: Abnormal)
            n_beats = len(results["beats"])
            labels = np.random.choice([0, 1], size=n_beats, p=[0.8, 0.2])

            try:
                # Ensure model directory exists
                ECGConfig.ensure_model_dir()

                training_success = preprocessor.train_classifier(
                    results["beats"], labels
                )
                if training_success:
                    print("Classifier trained successfully")
                    print(f"Models saved to {ECGConfig.MODEL_DIR}")

                    # Test classification
                    try:
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
                    except Exception as e:
                        print(f"Error in classification: {str(e)}")
                else:
                    print("Classifier training failed")

            except Exception as e:
                print(f"Error in classifier training/testing: {str(e)}")
                import traceback

                traceback.print_exc()

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback

        traceback.print_exc()
