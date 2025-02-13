# ecg_preprocessor_torch.py

import torch
from typing import Dict, List, Optional
import numpy as np
import logging
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from scipy.signal import filtfilt

from .config import ECGConfig
from .ecg_data_loader import ECGDataLoader
from .utils import create_bandpass_filter, create_notch_filter, normalize_signal
from .features import extract_statistical_features, extract_morphological_features
from .hrv import calculate_time_domain_hrv, calculate_frequency_domain_hrv
from .quality import calculate_signal_quality

# Import advanced denoising functions
from .advanced_denoising import (
    wavelet_denoise,
    adaptive_lms_filter,
    emd_denoise,
    median_filter_signal,
    smooth_signal,
)
from .features_transform import (
    extract_stft_features,
    extract_wavelet_features,
    extract_hybrid_features,
)

logger = logging.getLogger(__name__)


def _to_numpy(data):
    """
    Convert input to a NumPy array if it is a torch.Tensor.
    Otherwise, return as is.
    """
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return data


class ECGPreprocessor:
    def __init__(
        self,
        sampling_rate: int = ECGConfig.DEFAULT_SAMPLING_RATE,
        lead_config: str = "single",
        debug: bool = False,
    ):
        """Initialize the ECG preprocessor with configuration and filters."""
        # Validate initialization parameters
        if not isinstance(sampling_rate, int) or sampling_rate <= 0:
            raise ValueError(
                f"Sampling rate must be a positive integer, got {sampling_rate}"
            )
        if lead_config not in ["single", "multi"]:
            raise ValueError(f"Invalid lead configuration: {lead_config}")

        self.fs = sampling_rate
        self.lead_config = lead_config
        self.debug = debug
        self.scaler = StandardScaler()
        self.classifier = None

        # Create filters using the torch-adapted utility functions.
        self.bp_b, self.bp_a = create_bandpass_filter(
            ECGConfig.FILTER_LOWCUT,
            ECGConfig.FILTER_HIGHCUT,
            self.fs,
            ECGConfig.FILTER_ORDER,
        )
        self.notch_b, self.notch_a = create_notch_filter(
            ECGConfig.NOTCH_FREQ, ECGConfig.NOTCH_Q, self.fs
        )

        self.data_loader = ECGDataLoader()

    def load_and_process(self, file_path: str, **kwargs) -> Dict:
        """Load ECG data from file and process the signal."""
        # Load data (the torch version of your data loader returns torch tensors)
        data = self.data_loader.load_data(file_path, **kwargs)

        # Process the signal. The process_signal method will convert inputs as needed.
        results = self.process_signal(data["signal"])

        # Add metadata and annotations if available.
        results["metadata"] = data.get("metadata", {})
        results["annotations"] = data.get("annotations", None)

        return results

    def process_signal(self, signal, denoise_method: str = "none", **kwargs) -> Dict:
        """Process an ECG signal using filtering, denoising, segmentation, feature extraction, and quality assessment."""
        try:
            # Accept signal as either a NumPy array or a torch.Tensor.
            signal = _to_numpy(signal)

            if not isinstance(signal, np.ndarray):
                raise ValueError("Signal must be a numpy array or a torch tensor")
            if len(signal) == 0:
                raise ValueError("Signal cannot be empty")
            if np.any(np.isnan(signal)):
                raise ValueError("Signal contains NaN values")
            if len(signal) < self.fs:
                raise ValueError(
                    f"Signal too short (minimum length: {self.fs} samples)"
                )

            # Store original signal for quality comparisons.
            original_signal = signal.copy()

            # Basic filtering: use SciPy’s filtfilt. (These functions require NumPy arrays.)
            signal = self._apply_filters(signal)

            # Optionally, apply an advanced denoising method.
            if denoise_method != "none":
                signal = self._apply_denoising(signal, method=denoise_method, **kwargs)

            # Perform normalization using your torch-based normalization.
            signal = normalize_signal(signal)

            # QRS detection and beat segmentation.
            try:
                peaks = self._detect_qrs_peaks(signal)
                if peaks is None or len(peaks) == 0:
                    logger.warning("No QRS peaks detected in the signal")
                    peaks = np.array([])
            except Exception as e:
                logger.warning(f"QRS detection failed: {str(e)}")
                peaks = np.array([])

            try:
                beats = self._segment_beats(signal, peaks) if len(peaks) > 0 else []
            except Exception as e:
                logger.warning(f"Beat segmentation failed: {str(e)}")
                beats = []

            try:
                hrv_metrics = self._calculate_hrv(peaks) if len(peaks) > 1 else {}
            except Exception as e:
                logger.warning(f"HRV calculation failed: {str(e)}")
                hrv_metrics = {}

            try:
                beat_features = self._extract_features(beats) if len(beats) > 0 else {}
            except Exception as e:
                logger.warning(f"Feature extraction failed: {str(e)}")
                beat_features = {}

            try:
                quality_metrics = calculate_signal_quality(
                    original_signal, signal, self.fs
                )
            except Exception as e:
                logger.warning(f"Quality assessment failed: {str(e)}")
                quality_metrics = {}

            return {
                "original_signal": original_signal,
                "processed_signal": signal,
                "peaks": peaks,
                "beats": beats,
                "hrv_metrics": hrv_metrics,
                "beat_features": beat_features,
                "quality_metrics": quality_metrics,
            }
        except Exception as e:
            logger.error(f"Error processing signal: {str(e)}")
            raise

    def _apply_filters(self, signal: np.ndarray) -> np.ndarray:
        """Apply bandpass and notch filters to the input ECG signal."""
        signal = filtfilt(self.bp_b, self.bp_a, signal)
        signal = filtfilt(self.notch_b, self.notch_a, signal)
        return signal

    def _apply_denoising(
        self, signal: np.ndarray, method: str = "none", **kwargs
    ) -> np.ndarray:
        """
        Apply additional denoising techniques.

        method: 'wavelet', 'adaptive', 'emd', 'median', 'smoothing', or 'none'
        kwargs: Additional parameters for the specified method.
        """
        if method == "wavelet":
            return wavelet_denoise(signal, **kwargs)
        elif method == "adaptive":
            reference = kwargs.get("reference", None)
            if reference is None:
                raise ValueError(
                    "A reference signal is required for adaptive filtering."
                )
            mu = kwargs.get("mu", 0.01)
            filter_order = kwargs.get("filter_order", 32)
            return adaptive_lms_filter(signal, reference, mu, filter_order)
        elif method == "emd":
            return emd_denoise(signal, **kwargs)
        elif method == "median":
            return median_filter_signal(signal, **kwargs)
        elif method == "smoothing":
            return smooth_signal(signal, **kwargs)
        elif method == "none":
            return signal
        else:
            raise ValueError(f"Unknown denoising method: {method}")

    def _detect_qrs_peaks(self, signal: np.ndarray) -> np.ndarray:
        """Detect QRS complexes using NeuroKit2."""
        try:
            # Use nk.ecg_process for comprehensive analysis.
            processed, info = nk.ecg_process(signal, sampling_rate=self.fs)
            peaks = info.get("ECG_R_Peaks")
            if peaks is None or len(peaks) == 0:
                raise ValueError("No R-peaks detected in the ECG signal.")
            return np.array(peaks)
        except Exception as e:
            logger.error(f"Error detecting QRS peaks: {str(e)}")
            raise

    def _segment_beats(self, signal: np.ndarray, peaks: np.ndarray) -> List[np.ndarray]:
        """
        Segment the ECG signal into individual beats using detected R-peaks.
        Uses configuration parameters SEGMENT_BEFORE and SEGMENT_AFTER defined in ECGConfig.
        """
        beats = []
        for r_peak in peaks:
            start_idx = int(max(r_peak - ECGConfig.SEGMENT_BEFORE, 0))
            end_idx = int(min(r_peak + ECGConfig.SEGMENT_AFTER, len(signal)))
            beat = signal[start_idx:end_idx]
            beats.append(beat)
        return beats

    def _calculate_hrv(self, peaks: np.ndarray) -> Dict:
        """Calculate HRV metrics from detected peaks."""
        rr_intervals = np.diff(peaks) / self.fs * 1000  # Convert to milliseconds.
        time_metrics = calculate_time_domain_hrv(rr_intervals)
        freq_metrics = calculate_frequency_domain_hrv(rr_intervals)
        return {**time_metrics, **freq_metrics}

    def _extract_features(self, beats: List[np.ndarray]) -> Dict:
        """Extract features from each beat by combining statistical, wavelet, morphological, STFT, and hybrid features."""
        features = {}
        for i, beat in enumerate(beats):
            if len(beat) < 128:  # Ensure beat length is sufficient.
                logger.warning(
                    f"Beat {i} is too short for feature extraction (length: {len(beat)})"
                )
                continue
            try:
                features[f"beat_{i}"] = {
                    **extract_statistical_features(beat),
                    **extract_wavelet_features(beat, wavelet="db4", level=4),
                    **extract_morphological_features(beat, self.fs),
                    **extract_stft_features(beat, self.fs, nperseg=128),
                    **extract_hybrid_features(
                        beat, self.fs, wavelet="db4", level=4, nperseg=128
                    ),
                }
            except Exception as e:
                logger.warning(f"Error extracting features for beat {i}: {str(e)}")
        return features

    def train_classifier(self, beats: List[np.ndarray], labels: np.ndarray) -> bool:
        """Train a beat classifier using a Random Forest model."""
        try:
            X = self._prepare_features_for_classification(beats)
            if X.size == 0:
                raise ValueError("No valid features extracted for training")
            if len(X) != len(labels):
                raise ValueError(
                    f"Number of feature vectors ({len(X)}) does not match number of labels ({len(labels)})"
                )

            X = self.scaler.fit_transform(X)
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.classifier.fit(X, labels)

            # Save the trained model and scaler.
            ECGConfig.ensure_model_dir()
            joblib.dump(self.classifier, ECGConfig.MODEL_PATH)
            joblib.dump(self.scaler, ECGConfig.SCALER_PATH)
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

    def load_classifier(self):
        """Load the trained classifier and scaler."""
        try:
            self.classifier = joblib.load(ECGConfig.MODEL_PATH)
            self.scaler = joblib.load(ECGConfig.SCALER_PATH)
        except Exception:
            raise ValueError("No trained classifier found")

    def _prepare_features_for_classification(
        self, beats: List[np.ndarray]
    ) -> np.ndarray:
        """
        Prepare a feature matrix by extracting and concatenating features from all beats.
        Features from each beat are truncated or padded so that all feature vectors have equal length.
        """
        try:
            features = self._extract_features(beats)
            if not features:
                raise ValueError("No features extracted")

            X = []
            for beat_features in features.values():
                # Convert feature dictionary into a feature vector.
                feature_vector = []
                for value in beat_features.values():
                    if isinstance(value, (int, float)) and value is not None:
                        feature_vector.append(value)
                if feature_vector:  # Only add nonempty feature vectors.
                    X.append(feature_vector)

            if not X:
                raise ValueError("No valid feature vectors created")

            # Ensure all feature vectors have the same length.
            min_length = min(len(x) for x in X)
            X = [x[:min_length] for x in X]
            return np.array(X)

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return np.array([])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import neurokit2 as nk
    import numpy as np
    from ecg_processor.config import ECGConfig  # Adjust the path if needed

    # 1. Simulate a realistic 10-second ECG signal with NeuroKit2.
    fs = ECGConfig.DEFAULT_SAMPLING_RATE  # e.g., 500 Hz
    duration = 10  # in seconds
    ecg_simulated = nk.ecg_simulate(duration=duration, sampling_rate=fs, noise=0.01)

    # 2. Create an instance of the ECGPreprocessor.
    preprocessor = ECGPreprocessor()

    # 3. Process the ECG signal.
    #    Use 'none' as the denoising method here (try others like 'wavelet', 'adaptive', etc. as needed).
    try:
        result = preprocessor.process_signal(ecg_simulated, denoise_method="none")
    except Exception as e:
        print("Error during ECG processing:", e)
        exit(1)

    # 4. Display information about the processed signal.
    print("Original signal shape:", result["original_signal"].shape)
    print("Processed signal shape:", result["processed_signal"].shape)
    print("Number of detected peaks:", len(result["peaks"]))
    print("Number of beats segmented:", len(result["beats"]))
    print("HRV Metrics:", result["hrv_metrics"])
    print("Quality Metrics:", result["quality_metrics"])

    # 5. Plot original vs. processed signal.
    t = np.linspace(0, duration, len(ecg_simulated))
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t, result["original_signal"], label="Original ECG", color="C0")
    plt.title("Original ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, result["processed_signal"], label="Processed ECG", color="C1")
    plt.title("Processed ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 6. (Optional) Train and test the beat classifier if beats were detected.
    if len(result["beats"]) > 0:
        # For demonstration, assign a random binary label to each beat.
        labels = np.random.randint(0, 2, size=len(result["beats"]))
        training_success = preprocessor.train_classifier(result["beats"], labels)
        if training_success:
            print("Classifier trained successfully.")
            classification_result = preprocessor.classify_beats(result["beats"])
            print("Classification Results:")
            print("  Beat classifications:", classification_result["classifications"])
            print("  Class probabilities:", classification_result["probabilities"])
        else:
            print("Classifier training failed.")
    else:
        print("No beats were detected; classifier training was skipped.")
