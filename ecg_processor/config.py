from pathlib import Path


class ECGConfig:
    # Existing parameters...
    DEFAULT_SAMPLING_RATE = 500
    FILTER_LOWCUT = 0.5
    FILTER_HIGHCUT = 40.0
    FILTER_ORDER = 4
    NOTCH_FREQ = 50.0  # or 60.0 for US power line frequency
    NOTCH_Q = 30.0

    # QRS detection parameters
    QRS_DISTANCE = 150
    PAN_TOMPKINS_LOW = 5
    PAN_TOMPKINS_HIGH = 15

    # Segmentation parameters
    SEGMENT_BEFORE = 100
    SEGMENT_AFTER = 200

    # HRV analysis parameters
    VLF_LOW = 0.003
    VLF_HIGH = 0.04
    LF_LOW = 0.04
    LF_HIGH = 0.15
    HF_LOW = 0.15
    HF_HIGH = 0.4

    # Non-linear HRV parameters
    POINCARE_PLOT_SIZE = (8, 8)
    ENTROPY_M = 2  # embedding dimension
    ENTROPY_R = 0.2  # tolerance
    DFA_SCALE_MIN = 4
    DFA_SCALE_MAX = None  # will be calculated as N/4

    # Advanced denoising configuration
    ADVANCED_DENOISING = True  # Enable/disable advanced denoising
    REMOVE_RESPIRATORY = True  # Remove respiratory noise
    REMOVE_EMG = True  # Remove EMG noise
    REMOVE_EDA = True  # Remove EDA noise

    # Respiratory noise parameters
    RESP_FREQ_RANGE = (0.15, 0.4)  # Hz

    # EMG noise parameters
    EMG_FREQ_THRESHOLD = 20.0  # Hz

    # EDA noise parameters
    EDA_FREQ_RANGE = (0.01, 1.0)  # Hz

    # Signal quality thresholds
    MIN_SNR_THRESHOLD = 10.0  # Minimum acceptable Signal-to-Noise Ratio in dB
    MIN_BEAT_AMPLITUDE = 0.1  # Minimum acceptable beat amplitude
    MAX_BEAT_NOISE = 0.05  # Maximum acceptable noise level in a beat
    MAX_BASELINE_DRIFT = 0.2  # Maximum acceptable baseline drift

    # Model paths
    MODEL_DIR = Path("models")
    MODEL_PATH = MODEL_DIR / "ecg_classifier.joblib"
    SCALER_PATH = MODEL_DIR / "ecg_scaler.joblib"

    @classmethod
    def ensure_model_dir(cls):
        """Ensure model directory exists"""
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
