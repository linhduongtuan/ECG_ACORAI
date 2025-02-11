import numpy as np
import pywt
from scipy import signal, stats
from typing import Dict, Optional, Tuple, List
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def extract_stft_features(
    beat: np.ndarray,
    fs: float,
    nperseg: int = 128,
    noverlap: Optional[int] = None,
    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict:
    """
    Extract features from the Short-Time Fourier Transform (STFT) of an ECG beat.

    Parameters
    ----------
    beat : np.ndarray
        1D NumPy array representing the ECG beat
    fs : float
        Sampling frequency in Hz
    nperseg : int, optional
        Length of each segment for STFT, by default 128
    noverlap : Optional[int], optional
        Number of overlapping samples between segments, by default None
    freq_bands : Optional[Dict[str, Tuple[float, float]]], optional
        Dictionary of frequency bands to analyze, by default None
        Example: {'low': (0, 5), 'medium': (5, 20), 'high': (20, 50)}

    Returns
    -------
    Dict
        Dictionary containing STFT features:
        - Basic energy statistics
        - Frequency band powers
        - Spectral shape metrics

    Raises
    ------
    ValueError
        If input parameters are invalid
    """
    try:
        # Input validation
        if not isinstance(beat, np.ndarray):
            raise ValueError("Beat must be a numpy array")
        if len(beat.shape) != 1:
            raise ValueError("Beat must be a 1D array")
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("Sampling frequency must be positive")
        if len(beat) < nperseg:
            raise ValueError(f"Beat length must be at least {nperseg}")

        # Default frequency bands if not provided
        if freq_bands is None:
            freq_bands = {
                "low": (0, 5),  # 0-5 Hz
                "medium": (5, 20),  # 5-20 Hz
                "high": (20, 50),  # 20-50 Hz
            }

        # Compute STFT
        f, t, Zxx = signal.stft(beat, fs=fs, nperseg=nperseg, noverlap=noverlap)
        spectrogram = np.abs(Zxx)

        # Basic energy statistics
        energy_stats = {
            "stft_mean_energy": float(np.mean(spectrogram)),
            "stft_std_energy": float(np.std(spectrogram)),
            "stft_max_energy": float(np.max(spectrogram)),
            "stft_min_energy": float(np.min(spectrogram)),
            "stft_total_energy": float(np.sum(spectrogram)),
            "stft_median_energy": float(np.median(spectrogram)),
        }

        # Frequency band powers
        band_powers = {}
        for band_name, (low_freq, high_freq) in freq_bands.items():
            mask = (f >= low_freq) & (f <= high_freq)
            band_power = np.sum(spectrogram[mask, :]) / spectrogram.size
            band_powers[f"stft_power_{band_name}"] = float(band_power)

        # Spectral shape metrics
        shape_metrics = {
            "stft_spectral_centroid": float(_spectral_centroid(f, spectrogram)),
            "stft_spectral_bandwidth": float(_spectral_bandwidth(f, spectrogram)),
            "stft_spectral_rolloff": float(_spectral_rolloff(spectrogram)),
            "stft_spectral_flatness": float(_spectral_flatness(spectrogram)),
        }

        # Combine all features
        features = {**energy_stats, **band_powers, **shape_metrics}
        return features

    except Exception as e:
        logger.error(f"Error extracting STFT features: {str(e)}")
        raise


def extract_wavelet_features(
    beat: np.ndarray, wavelet: str = "db4", level: int = 4
) -> Dict:
    """
    Extract features using wavelet transform from an ECG beat.

    Parameters
    ----------
    beat : np.ndarray
        1D NumPy array representing the ECG beat
    wavelet : str, optional
        Wavelet type to use, by default 'db4'
    level : int, optional
        Decomposition level, by default 4

    Returns
    -------
    Dict
        Dictionary containing wavelet features:
        - Energy at each level
        - Entropy at each level
        - Statistical features of coefficients
        - Relative energy distribution

    Raises
    ------
    ValueError
        If input parameters are invalid
    """
    try:
        # Input validation
        if not isinstance(beat, np.ndarray):
            raise ValueError("Beat must be a numpy array")
        if len(beat.shape) != 1:
            raise ValueError("Beat must be a 1D array")
        if len(beat) < 2**level:
            raise ValueError(
                f"Beat length must be at least {2**level} for level {level} decomposition"
            )

        # Perform wavelet decomposition
        coeffs = pywt.wavedec(beat, wavelet, level=level)
        features = {}

        # Calculate features for each level
        total_energy = 0
        for i, coef in enumerate(coeffs):
            # Energy features
            energy = np.sum(np.square(coef))
            total_energy += energy
            features[f"wavelet_energy_{i}"] = float(energy)

            # Entropy features
            hist, _ = np.histogram(coef, bins="auto", density=True)
            features[f"wavelet_entropy_{i}"] = float(stats.entropy(hist))

            # Statistical features
            features[f"wavelet_mean_{i}"] = float(np.mean(coef))
            features[f"wavelet_std_{i}"] = float(np.std(coef))
            features[f"wavelet_max_{i}"] = float(np.max(np.abs(coef)))
            features[f"wavelet_median_{i}"] = float(np.median(np.abs(coef)))

        # Calculate relative energies
        if total_energy > 0:
            for i in range(len(coeffs)):
                features[f"wavelet_relative_energy_{i}"] = float(
                    features[f"wavelet_energy_{i}"] / total_energy
                )

        # Add global wavelet features
        features.update(
            {
                "wavelet_total_energy": float(total_energy),
                "wavelet_num_zero_crossings": int(_count_zero_crossings(coeffs)),
                "wavelet_energy_ratio": float(_energy_ratio(coeffs)),
            }
        )

        return features

    except Exception as e:
        logger.error(f"Error extracting wavelet features: {str(e)}")
        raise


def extract_hybrid_features(
    beat: np.ndarray,
    fs: float,
    wavelet: str = "db4",
    level: int = 4,
    nperseg: int = 128,
    noverlap: Optional[int] = None,
    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict:
    """
    Extract hybrid features by combining STFT and wavelet transform features.

    Parameters
    ----------
    beat : np.ndarray
        1D NumPy array representing the ECG beat
    fs : float
        Sampling frequency in Hz
    wavelet : str, optional
        Wavelet type to use, by default 'db4'
    level : int, optional
        Decomposition level, by default 4
    nperseg : int, optional
        STFT segment length, by default 128
    noverlap : Optional[int], optional
        Overlap for STFT segments, by default None
    freq_bands : Optional[Dict[str, Tuple[float, float]]], optional
        Frequency bands for STFT analysis, by default None

    Returns
    -------
    Dict
        Dictionary containing combined features from:
        - STFT analysis
        - Wavelet decomposition
        - Cross-domain relationships

    Raises
    ------
    ValueError
        If input parameters are invalid
    """
    try:
        # Input validation
        if not isinstance(beat, np.ndarray):
            raise ValueError("Beat must be a numpy array")
        if len(beat.shape) != 1:
            raise ValueError("Beat must be a 1D array")
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("Sampling frequency must be positive")

        # Extract individual feature sets
        stft_feats = extract_stft_features(beat, fs, nperseg, noverlap, freq_bands)
        wavelet_feats = extract_wavelet_features(beat, wavelet, level)

        # Combine features
        hybrid_features = {**stft_feats, **wavelet_feats}

        # Add cross-domain features
        try:
            cross_domain = _calculate_cross_domain_features(stft_feats, wavelet_feats)
            hybrid_features.update(cross_domain)
        except Exception as e:
            logger.warning(f"Failed to calculate cross-domain features: {str(e)}")

        return hybrid_features

    except Exception as e:
        logger.error(f"Error extracting hybrid features: {str(e)}")
        raise


# Helper functions for spectral analysis
def _spectral_centroid(freqs: np.ndarray, magnitude: np.ndarray) -> float:
    """
    Calculate spectral centroid (weighted mean of frequencies).

    Parameters
    ----------
    freqs : np.ndarray
        Frequency values
    magnitude : np.ndarray
        Magnitude spectrum

    Returns
    -------
    float
        Spectral centroid frequency
    """
    magnitude_sum = np.sum(magnitude, axis=1)
    return np.sum(freqs * magnitude_sum) / (np.sum(magnitude_sum) + 1e-10)


def _spectral_bandwidth(freqs: np.ndarray, magnitude: np.ndarray) -> float:
    """
    Calculate spectral bandwidth (weighted standard deviation of frequencies).

    Parameters
    ----------
    freqs : np.ndarray
        Frequency values
    magnitude : np.ndarray
        Magnitude spectrum

    Returns
    -------
    float
        Spectral bandwidth
    """
    centroid = _spectral_centroid(freqs, magnitude)
    magnitude_sum = np.sum(magnitude, axis=1)
    variance = np.sum(((freqs - centroid) ** 2) * magnitude_sum) / (
        np.sum(magnitude_sum) + 1e-10
    )
    return np.sqrt(variance)


def _spectral_rolloff(magnitude: np.ndarray, percentile: float = 0.85) -> float:
    """
    Calculate frequency below which percentile of magnitude spectrum energy lies.

    Parameters
    ----------
    magnitude : np.ndarray
        Magnitude spectrum
    percentile : float, optional
        Percentile threshold, by default 0.85

    Returns
    -------
    float
        Frequency index at rolloff point
    """
    power = np.sum(magnitude**2, axis=1)
    cumsum = np.cumsum(power)
    return float(np.where(cumsum >= percentile * cumsum[-1])[0][0])


def _spectral_flatness(magnitude: np.ndarray) -> float:
    """
    Calculate spectral flatness (Wiener entropy).

    Parameters
    ----------
    magnitude : np.ndarray
        Magnitude spectrum

    Returns
    -------
    float
        Spectral flatness value
    """
    power = np.abs(magnitude) ** 2 + 1e-10
    geometric_mean = np.exp(np.mean(np.log(power)))
    arithmetic_mean = np.mean(power)
    return float(geometric_mean / arithmetic_mean)


def _count_zero_crossings(coeffs: List[np.ndarray]) -> int:
    """
    Count zero crossings in wavelet coefficients.

    Parameters
    ----------
    coeffs : List[np.ndarray]
        List of wavelet coefficient arrays

    Returns
    -------
    int
        Total number of zero crossings
    """
    return int(sum(np.sum(np.diff(np.signbit(coef))) for coef in coeffs))


def _energy_ratio(coeffs: List[np.ndarray]) -> float:
    """
    Calculate ratio of high to low frequency energy.

    Parameters
    ----------
    coeffs : List[np.ndarray]
        List of wavelet coefficient arrays

    Returns
    -------
    float
        Energy ratio
    """
    if len(coeffs) < 2:
        return 0.0
    high_freq_energy = np.sum([np.sum(np.square(coef)) for coef in coeffs[:-1]])
    low_freq_energy = np.sum(np.square(coeffs[-1]))
    return float(high_freq_energy / (low_freq_energy + 1e-10))


def _calculate_cross_domain_features(stft_feats: Dict, wavelet_feats: Dict) -> Dict:
    """
    Calculate features that relate STFT and wavelet domains.

    Parameters
    ----------
    stft_feats : Dict
        STFT features dictionary
    wavelet_feats : Dict
        Wavelet features dictionary

    Returns
    -------
    Dict
        Cross-domain features
    """
    cross_features = {}

    try:
        # Energy ratios between domains
        stft_total = stft_feats.get("stft_total_energy", 0)
        wavelet_total = wavelet_feats.get("wavelet_total_energy", 0)

        if stft_total > 0 and wavelet_total > 0:
            cross_features["stft_to_wavelet_energy_ratio"] = float(
                stft_total / wavelet_total
            )

        # Correlation between energy distributions
        stft_energies = [v for k, v in stft_feats.items() if "energy" in k]
        wavelet_energies = [v for k, v in wavelet_feats.items() if "energy" in k]

        if stft_energies and wavelet_energies:
            min_len = min(len(stft_energies), len(wavelet_energies))
            correlation = np.corrcoef(
                stft_energies[:min_len], wavelet_energies[:min_len]
            )[0, 1]
            cross_features["energy_distribution_correlation"] = float(correlation)

    except Exception as e:
        logger.warning(f"Error in cross-domain feature calculation: {str(e)}")

    return cross_features


def test_feature_extraction():
    """
    Test the feature extraction functions with synthetic ECG data.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic ECG-like signal
    fs = 500  # Sampling frequency in Hz
    duration = 2  # seconds
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)

    # Create a more realistic ECG-like signal with multiple frequency components
    fundamental = 1.0  # Fundamental frequency of the heartbeat
    beat = (
        1.0 * np.sin(2 * np.pi * fundamental * t)  # Basic rhythm
        + 0.5 * np.sin(2 * np.pi * 5 * t)  # QRS complex
        + 0.3 * np.sin(2 * np.pi * 10 * t)  # P wave
        + 0.2 * np.sin(2 * np.pi * 15 * t)  # T wave
    )

    # Add some noise
    noise = 0.1 * np.random.randn(len(t))
    beat_noisy = beat + noise

    try:
        # Test STFT features
        logger.info("Testing STFT feature extraction...")
        stft_features = extract_stft_features(beat_noisy, fs)
        logger.info(f"Extracted {len(stft_features)} STFT features")

        # Test wavelet features
        logger.info("Testing wavelet feature extraction...")
        wavelet_features = extract_wavelet_features(beat_noisy)
        logger.info(f"Extracted {len(wavelet_features)} wavelet features")

        # Test hybrid features
        logger.info("Testing hybrid feature extraction...")
        hybrid_features = extract_hybrid_features(beat_noisy, fs)
        logger.info(f"Extracted {len(hybrid_features)} hybrid features")

        # Visualize results
        plt.figure(figsize=(15, 10))

        # Plot 1: Original and noisy signals
        plt.subplot(3, 1, 1)
        plt.plot(t, beat, "b-", label="Clean ECG", alpha=0.7)
        plt.plot(t, beat_noisy, "r-", label="Noisy ECG", alpha=0.5)
        plt.title("Synthetic ECG Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)

        # Plot 2: STFT Spectrogram
        plt.subplot(3, 1, 2)
        f, t_vals, Zxx = signal.stft(beat_noisy, fs=fs, nperseg=256)
        plt.pcolormesh(t_vals, f, np.abs(Zxx), shading="gouraud")
        plt.title("STFT Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Magnitude")

        # Plot 3: Wavelet Decomposition
        plt.subplot(3, 1, 3)
        coeffs = pywt.wavedec(beat_noisy, "db4", level=4)
        plt.plot(coeffs[0], label="Approximation")
        for i, detail in enumerate(coeffs[1:], 1):
            plt.plot(detail, label=f"Detail {i}")
        plt.title("Wavelet Decomposition")
        plt.xlabel("Sample")
        plt.ylabel("Coefficient Value")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Print feature statistics
        logger.info("\nFeature Statistics:")
        logger.info(f"STFT Features: {', '.join(stft_features.keys())}")
        logger.info(f"Wavelet Features: {', '.join(wavelet_features.keys())}")
        logger.info(
            f"Number of cross-domain features: {len(hybrid_features) - len(stft_features) - len(wavelet_features)}"
        )

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_feature_extraction()
