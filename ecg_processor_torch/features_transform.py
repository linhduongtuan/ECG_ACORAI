# features_transformer_torch.py

import math
import torch
import torch.nn.functional as F
import numpy as np
import pywt
from scipy import stats
import matplotlib.pyplot as plt
import logging
from typing import Dict, Optional, Tuple, List, Union

logger = logging.getLogger(__name__)


def extract_stft_features(
    beat: Union[np.ndarray, torch.Tensor],
    fs: float,
    nperseg: int = 128,
    noverlap: Optional[int] = None,
    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict:
    """
    Extract features from the Short-Time Fourier Transform (STFT) of an ECG beat,
    using PyTorch’s STFT function.

    Parameters
    ----------
    beat : np.ndarray or torch.Tensor
        1D array/tensor representing the ECG beat.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each segment for STFT (default is 128).
    noverlap : Optional[int], optional
        Number of overlapping samples between segments (default: nperseg//2).
    freq_bands : Optional[Dict[str, Tuple[float, float]]], optional
        Dictionary of frequency bands to analyze.
        Example: {'low': (0, 5), 'medium': (5, 20), 'high': (20, 50)}
        (default bands are used if None)

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
        If input parameters are invalid.
    """
    try:
        # Validate and convert beat to a torch tensor.
        if isinstance(beat, torch.Tensor):
            beat_tensor = beat.float()
        else:
            beat_tensor = torch.tensor(beat, dtype=torch.float32)
        if beat_tensor.dim() != 1:
            raise ValueError("Beat must be a 1D array or tensor")
        if fs <= 0:
            raise ValueError("Sampling frequency must be positive")
        if beat_tensor.size(0) < nperseg:
            raise ValueError(f"Beat length must be at least {nperseg}")

        # Set default overlap if not provided (mimicking scipy.signal.stft default)
        if noverlap is None:
            noverlap = nperseg // 2
        hop_length = nperseg - noverlap

        # Compute STFT using torch.stft.
        window = torch.hann_window(nperseg)
        # The default behavior returns a complex tensor. With onesided=True (default for real input)
        X = torch.stft(
            beat_tensor,
            n_fft=nperseg,
            hop_length=hop_length,
            win_length=nperseg,
            window=window,
            center=True,
            return_complex=True,
        )
        spectrogram = torch.abs(X)  # shape: (n_freq, n_time)

        # Frequency vector (for one-sided STFT)
        f = torch.linspace(0, fs / 2, steps=nperseg // 2 + 1)
        # Energy statistics
        energy_stats = {
            "stft_mean_energy": spectrogram.mean().item(),
            "stft_std_energy": spectrogram.std().item(),
            "stft_max_energy": spectrogram.max().item(),
            "stft_min_energy": spectrogram.min().item(),
            "stft_total_energy": spectrogram.sum().item(),
            "stft_median_energy": spectrogram.median().item(),
        }

        # Frequency band powers
        if freq_bands is None:
            freq_bands = {
                "low": (0, 5),      # 0-5 Hz
                "medium": (5, 20),  # 5-20 Hz
                "high": (20, 50),   # 20-50 Hz
            }

        band_powers = {}
        for band_name, (low_freq, high_freq) in freq_bands.items():
            mask = (f >= low_freq) & (f <= high_freq)  # boolean mask
            # Sum over the masked frequency bins then average over entire spectrogram.
            band_power = torch.sum(spectrogram[mask, :]) / spectrogram.numel()
            band_powers[f"stft_power_{band_name}"] = band_power.item()

        # Spectral shape metrics (using torch-based helper functions)
        shape_metrics = {
            "stft_spectral_centroid": _spectral_centroid(f, spectrogram),
            "stft_spectral_bandwidth": _spectral_bandwidth(f, spectrogram),
            "stft_spectral_rolloff": _spectral_rolloff(spectrogram, percentile=0.85),
            "stft_spectral_flatness": _spectral_flatness(spectrogram),
        }

        features = {**energy_stats, **band_powers, **shape_metrics}
        return features

    except Exception as e:
        logger.error(f"Error extracting STFT features: {str(e)}")
        raise


def _spectral_centroid(freqs: torch.Tensor, magnitude: torch.Tensor) -> float:
    """
    Calculate the spectral centroid (weighted mean of frequencies) using torch.

    Parameters
    ----------
    freqs : torch.Tensor
        1D tensor of frequency values.
    magnitude : torch.Tensor
        2D tensor (n_freq x n_time) of magnitudes.

    Returns
    -------
    float
        Spectral centroid.
    """
    mag_sum = torch.sum(magnitude, dim=1)  # sum over time for each freq bin
    centroid = torch.sum(freqs * mag_sum) / (torch.sum(mag_sum) + 1e-10)
    return centroid.item()


def _spectral_bandwidth(freqs: torch.Tensor, magnitude: torch.Tensor) -> float:
    """
    Calculate the spectral bandwidth (weighted standard deviation of frequencies).

    Parameters
    ----------
    freqs : torch.Tensor
        1D tensor of frequency values.
    magnitude : torch.Tensor
        2D tensor (n_freq x n_time) of magnitudes.

    Returns
    -------
    float
        Spectral bandwidth.
    """
    centroid = _spectral_centroid(freqs, magnitude)
    mag_sum = torch.sum(magnitude, dim=1)
    variance = torch.sum(((freqs - centroid) ** 2) * mag_sum) / (torch.sum(mag_sum) + 1e-10)
    return torch.sqrt(variance).item()


def _spectral_rolloff(magnitude: torch.Tensor, percentile: float = 0.85) -> float:
    """
    Calculate the spectral rolloff (frequency bin index below which a given percentile
    of the total spectral power lies) using torch.

    Parameters
    ----------
    magnitude : torch.Tensor
        2D tensor (n_freq x n_time) of magnitudes.
    percentile : float, optional
        Percentile threshold (default is 0.85).

    Returns
    -------
    float
        Rolloff frequency bin index (as a float).
    """
    power = torch.sum(magnitude**2, dim=1)
    cumulative = torch.cumsum(power, dim=0)
    threshold = percentile * cumulative[-1]
    idx = (cumulative >= threshold).nonzero(as_tuple=False)
    rolloff_idx = idx[0].item() if idx.numel() > 0 else 0
    return float(rolloff_idx)


def _spectral_flatness(magnitude: torch.Tensor) -> float:
    """
    Calculate the spectral flatness (Wiener entropy) using torch.

    Parameters
    ----------
    magnitude : torch.Tensor
        2D tensor (n_freq x n_time) of magnitudes.

    Returns
    -------
    float
        Spectral flatness value.
    """
    power = (magnitude ** 2) + 1e-10
    geo_mean = torch.exp(torch.mean(torch.log(power)))
    arith_mean = torch.mean(power)
    return (geo_mean / arith_mean).item()


def extract_wavelet_features(
    beat: Union[np.ndarray, torch.Tensor], wavelet: str = "db4", level: int = 4
) -> Dict:
    """
    Extract features using wavelet transform from an ECG beat.

    Parameters
    ----------
    beat : np.ndarray or torch.Tensor
        1D array representing the ECG beat.
    wavelet : str, optional
        Wavelet type (default is 'db4').
    level : int, optional
        Decomposition level (default is 4).

    Returns
    -------
    Dict
        Dictionary containing wavelet features.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    try:
        # If beat is a torch tensor, convert to numpy.
        if isinstance(beat, torch.Tensor):
            beat = beat.detach().cpu().numpy()
        if not isinstance(beat, np.ndarray):
            raise ValueError("Beat must be a numpy array")
        if beat.ndim != 1:
            raise ValueError("Beat must be a 1D array")
        if len(beat) < 2**level:
            raise ValueError(
                f"Beat length must be at least {2**level} for level {level} decomposition"
            )
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(beat, wavelet, level=level)
        features = {}
        total_energy = 0
        for i, coef in enumerate(coeffs):
            energy = np.sum(np.square(coef))
            total_energy += energy
            features[f"wavelet_energy_{i}"] = float(energy)
            hist, _ = np.histogram(coef, bins="auto", density=True)
            features[f"wavelet_entropy_{i}"] = float(stats.entropy(hist))
            features[f"wavelet_mean_{i}"] = float(np.mean(coef))
            features[f"wavelet_std_{i}"] = float(np.std(coef))
            features[f"wavelet_max_{i}"] = float(np.max(np.abs(coef)))
            features[f"wavelet_median_{i}"] = float(np.median(np.abs(coef)))
        if total_energy > 0:
            for i in range(len(coeffs)):
                features[f"wavelet_relative_energy_{i}"] = float(
                    features[f"wavelet_energy_{i}"] / total_energy
                )
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
    beat: Union[np.ndarray, torch.Tensor],
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
    beat : np.ndarray or torch.Tensor
        1D array representing the ECG beat.
    fs : float
        Sampling frequency in Hz.
    wavelet : str, optional
        Wavelet type (default is 'db4').
    level : int, optional
        Decomposition level (default is 4).
    nperseg : int, optional
        STFT segment length (default is 128).
    noverlap : Optional[int], optional
        Overlap for STFT segments, by default None.
    freq_bands : Optional[Dict[str, Tuple[float, float]]], optional
        Frequency bands for STFT analysis (default bands are used if None).

    Returns
    -------
    Dict
        Dictionary containing combined features.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    try:
        if not isinstance(beat, np.ndarray) and not isinstance(beat, torch.Tensor):
            raise ValueError("Beat must be a numpy array or tensor")
        if (isinstance(beat, np.ndarray) and beat.ndim != 1) or (isinstance(beat, torch.Tensor) and beat.dim() != 1):
            raise ValueError("Beat must be a 1D array or tensor")
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("Sampling frequency must be positive")

        stft_feats = extract_stft_features(beat, fs, nperseg, noverlap, freq_bands)
        wavelet_feats = extract_wavelet_features(beat, wavelet, level)
        hybrid_features = {**stft_feats, **wavelet_feats}
        try:
            cross_domain = _calculate_cross_domain_features(stft_feats, wavelet_feats)
            hybrid_features.update(cross_domain)
        except Exception as e:
            logger.warning(f"Failed to calculate cross-domain features: {str(e)}")
        return hybrid_features

    except Exception as e:
        logger.error(f"Error extracting hybrid features: {str(e)}")
        raise


# Helper functions for wavelet features and cross-domain metrics remain numpy-based.
def _count_zero_crossings(coeffs: List[np.ndarray]) -> int:
    return int(sum(np.sum(np.diff(np.signbit(coef))) for coef in coeffs))


def _energy_ratio(coeffs: List[np.ndarray]) -> float:
    if len(coeffs) < 2:
        return 0.0
    high_freq_energy = np.sum([np.sum(np.square(coef)) for coef in coeffs[:-1]])
    low_freq_energy = np.sum(np.square(coeffs[-1]))
    return float(high_freq_energy / (low_freq_energy + 1e-10))


def _calculate_cross_domain_features(stft_feats: Dict, wavelet_feats: Dict) -> Dict:
    cross_features = {}
    try:
        stft_total = stft_feats.get("stft_total_energy", 0)
        wavelet_total = wavelet_feats.get("wavelet_total_energy", 0)
        if stft_total > 0 and wavelet_total > 0:
            cross_features["stft_to_wavelet_energy_ratio"] = float(stft_total / wavelet_total)
        stft_energies = [v for k, v in stft_feats.items() if "energy" in k]
        wavelet_energies = [v for k, v in wavelet_feats.items() if "energy" in k]
        if stft_energies and wavelet_energies:
            min_len = min(len(stft_energies), len(wavelet_energies))
            correlation = np.corrcoef(stft_energies[:min_len], wavelet_energies[:min_len])[0, 1]
            cross_features["energy_distribution_correlation"] = float(correlation)
    except Exception as e:
        logger.warning(f"Error in cross-domain feature calculation: {str(e)}")
    return cross_features


def test_feature_extraction():
    """
    Test the feature extraction functions with synthetic ECG data.
    """
    logging.basicConfig(level=logging.INFO)
    fs = 500  # Sampling frequency in Hz
    duration = 2  # seconds
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    # Create a synthetic ECG-like signal with multiple frequency components.
    fundamental = 1.0
    beat = (
        1.0 * np.sin(2 * np.pi * fundamental * t) +      # Basic rhythm
        0.5 * np.sin(2 * np.pi * 5 * t) +                  # QRS complex
        0.3 * np.sin(2 * np.pi * 10 * t) +                 # P wave
        0.2 * np.sin(2 * np.pi * 15 * t)                   # T wave
    )
    noise = 0.1 * np.random.randn(len(t))
    beat_noisy = beat + noise

    try:
        logger.info("Testing STFT feature extraction...")
        stft_features = extract_stft_features(beat_noisy, fs)
        logger.info(f"Extracted {len(stft_features)} STFT features")

        logger.info("Testing wavelet feature extraction...")
        wavelet_features = extract_wavelet_features(beat_noisy)
        logger.info(f"Extracted {len(wavelet_features)} wavelet features")

        logger.info("Testing hybrid feature extraction...")
        hybrid_features = extract_hybrid_features(beat_noisy, fs)
        logger.info(f"Extracted {len(hybrid_features)} hybrid features")

        # For visualization, compute a STFT spectrogram with torch and convert to numpy.
        nperseg = 256
        noverlap = nperseg // 2
        hop_length = nperseg - noverlap
        window = torch.hann_window(nperseg)
        beat_tensor = torch.tensor(beat_noisy, dtype=torch.float32)
        X = torch.stft(
            beat_tensor,
            n_fft=nperseg,
            hop_length=hop_length,
            win_length=nperseg,
            window=window,
            center=True,
            return_complex=True,
        )
        Zxx = torch.abs(X).detach().cpu().numpy()
        freqs = np.linspace(0, fs / 2, num=nperseg // 2 + 1)
        time_vals = np.arange(Zxx.shape[1]) * hop_length / fs

        plt.figure(figsize=(15, 10))
        plt.subplot(3, 1, 1)
        plt.plot(t, beat, "b-", label="Clean ECG", alpha=0.7)
        plt.plot(t, beat_noisy, "r-", label="Noisy ECG", alpha=0.5)
        plt.title("Synthetic ECG Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.pcolormesh(time_vals, freqs, Zxx, shading="gouraud")
        plt.title("STFT Spectrogram (PyTorch)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Magnitude")

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

        logger.info("\nFeature Statistics:")
        logger.info(f"STFT Features: {', '.join(stft_features.keys())}")
        logger.info(f"Wavelet Features: {', '.join(wavelet_features.keys())}")
        cross_domain_count = len(hybrid_features) - len(stft_features) - len(wavelet_features)
        logger.info(f"Number of cross-domain features: {cross_domain_count}")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_feature_extraction()