# tests/conftest.py
import pytest
import neurokit2 as nk


@pytest.fixture
def sample_ecg():
    """Generate a sample ECG signal."""
    sampling_rate = 500
    duration = 10
    return nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, noise=0.1)


@pytest.fixture
def preprocessor():
    """Create an ECG preprocessor instance."""
    from ecg_processor import ECGPreprocessor

    return ECGPreprocessor(sampling_rate=500, debug=True)


@pytest.fixture
def realtime_processor():
    """Create a real-time ECG processor instance."""
    from ecg_processor import RealTimeECGProcessor

    return RealTimeECGProcessor(sampling_rate=500)


def create_sample_ecg(sampling_rate=500, duration=10):
    """Create a sample ECG signal for testing."""
    return nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, noise=0.1)
