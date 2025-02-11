# ecg_processor/realtime.py
import numpy as np
import collections
import logging
from scipy import signal
from .exceptions import ProcessingError
from .ecg_preprocessor import ECGPreprocessor

logger = logging.getLogger(__name__)


class RealTimeECGProcessor:
    """Real-time ECG signal processor with streaming capabilities."""

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

        # Initialize preprocessor with the given sampling rate.
        self.preprocessor = ECGPreprocessor(sampling_rate=sampling_rate)

        # Initialize online filters.
        self.initialize_filters()

        # Set a quality threshold for further processing
        self.quality_threshold = 0.6

    def initialize_filters(self):
        """Initialize the bandpass and notch filters for online processing."""
        nyq = self.sampling_rate / 2
        self.bp_b, self.bp_a = signal.butter(3, [0.5 / nyq, 40 / nyq], btype="band")
        self.notch_b, self.notch_a = signal.iirnotch(50 / nyq, 30)
        self.bp_state = signal.lfilter_zi(self.bp_b, self.bp_a)
        self.notch_state = signal.lfilter_zi(self.notch_b, self.notch_a)

    def process_sample(self, sample: float) -> dict:
        """
        Process a single ECG sample in real-time.

        This method adds the new sample to an internal buffer. Once the buffer is full,
        it applies the bandpass and notch filters, passes the filtered window to the ECGPreprocessor,
        and then removes a number of samples so that only the intended overlap remains in the buffer.
        The final result is updated to include the filtered signal under the key 'processed_signal'.

        Parameters
        ----------
        sample : float
            A new ECG sample.

        Returns
        -------
        dict
            The result from the ECGPreprocessor, updated to include the 'processed_signal'.
            If the buffer is not yet full, returns an empty dict.
        """
        try:
            # Add the new sample to the buffer.
            self.signal_buffer.append(sample)

            # Only trigger processing when the buffer is full.
            if len(self.signal_buffer) < self.buffer_size:
                return {}

            # Convert the current buffer to a numpy array.
            window_data = np.array(list(self.signal_buffer))

            # Apply the bandpass filter.
            filtered, self.bp_state = signal.lfilter(
                self.bp_b, self.bp_a, window_data, zi=self.bp_state * window_data[0]
            )

            # Apply the notch filter.
            filtered, self.notch_state = signal.lfilter(
                self.notch_b, self.notch_a, filtered, zi=self.notch_state * filtered[0]
            )

            # Process the filtered window using the preprocessor.
            result = self.preprocessor.process_signal(filtered)

            # Remove samples so that 'overlap' samples remain in the buffer.
            samples_to_remove = self.buffer_size - self.overlap
            for _ in range(samples_to_remove):
                self.signal_buffer.popleft()

            # Ensure the result contains the key 'processed_signal'.
            if isinstance(result, dict):
                result["processed_signal"] = filtered
            else:
                result = {"processed_signal": filtered}

            return result

        except Exception as e:
            logger.exception("Error in process_sample:")
            raise ProcessingError(f"Error processing sample: {str(e)}")
