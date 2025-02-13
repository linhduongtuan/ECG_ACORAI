# realtime_torch.py

import collections
import logging
import numpy as np
import torch
from scipy import signal
from .exceptions import ProcessingError
from .ecg_preprocessor import ECGPreprocessor

logger = logging.getLogger(__name__)


def torch_lfilter(b: torch.Tensor, a: torch.Tensor, x: torch.Tensor, zi: torch.Tensor):
    """
    A simple IIR filter implementation in PyTorch.

    This function processes the input tensor x sample‐by‐sample with the difference
    equation defined by coefficients `b` and `a` (assumes a[0] == 1). The initial state
    is provided by zi and the function returns the filtered output and updated state.

    Parameters
    ----------
    b : torch.Tensor
         1D tensor of numerator coefficients.
    a : torch.Tensor
         1D tensor of denominator coefficients (with a[0]==1).
    x : torch.Tensor
         Input 1D tensor.
    zi : torch.Tensor
         Initial state (shape = (len(b)-1,)).

    Returns
    -------
    y : torch.Tensor
         Filtered output (same shape as x).
    zf : torch.Tensor
         Final state (shape = (len(b)-1,)).
    """
    M = b.numel()
    if M == 1:
        # FIR filter of order 0
        y = b[0] * x
        return y, torch.zeros(0, dtype=x.dtype, device=x.device)

    y = torch.empty_like(x)
    # Clone the state to avoid modifying the input state
    z = zi.clone()
    # For each input sample, update output and state recursively.
    for n in range(x.numel()):
        # Compute output
        y[n] = b[0] * x[n] + z[0]
        # Update state for i=1,...,M-1: according to standard lfilter recurrence.
        for i in range(1, M):
            if i < M - 1:
                z[i - 1] = z[i] + b[i] * x[n] - a[i] * y[n]
            else:
                z[i - 1] = b[i] * x[n] - a[i] * y[n]
    return y, z


class RealTimeECGProcessor:
    """Real-time ECG signal processor with streaming capabilities (using PyTorch)."""

    def __init__(
        self, sampling_rate: int = 250, buffer_size: int = 2000, overlap: int = 500
    ):
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.overlap = overlap

        # Initialize buffers (store raw samples as Python floats)
        self.signal_buffer = collections.deque(maxlen=buffer_size)
        self.feature_buffer = collections.deque(maxlen=100)
        self.quality_buffer = collections.deque(maxlen=100)

        # Initialize preprocessor with the given sampling rate.
        self.preprocessor = ECGPreprocessor(sampling_rate=sampling_rate)

        # Initialize online filters.
        self.initialize_filters()

        # Set a quality threshold for further processing.
        self.quality_threshold = 0.6

    def initialize_filters(self):
        """Initialize the bandpass and notch filters for online processing."""
        nyq = self.sampling_rate / 2

        # Design filters using SciPy.
        bp_b, bp_a = signal.butter(3, [0.5 / nyq, 40 / nyq], btype="band")
        notch_b, notch_a = signal.iirnotch(50 / nyq, 30)

        # Convert coefficients to torch tensors (float64 for precision).
        self.bp_b = torch.tensor(bp_b, dtype=torch.float64)
        self.bp_a = torch.tensor(bp_a, dtype=torch.float64)
        self.notch_b = torch.tensor(notch_b, dtype=torch.float64)
        self.notch_a = torch.tensor(notch_a, dtype=torch.float64)

        # Compute initial filter states using SciPy and convert to torch.
        bp_zi = signal.lfilter_zi(bp_b, bp_a)
        notch_zi = signal.lfilter_zi(notch_b, notch_a)
        self.bp_state = torch.tensor(bp_zi, dtype=torch.float64)
        self.notch_state = torch.tensor(notch_zi, dtype=torch.float64)

    def process_sample(self, sample: float) -> dict:
        """
        Process a single ECG sample in real-time.

        The sample is added to an internal buffer. When the buffer is full,
        the method applies a bandpass filter followed by a notch filter using a
        PyTorch-based lfilter implementation. The filtered window is then passed
        to the ECGPreprocessor. After processing, the buffer is updated so that only
        the intended overlap remains.

        Parameters
        ----------
        sample : float
            A new ECG sample.

        Returns
        -------
        dict
            Dictionary result from ECGPreprocessor updated to include the key 'processed_signal'.
            If the buffer is not yet full, an empty dict is returned.
        """
        try:
            # Append new sample to the buffer.
            self.signal_buffer.append(sample)

            # Process only when the buffer is full.
            if len(self.signal_buffer) < self.buffer_size:
                return {}

            # Convert the current buffer to a torch tensor (1D, float64).
            window_data = torch.tensor(list(self.signal_buffer), dtype=torch.float64)

            # Apply the bandpass filter using torch_lfilter.
            # Scale initial state by the first sample (as in original code).
            bp_zi_scaled = self.bp_state * window_data[0]
            filtered, self.bp_state = torch_lfilter(
                self.bp_b, self.bp_a, window_data, bp_zi_scaled
            )

            # Apply the notch filter.
            notch_zi_scaled = self.notch_state * filtered[0]
            filtered, self.notch_state = torch_lfilter(
                self.notch_b, self.notch_a, filtered, notch_zi_scaled
            )

            # Process the filtered signal via the preprocessor.
            # (Assuming ECGPreprocessor.process_signal accepts a numpy array.)
            filtered_np = filtered.detach().cpu().numpy()
            result = self.preprocessor.process_signal(filtered_np)

            # Remove samples from the buffer so that only 'overlap' samples remain.
            samples_to_remove = self.buffer_size - self.overlap
            for _ in range(samples_to_remove):
                self.signal_buffer.popleft()

            # Ensure result is a dict and include the processed signal.
            if isinstance(result, dict):
                result["processed_signal"] = filtered_np
            else:
                result = {"processed_signal": filtered_np}
            return result

        except Exception as e:
            logger.exception("Error in process_sample:")
            raise ProcessingError(f"Error processing sample: {str(e)}")
