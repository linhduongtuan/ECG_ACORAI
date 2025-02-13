# ecg_data_loader_torch.py

import numpy as np
import pandas as pd
import wfdb
import pyedflib
import logging
import h5py
import scipy.io
import scipy.io.wavfile
from typing import Dict, List
from pathlib import Path
import torch


# Helper function: if an array is a numpy array, converts it to a torch.Tensor on the desired device.
def _to_tensor(arr, device):
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).to(device)
    return arr


class ECGDataLoader:
    """
    Comprehensive ECG data loader supporting multiple file formats.
    Returns the 'signal' field as a torch.Tensor.
    """

    def __init__(self, base_path: str = None, device: str = "cpu"):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.format_handlers = {
            ".dat": self._load_wfdb_data,
            ".hea": self._load_wfdb_data,
            ".atr": self._load_wfdb_annotations,
            ".qrs": self._load_wfdb_annotations,
            ".csv": self._load_csv_data,
            ".txt": self._load_txt_data,
            ".mat": self._load_matlab_data,
            ".edf": self._load_edf_data,
            ".wav": self._load_wav_data,
            ".h5": self._load_hdf5_data,
            ".hdf5": self._load_hdf5_data,
            ".xyz": self._load_xyz_data,
        }

        # Add MIT-BIH specific formats
        for fmt in [".wqrs", ".gqrsh", ".gqrsl", ".sqrs"]:
            self.format_handlers[fmt] = self._load_wfdb_annotations

        # Add numbered formats (d0-d9)
        for i in range(10):
            self.format_handlers[f".d{i}"] = self._load_numbered_format

    def load_data(self, file_path: str, **kwargs) -> Dict:
        try:
            file_path = Path(file_path)
            suffix = file_path.suffix.lower()

            if suffix not in self.format_handlers:
                raise ValueError(f"Unsupported file format: {suffix}")

            result = self.format_handlers[suffix](file_path, **kwargs)
            self._validate_data(result)
            return result

        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            raise

    def _load_wfdb_data(self, file_path: Path, **kwargs) -> Dict:
        try:
            record_name = str((self.base_path / file_path).with_suffix(""))
            record = wfdb.rdrecord(record_name)

            annotations = None
            try:
                annotations = wfdb.rdann(record_name, "atr")
            except FileNotFoundError:
                self.logger.info(f"No annotation file found for {file_path}")
            except Exception as e:
                self.logger.warning(
                    f"Error loading annotations for {file_path}: {str(e)}"
                )

            return {
                "signal": _to_tensor(record.p_signal, self.device),
                "metadata": {
                    "units": record.units,
                    "signal_names": record.sig_name,
                    "n_sig": record.n_sig,
                    "fs": record.fs,
                    "baseline": record.baseline,
                    "gain": record.adc_gain,
                    "patient_info": getattr(record, "comments", None),
                },
                "annotations": annotations.__dict__ if annotations else None,
                "sampling_rate": record.fs,
            }
        except Exception as e:
            self.logger.error(f"Error in _load_wfdb_data for {file_path}: {str(e)}")
            raise

    def _load_wfdb_annotations(self, file_path: Path, **kwargs) -> Dict:
        try:
            record_name = str(file_path.with_suffix(""))
            extension = file_path.suffix[1:]
            ann = wfdb.rdann(record_name, extension)
            return {
                "annotations": {
                    attr: getattr(ann, attr)
                    for attr in [
                        "sample",
                        "symbol",
                        "subtype",
                        "chan",
                        "num",
                        "aux_note",
                    ]
                }
            }
        except Exception as e:
            self.logger.error(f"Error loading annotations: {str(e)}")
            raise

    def _load_csv_data(self, file_path: Path, **kwargs) -> Dict:
        """Load data from CSV file."""
        try:
            # Read CSV using pandas
            df = pd.read_csv(file_path, delimiter=kwargs.get("delimiter", ","))
            signal = df.values
            if signal.ndim == 1:
                signal = signal.reshape(-1, 1)

            return {
                "signal": _to_tensor(signal, self.device),
                "metadata": {"columns": df.columns.tolist(), "shape": signal.shape},
                "sampling_rate": kwargs.get("sampling_rate"),
            }
        except Exception as e:
            self.logger.error(f"Error loading CSV data: {str(e)}")
            raise

    def _load_txt_data(self, file_path: Path, **kwargs) -> Dict:
        try:
            data = np.loadtxt(file_path, delimiter=kwargs.get("delimiter"))
            return {
                "signal": _to_tensor(data, self.device),
                "metadata": {"shape": data.shape},
                "sampling_rate": kwargs.get("sampling_rate"),
            }
        except Exception as e:
            self.logger.error(f"Error loading TXT data: {str(e)}")
            raise

    def _load_matlab_data(self, file_path: Path, **kwargs) -> Dict:
        try:
            mat_data = scipy.io.loadmat(file_path)
            signal_vars = [k for k in mat_data.keys() if not k.startswith("__")]
            if not signal_vars:
                raise ValueError("No valid signal data found in .mat file")
            signal = mat_data[signal_vars[0]]
            return {
                "signal": _to_tensor(signal, self.device),
                "metadata": {
                    "variables": signal_vars,
                    "shape": signal.shape,
                },
                "sampling_rate": kwargs.get("sampling_rate"),
            }
        except Exception as e:
            self.logger.error(f"Error loading Matlab data: {str(e)}")
            raise

    def _load_edf_data(self, file_path: Path, **kwargs) -> Dict:
        try:
            with pyedflib.EdfReader(str(self.base_path / file_path)) as f:
                n_channels = f.signals_in_file
                signals = np.array([f.readSignal(i) for i in range(n_channels)])
                channel_info = {
                    "labels": f.getSignalLabels(),
                    "sample_rates": f.getSampleFrequencies(),
                    "physical_max": f.getPhysicalMaximum(),
                    "physical_min": f.getPhysicalMinimum(),
                    "digital_max": f.getDigitalMaximum(),
                    "digital_min": f.getDigitalMinimum(),
                    "prefilter": f.getPrefilter(),
                    "transducer": f.getTransducer(),
                }
                return {
                    "signal": _to_tensor(signals, self.device),
                    "metadata": {
                        "channel_info": channel_info,
                        "patient_info": f.getPatientCode(),
                        "recording_info": f.getRecordingAdditional(),
                    },
                    "sampling_rate": f.getSampleFrequency(0),
                }
        except Exception as e:
            self.logger.error(f"Error loading EDF data: {str(e)}")
            raise

    def _load_wav_data(self, file_path: Path, **kwargs) -> Dict:
        try:
            sample_rate, data = scipy.io.wavfile.read(str(file_path))
            return {
                "signal": _to_tensor(data, self.device),
                "metadata": {"shape": data.shape, "dtype": data.dtype},
                "sampling_rate": sample_rate,
            }
        except Exception as e:
            self.logger.error(f"Error loading WAV data: {str(e)}")
            raise

    def _load_hdf5_data(self, file_path: Path, **kwargs) -> Dict:
        try:
            with h5py.File(file_path, "r") as f:
                dataset_names = list(f.keys())
                signal_np = f[dataset_names[0]][:]
                metadata = {name: f[name][:] for name in dataset_names[1:]}
                return {
                    "signal": _to_tensor(signal_np, self.device),
                    "metadata": metadata,
                    "sampling_rate": kwargs.get("sampling_rate"),
                }
        except Exception as e:
            self.logger.error(f"Error loading HDF5 data: {str(e)}")
            raise

    def _load_xyz_data(self, file_path: Path, **kwargs) -> Dict:
        raise NotImplementedError("XYZ format loading not implemented")

    def _load_numbered_format(self, file_path: Path, **kwargs) -> Dict:
        raise NotImplementedError("Numbered format loading not implemented")

    def _validate_data(self, data: Dict) -> None:
        if "signal" not in data:
            raise ValueError("Missing required key in loaded data: signal")
        if not isinstance(data["signal"], torch.Tensor) or data["signal"].numel() == 0:
            raise ValueError("Signal data must be a non-empty torch.Tensor")
        if "sampling_rate" in data and data["sampling_rate"] is not None:
            if (
                not isinstance(data["sampling_rate"], (int, float))
                or data["sampling_rate"] <= 0
            ):
                raise ValueError("Sampling rate must be a positive number")
        if "metadata" in data and data["metadata"] is not None:
            if not isinstance(data["metadata"], dict):
                raise TypeError("Metadata must be a dictionary")

    def get_supported_formats(self) -> List[str]:
        return list(self.format_handlers.keys())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    loader = ECGDataLoader(device="cpu")  # or change to "cuda" if available
    print("Supported formats:", loader.get_supported_formats())

    # Example usage with one file (update the path as needed)
    try:
        data = loader.load_data("/Users/linh/Downloads/ECGPreprocessor/data/n16.dat")
        print("Signal shape:", data["signal"].shape)
        print("Sampling rate:", data["sampling_rate"])
        print("Metadata:", data["metadata"])
        if data.get("annotations"):
            print("Annotations:", data["annotations"])
    except Exception as e:
        print(f"Error loading data: {e}")
