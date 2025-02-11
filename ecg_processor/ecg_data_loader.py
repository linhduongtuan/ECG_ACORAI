# ecg_data_loader.py

import numpy as np
import pandas as pd
import wfdb
import pyedflib
import logging
import h5py
import scipy.io
from typing import Dict, List
from pathlib import Path


class ECGDataLoader:
    """
    Comprehensive ECG data loader supporting multiple file formats.
    """

    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
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
                "signal": record.p_signal,
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

            # Convert to numpy array and ensure 2D
            signal = df.values
            if signal.ndim == 1:
                signal = signal.reshape(-1, 1)

            return {
                "signal": signal,
                "metadata": {"columns": df.columns.tolist(), "shape": signal.shape},
                "sampling_rate": kwargs.get("sampling_rate"),
            }
        except Exception as e:
            self.logger.error(f"Error loading CSV data: {str(e)}")
            raise

    def _load_txt_data(self, file_path: Path, **kwargs) -> Dict:
        data = np.loadtxt(file_path, delimiter=kwargs.get("delimiter"))
        return {
            "signal": data,
            "metadata": {"shape": data.shape},
            "sampling_rate": kwargs.get("sampling_rate"),
        }

    def _load_matlab_data(self, file_path: Path, **kwargs) -> Dict:
        mat_data = scipy.io.loadmat(file_path)
        signal_vars = [k for k in mat_data.keys() if not k.startswith("__")]
        if not signal_vars:
            raise ValueError("No valid signal data found in .mat file")
        return {
            "signal": mat_data[signal_vars[0]],
            "metadata": {
                "variables": signal_vars,
                "shape": mat_data[signal_vars[0]].shape,
            },
            "sampling_rate": kwargs.get("sampling_rate"),
        }

    def _load_edf_data(self, file_path: Path, **kwargs) -> Dict:
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
                "signal": signals,
                "metadata": {
                    "channel_info": channel_info,
                    "patient_info": f.getPatientCode(),
                    "recording_info": f.getRecordingAdditional(),
                },
                "sampling_rate": f.getSampleFrequency(0),
            }

    def _load_wav_data(self, file_path: Path, **kwargs) -> Dict:
        sample_rate, data = scipy.io.wavfile.read(str(file_path))
        return {
            "signal": data,
            "metadata": {"shape": data.shape, "dtype": data.dtype},
            "sampling_rate": sample_rate,
        }

    def _load_hdf5_data(self, file_path: Path, **kwargs) -> Dict:
        with h5py.File(file_path, "r") as f:
            dataset_names = list(f.keys())
            return {
                "signal": f[dataset_names[0]][:],
                "metadata": {name: f[name][:] for name in dataset_names[1:]},
                "sampling_rate": kwargs.get("sampling_rate"),
            }

    def _load_xyz_data(self, file_path: Path, **kwargs) -> Dict:
        raise NotImplementedError("XYZ format loading not implemented")

    def _load_numbered_format(self, file_path: Path, **kwargs) -> Dict:
        raise NotImplementedError("Numbered format loading not implemented")

    def _validate_data(self, data: Dict) -> None:
        if "signal" not in data:
            raise ValueError("Missing required key in loaded data: signal")
        if not isinstance(data["signal"], np.ndarray) or data["signal"].size == 0:
            raise ValueError("Signal data must be a non-empty numpy array")
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
    loader = ECGDataLoader()
    print("Supported formats:", loader.get_supported_formats())

    try:
        data = loader.load_data("/Users/linh/Downloads/ECGPreprocessor/data/n16.dat")
        print("Signal shape:", data["signal"].shape)
        print("Sampling rate:", data["sampling_rate"])
        print("Metadata:", data["metadata"])
        if data.get("annotations"):
            print("Annotations:", data["annotations"])
    except Exception as e:
        print(f"Error loading data: {e}")

    try:
        data = loader.load_data(
            "/Users/linh/Downloads/ECGPreprocessor/data/3000003_0003.hea"
        )
        print("Signal shape:", data["signal"].shape)
        print("Sampling rate:", data["sampling_rate"])
        print("Metadata:", data["metadata"])
        if data.get("annotations"):
            print("Annotations:", data["annotations"])
    except Exception as e:
        print(f"Error loading data: {e}")
    try:
        data = loader.load_data(
            "/Users/linh/Downloads/ECGPreprocessor/data/3000003_0003.dat"
        )
        print("Signal shape:", data["signal"].shape)
        print("Sampling rate:", data["sampling_rate"])
        print("Metadata:", data["metadata"])
        if data.get("annotations"):
            print("Annotations:", data["annotations"])
    except Exception as e:
        print(f"Error loading data: {e}")

    try:
        data = loader.load_data("data/100.csv", delimiter=",", sampling_rate=500)
        print("CSV data shape:", data["signal"].shape)
        print("CSV sampling rate:", data["sampling_rate"])
        if data.get("annotations"):
            print("CSV Annotations:", data["annotations"])
    except Exception as e:
        print(f"Error loading CSV: {e}")

    try:
        data = loader.load_data("/Users/linh/Downloads/ECGPreprocessor/data/100.hea")
        print("Signal shape:", data["signal"].shape)
        print("Sampling rate:", data["sampling_rate"])
        print("Metadata:", data["metadata"])
        if data.get("annotations"):
            print("Annotations:", data["annotations"])
    except Exception as e:
        print(f"Error loading data: {e}")
    try:
        data = loader.load_data("/Users/linh/Downloads/ECGPreprocessor/data/100.atr")
        print("Signal shape:", data["signal"].shape)
        print("Sampling rate:", data["sampling_rate"])
        print("Metadata:", data["metadata"])
        if data.get("annotations"):
            print("Annotations:", data["annotations"])
    except Exception as e:
        print(f"Error loading data: {e}")
    try:
        data = loader.load_data("/Users/linh/Downloads/ECGPreprocessor/data/100.dat")
        print("Signal shape:", data["signal"].shape)
        print("Sampling rate:", data["sampling_rate"])
        print("Metadata:", data["metadata"])
        if data.get("annotations"):
            print("Annotations:", data["annotations"])
    except Exception as e:
        print(f"Error loading data: {e}")
    try:
        data = loader.load_data("/Users/linh/Downloads/ECGPreprocessor/data/100.qrs")
        print("Signal shape:", data["signal"].shape)
        print("Sampling rate:", data["sampling_rate"])
        print("Metadata:", data["metadata"])
        if data.get("annotations"):
            print("Annotations:", data["annotations"])
    except Exception as e:
        print(f"Error loading data: {e}")
    try:
        data = loader.load_data(
            "/Users/linh/Downloads/ECGPreprocessor/data/100skew.hea"
        )
        print("Signal shape:", data["signal"].shape)
        print("Sampling rate:", data["sampling_rate"])
        print("Metadata:", data["metadata"])
        if data.get("annotations"):
            print("Annotations:", data["annotations"])
    except Exception as e:
        print(f"Error loading data: {e}")
    try:
        data = loader.load_data("/Users/linh/Downloads/ECGPreprocessor/data/12726.anl")
        print("Signal shape:", data["signal"].shape)
        print("Sampling rate:", data["sampling_rate"])
        print("Metadata:", data["metadata"])
        if data.get("annotations"):
            print("Annotations:", data["annotations"])
    except Exception as e:
        print(f"Error loading data: {e}")
    try:
        data = loader.load_data("/Users/linh/Downloads/ECGPreprocessor/data/12726.wabp")
        print("Signal shape:", data["signal"].shape)
        print("Sampling rate:", data["sampling_rate"])
        print("Metadata:", data["metadata"])
        if data.get("annotations"):
            print("Annotations:", data["annotations"])
    except Exception as e:
        print(f"Error loading data: {e}")
    try:
        data = loader.load_data("/Users/linh/Downloads/ECGPreprocessor/data/12726.wqrs")
        print("Signal shape:", data["signal"].shape)
        print("Sampling rate:", data["sampling_rate"])
        print("Metadata:", data["metadata"])
        if data.get("annotations"):
            print("Annotations:", data["annotations"])
    except Exception as e:
        print(f"Error loading data: {e}")

    try:
        data = loader.load_data(
            "/Users/linh/Downloads/ECGPreprocessor/data/03700181.gqrsh"
        )
        print("Signal shape:", data["signal"].shape)
        print("Sampling rate:", data["sampling_rate"])
        print("Metadata:", data["metadata"])
        if data.get("annotations"):
            print("Annotations:", data["annotations"])
    except Exception as e:
        print(f"Error loading data: {e}")
    try:
        data = loader.load_data(
            "/Users/linh/Downloads/ECGPreprocessor/data/03700181.gqrsl"
        )
        print("Signal shape:", data["signal"].shape)
        print("Sampling rate:", data["sampling_rate"])
        print("Metadata:", data["metadata"])
        if data.get("annotations"):
            print("Annotations:", data["annotations"])
    except Exception as e:
        print(f"Error loading data: {e}")
