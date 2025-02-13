# ECG_ACORAI Package Documentation

## Overview

The `ECG_ACORAI` package is a comprehensive Python library for processing and analyzing ECG (Electrocardiogram) signals. It provides tools for signal preprocessing, QRS detection, beat segmentation, feature extraction, heart rate variability (HRV) analysis, signal quality assessment, machine learning-based, deep learning-based, and Graph Neural Network-based for heart abnormality classification and future prediction.

The package is modular, with each functionality organized into separate modules for better maintainability and extensibility.

---

## Features

- **Signal Preprocessing:** Apply bandpass filtering, notch filtering, baseline wander removal, and normalization. (More active voice)

- **QRS Detection:** Detect QRS complexes using traditional and advanced algorithms (e.g., Pan-Tompkins). (No change needed, it's already well-phrased)

- **Beat Segmentation:** Segment ECG signals into individual heartbeats. (No change needed)

- **Feature Extraction:** Extract statistical, wavelet, morphological, and frequency-domain features. (No change needed)

- **HRV Analysis:** Calculate time-domain and frequency-domain HRV metrics. (No change needed)
Signal Quality Assessment: Assess signal quality using metrics such as SNR, baseline stability, and noise levels.

- **Machine Learning:** Train and apply classifiers for beat classification. (More active voice and concise)

- **Visualization:** Visualize signals, QRS detection, HRV metrics, and other relevant data. ("And more" can be replaced with "and other relevant data" or similar to be more specific if possible)

- **Train DL and GNN models with mock data:** Train deep learning and graph neural network models on simulated data for classification and time series prediction.

---

## Installation

1. Preinstall `uv` tooling (recommended)

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or, from [PyPI](https://pypi.org/project/uv/):

```bash
pip install uv
```

2. Clone the repository:

```bash
git clone https://github.com/linhduongtuan/ECG_ACORAI.git
cd ECG_ACORAI
```

3. Install the required dependencies:

```bash
uv pip install -r requirements.txt
```


---

## Usage

1. Import the Package
  
```python
from ecg_processor.core import ECGPreprocessor
```

2. Initialize the ECGPreprocessor
  
```python
preprocessor = ECGPreprocessor(sampling_rate=500)
```

3. Load and Process Data

```python
signal = preprocessor.load_data('raw_ecg_data.csv')
results = preprocessor.process_signal(signal)
```

4. Visualize Results

```python
from ecg_processor.visualization import plot_comprehensive_analysis
plot_comprehensive_analysis(results)
```

5. Train a Classifier

```python
beats = results['beats']
labels = [0, 1, 0, 1, ...]  # Example labels
preprocessor.train_classifier(beats, labels)
```

6. Classify Beats

```python
classification_results = preprocessor.classify_beats(beats)
print(classification_results)
```

---

## Modules and Functions

### 1. `core.py`

The main module containing the ECGPreprocessor class.

Key Methods:

- `process_signal(signal: np.ndarray) -> Dict`: Full preprocessing pipeline.
- `train_classifier(beats: List[np.ndarray], labels: np.ndarray) -> bool`: Train a beat classifier.
- `classify_beats(beats: List[np.ndarray]) -> Optional[Dict]`: Classify beats using a trained model.

---

### 2. `features.py`

Handles feature extraction from ECG beats.

Key Functions:

- `extract_statistical_features(beat: np.ndarray) -> Dict`: Extract statistical features.
- `extract_wavelet_features(beat: np.ndarray, wavelet: str = 'db4', level: int = 4) -> Dict`: Extract wavelet features.
- `extract_morphological_features(beat: np.ndarray, fs: float) -> Dict`: Extract morphological features.

---

3. `hrv.py`
Calculates heart rate variability metrics.

Key Functions:

- `calculate_time_domain_hrv(rr_intervals: np.ndarray) -> Dict`: Time-domain HRV metrics.
- `calculate_frequency_domain_hrv(rr_intervals: np.ndarray, fs: float = 4.0) -> Dict`: Frequency-domain HRV metrics.

---

### 4. `quality.py`

Assesses signal quality.

Key Functions:

- `calculate_signal_quality(original: np.ndarray, processed: np.ndarray, fs: float) -> Dict`: Comprehensive signal quality metrics.

---

### 5. `visualization.py`

Provides visualization tools.

Key Functions:

- `plot_signal_comparison(original: np.ndarray, processed: np.ndarray, fs: float, title: str = "ECG Signal Comparison")`: Plot original and processed signals.
- `plot_comprehensive_analysis(results: Dict)`: Plot comprehensive analysis results.

---

### 6. `utils.py`

Utility functions for filtering, normalization, and data loading.

Key Functions:

- `create_bandpass_filter(lowcut: float, highcut: float, fs: float, order: int) -> Tuple[np.ndarray, np.ndarray]`: Create bandpass filter coefficients.
- `create_notch_filter(freq: float, q: float, fs: float) -> Tuple[np.ndarray, np.ndarray]`: Create notch filter coefficients.
- `normalize_signal(signal: np.ndarray) -> np.ndarray`: Normalize signal to range [0, 1].
- `load_data(file_path: str) -> Optional[np.ndarray]`: Load data from various file formats.

---

## Use CLI with mock data, Deep Learning Models, GNN-Based models for classification and timeseries prediction

- **View Help:** to see all options.

```python
python ecg_cli_extended.py --help
```

- **Simulate an ECG Signal (customizing sampling rate, duration, etc.):**

```python
python ecg_cli_extended.py simulate --fs 500 --duration 60 --noise 0.05 --window-size 1000 --forecast-horizon 1000 --num-events 5 --event-duration 200
```

- **Train a Deep Learning Model (for example, the TransformerClassifier with custom hyperparameters and optimizer/scheduler):**

```python
python ecg_cli_extended.py train-dl --model-name TransformerClassifier --num-epochs 10 --batch-size 32 --optimizer-name adam --scheduler-name step --step-size 5 --gamma 0.5 --learning-rate 0.001 --d-model 128 --nhead 8 --transformer-layers 3
```

- **Train a GNN-Based Model (choose between pure GNN and hybrid GNN+LSTM):**

```python
python ecg_cli_extended.py train-gnn --model-name Hybrid_GNN_LSTM --num-epochs 10 --batch-size 32 --optimizer-name adam --scheduler-name none --learning-rate 0.001 --gnn-hidden-channels 32 --lstm-hidden-size 16 --lstm-layers 1
```

- **To improve code organization**, I've created a modular structure for `ecg_cli_extended.py`, separating it into `data_loader.py`, `models.py`, `training.py`, and `main.py` within the `DL_GNN` directory. The CLI commands are unaffected. For more details, let see `DL_GNN/README.md`.

---
## Tutorials (WIP)

I've added a `notebooks` folder with tutorials and test functions/classes for this repository.  Please feel free to contribute any insights you have on the source code!

---

## Testing

Unit tests for all modules to ensure correctness and reliability are provided in the `tests` directory. To run the tests:

```bash
pytest tests/
```

---

## Example Workflow

```python

# Initialize processor
# Load data
signal = processor.load_data("ecg_data.csv")

# Process signal
results = processor.process_signal(signal)

# Visualize results
from ecg_processor.visualization import plot_comprehensive_analysis
plot_comprehensive_analysis(results)

# Train classifier
# Classify beats
```

---

## Future Enhancements

- Add support for additional file formats (e.g., DICOM, EDF).
- Implement deep learning-based QRS detection.
- Extend HRV analysis with non-linear metrics.
- Add more advanced visualization tools.

---

## License

This package is licensed under the MIT License. See the LICENSE file for details.
