import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import wfdb
from ecg_processor.ecg_data_loader import ECGDataLoader


class TestECGDataLoader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.loader = ECGDataLoader()
        self.temp_dir = tempfile.mkdtemp()

        # Create sample data
        self.sample_rate = 250
        t = np.linspace(0, 4, 1000)
        self.sample_data = np.sin(2 * np.pi * 1 * t)
        # Ensure sample data is 2D from the start
        self.sample_data = self.sample_data.reshape(-1, 1)

        # Create temporary test files
        self.create_test_files()

    def tearDown(self):
        """Clean up after each test method."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_test_files(self):
        """Create temporary test files in various formats."""
        # Create WFDB format files
        wfdb.wrsamp(
            "test_ecg",
            fs=self.sample_rate,
            units=["mV"],
            sig_name=["ECG"],
            p_signal=self.sample_data,
            write_dir=self.temp_dir,
        )

        # Create CSV file using pandas to ensure consistent format
        df = pd.DataFrame(self.sample_data, columns=["ECG"])
        df.to_csv(os.path.join(self.temp_dir, "test.csv"), index=False)

        # Create text file
        np.savetxt(os.path.join(self.temp_dir, "test.txt"), self.sample_data)

    def test_load_csv_data(self):
        """Test loading CSV format data."""
        file_path = os.path.join(self.temp_dir, "test.csv")
        result = self.loader.load_data(file_path, sampling_rate=self.sample_rate)

        self.assertIn("signal", result)
        # Compare shapes
        self.assertEqual(
            result["signal"].shape,
            self.sample_data.shape,
            f"Shape mismatch: got {result['signal'].shape}, "
            f"expected {self.sample_data.shape}",
        )
        # Compare values
        np.testing.assert_array_almost_equal(
            result["signal"],
            self.sample_data,
            decimal=5,
            err_msg="Signal values do not match original data",
        )

    def test_load_wfdb_data(self):
        """Test loading WFDB format data."""
        file_path = os.path.join(self.temp_dir, "test_ecg.dat")
        result = self.loader.load_data(file_path)

        self.assertIn("signal", result)
        self.assertIn("metadata", result)
        self.assertIn("sampling_rate", result)
        self.assertTrue(isinstance(result["signal"], np.ndarray))
        self.assertEqual(result["sampling_rate"], self.sample_rate)

    def test_load_invalid_format(self):
        """Test loading an unsupported file format."""
        with self.assertRaises(NotImplementedError):
            self.loader.load_data("invalid.xyz")

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        with self.assertRaises(Exception):
            self.loader.load_data("nonexistent.dat")

    def test_get_supported_formats(self):
        """Test getting list of supported formats."""
        formats = self.loader.get_supported_formats()
        self.assertTrue(isinstance(formats, list))
        self.assertIn(".dat", formats)
        self.assertIn(".csv", formats)

    def test_load_data_with_annotations(self):
        """Test loading data with annotations."""
        # Create test annotation file
        wfdb.wrann(
            "test_ecg",
            "atr",
            np.array([100, 200, 300]),
            ["N", "N", "N"],
            write_dir=self.temp_dir,
        )

        file_path = os.path.join(self.temp_dir, "test_ecg.dat")
        result = self.loader.load_data(file_path)

        self.assertIn("annotations", result)
        self.assertIsNotNone(result["annotations"])

    def test_validate_data(self):
        """Test data validation."""
        # Test with invalid data
        with self.assertRaises(ValueError):
            self.loader._validate_data({})  # Missing required 'signal' key

        # Test with valid data
        valid_data = {
            "signal": np.array([1, 2, 3]),
            "sampling_rate": 250,
            "metadata": {},
        }
        try:
            self.loader._validate_data(valid_data)
        except Exception as e:
            self.fail(f"Validation raised an exception: {str(e)}")


if __name__ == "__main__":
    unittest.main()
