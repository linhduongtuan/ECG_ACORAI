# # tests/test_ecg_denoiser.py

# import unittest
# import numpy as np
# import torch
# import tempfile
# import os
# from pathlib import Path
# import neurokit2 as nk
# from ecg_processor.ecg_deep_denoiser import (
#     pad_to_multiple,
#     crop_to_length,
#     ConvAutoencoder,
#     ECGDeepDenoiser,
#     ECGAutoencoder,
#     ECGAnomalyDetector,
#     DenoiserError,
#     DenoiserInputError,
#     DenoiserConfigError
# )

# class TestUtilityFunctions(unittest.TestCase):
#     """Test utility functions for signal processing."""

#     def setUp(self):
#         """Set up test data."""
#         self.signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

#     def test_pad_to_multiple(self):
#         """Test padding function."""
#         # Test normal case
#         padded, pad_length = pad_to_multiple(self.signal, 4)
#         self.assertEqual(len(padded), 8)
#         self.assertEqual(pad_length, 3)

#         # Test when no padding needed
#         signal = np.array([1.0, 2.0, 3.0, 4.0])
#         padded, pad_length = pad_to_multiple(signal, 4)
#         self.assertEqual(len(padded), 4)
#         self.assertEqual(pad_length, 0)

#         # Test invalid inputs
#         with self.assertRaises(DenoiserInputError):
#             pad_to_multiple([1, 2, 3], 4)  # Not numpy array
#         with self.assertRaises(DenoiserInputError):
#             pad_to_multiple(self.signal, -1)  # Invalid multiple

#     def test_crop_to_length(self):
#         """Test cropping function."""
#         # Test normal case
#         cropped = crop_to_length(self.signal, 3)
#         self.assertEqual(len(cropped), 3)
#         np.testing.assert_array_equal(cropped, np.array([1.0, 2.0, 3.0]))

#         # Test invalid inputs
#         with self.assertRaises(DenoiserInputError):
#             crop_to_length(self.signal, 6)  # Length too long
#         with self.assertRaises(DenoiserInputError):
#             crop_to_length([1, 2, 3], 2)  # Not numpy array

# class TestConvAutoencoder(unittest.TestCase):
#     """Test the ConvAutoencoder model."""

#     def setUp(self):
#         """Set up test model and data."""
#         self.model = ConvAutoencoder(dropout_rate=0.1)
#         self.batch_size = 4
#         self.seq_length = 128
#         self.input_tensor = torch.randn(self.batch_size, 1, self.seq_length)

#     def test_model_architecture(self):
#         """Test model structure and forward pass."""
#         # Test forward pass
#         output = self.model(self.input_tensor)
#         self.assertEqual(output.shape, self.input_tensor.shape)

#         # Test model parameters
#         self.assertTrue(any(p.requires_grad for p in self.model.parameters()))

#     def test_invalid_input(self):
#         """Test model behavior with invalid inputs."""
#         # Test wrong dimensions
#         invalid_input = torch.randn(self.batch_size, self.seq_length)
#         with self.assertRaises(DenoiserInputError):
#             self.model(invalid_input)

# class TestECGDeepDenoiser(unittest.TestCase):
#     """Test the ECGDeepDenoiser class."""

#     def setUp(self):
#         """Set up test denoiser and data."""
#         self.input_length = 1000
#         self.denoiser = ECGDeepDenoiser(input_length=self.input_length)

#         # Generate synthetic ECG data
#         self.n_samples = 10
#         self.clean_signals = np.array([
#             nk.ecg_simulate(duration=2, sampling_rate=500)[:self.input_length]
#             for _ in range(self.n_samples)
#         ])
#         self.noisy_signals = self.clean_signals + 0.1 * np.random.randn(*self.clean_signals.shape)

#     def test_initialization(self):
#         """Test denoiser initialization."""
#         self.assertIsNotNone(self.denoiser.model)
#         self.assertEqual(self.denoiser.input_length, self.input_length)

#         # Test invalid initialization
#         with self.assertRaises(DenoiserConfigError):
#             ECGDeepDenoiser(input_length=-1)

#     def test_training(self):
#         """Test training process."""
#         history = self.denoiser.train(
#             x_train=self.noisy_signals,
#             epochs=2,
#             batch_size=2,
#             validation_split=0.2
#         )

#         self.assertIn('train_loss', history)
#         self.assertIn('val_loss', history)
#         self.assertTrue(len(history['train_loss']) > 0)

#     def test_denoising(self):
#         """Test denoising functionality."""
#         # Test single signal
#         denoised_signal = self.denoiser.denoise(self.noisy_signals[0])
#         self.assertEqual(len(denoised_signal), self.input_length)

#         # Test batch of signals
#         denoised_batch = self.denoiser.denoise(self.noisy_signals[:2])
#         self.assertEqual(denoised_batch.shape, (2, self.input_length))

# class TestECGAnomalyDetector(unittest.TestCase):
#     """Test the ECGAnomalyDetector class."""

#     def setUp(self):
#         """Set up test detector and data."""
#         self.input_size = 512
#         self.detector = ECGAnomalyDetector(input_size=self.input_size)

#         # Generate synthetic normal and anomalous data
#         self.n_samples = 20
#         self.normal_data = np.random.randn(self.n_samples, self.input_size)
#         self.anomalous_data = np.random.randn(5, self.input_size) * 3  # Different distribution

#     def test_training(self):
#         """Test anomaly detector training."""
#         history = self.detector.fit(
#             self.normal_data,
#             epochs=2,
#             batch_size=4
#         )

#         self.assertIn('loss', history)
#         self.assertTrue(len(history['loss']) > 0)

#     def test_prediction(self):
#         """Test anomaly detection."""
#         # Train the detector
#         self.detector.fit(self.normal_data, epochs=2)

#         # Test prediction
#         anomalies, scores = self.detector.predict(self.anomalous_data)
#         self.assertEqual(len(anomalies), len(self.anomalous_data))
#         self.assertEqual(len(scores), len(self.anomalous_data))

#     def test_model_save_load(self):
#         """Test model saving and loading."""
#         try:
#             # Train the model
#             self.detector.fit(self.normal_data, epochs=2)

#             # Create temporary directory for model
#             with tempfile.TemporaryDirectory() as temp_dir:
#                 model_path = os.path.join(temp_dir, 'model.pt')

#                 # Save model
#                 self.detector.save_model(model_path)
#                 self.assertTrue(os.path.exists(model_path))

#                 # Create new detector and load model
#                 new_detector = ECGAnomalyDetector(input_size=self.input_size)
#                 new_detector.load_model(model_path)

#                 # Test prediction with both models
#                 test_data = self.normal_data[:5]
#                 anomalies1, scores1 = self.detector.predict(test_data)
#                 anomalies2, scores2 = new_detector.predict(test_data)

#                 # Compare results
#                 np.testing.assert_array_equal(anomalies1, anomalies2)
#                 np.testing.assert_array_almost_equal(scores1, scores2, decimal=5)

#         except Exception as e:
#             self.fail(f"Test failed with error: {str(e)}")

# def run_tests():
#     """Run all tests."""
#     # Create test suite
#     suite = unittest.TestSuite()
#     suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUtilityFunctions))
#     suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConvAutoencoder))
#     suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestECGDeepDenoiser))
#     suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestECGAnomalyDetector))

#     # Run tests
#     runner = unittest.TextTestRunner(verbosity=2)
#     runner.run(suite)

# if __name__ == '__main__':
#     run_tests()


# test_ecg_deep_denoiser.py
import unittest
import numpy as np
import torch
from ecg_processor.ecg_deep_denoiser import (
    pad_to_multiple,
    crop_to_length,
    ConvAutoencoder,
    ECGDeepDenoiser,
    ECGAutoencoder,
    ECGAnomalyDetector,
    DenoiserInputError,
    DenoiserConfigError,
)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def test_pad_to_multiple(self):
        """Test padding function."""
        # Test valid input
        signal = np.array([1, 2, 3, 4, 5])
        padded, pad_length = pad_to_multiple(signal, 4)
        self.assertEqual(len(padded), 8)
        self.assertEqual(pad_length, 3)

        # Test signal already multiple of n
        signal = np.array([1, 2, 3, 4])
        padded, pad_length = pad_to_multiple(signal, 4)
        self.assertEqual(len(padded), 4)
        self.assertEqual(pad_length, 0)

        # Test invalid inputs
        with self.assertRaises(DenoiserInputError):
            pad_to_multiple([1, 2, 3], 4)  # Not numpy array
        with self.assertRaises(DenoiserInputError):
            pad_to_multiple(np.array([1, 2, 3]), -1)  # Invalid multiple

    def test_crop_to_length(self):
        """Test cropping function."""
        # Test valid input
        signal = np.array([1, 2, 3, 4, 5])
        cropped = crop_to_length(signal, 3)
        self.assertEqual(len(cropped), 3)
        np.testing.assert_array_equal(cropped, np.array([1, 2, 3]))

        # Test invalid inputs
        with self.assertRaises(DenoiserInputError):
            crop_to_length([1, 2, 3], 2)  # Not numpy array
        with self.assertRaises(DenoiserInputError):
            crop_to_length(np.array([1, 2, 3]), 4)  # Length too long


class TestConvAutoencoder(unittest.TestCase):
    """Test cases for ConvAutoencoder."""

    def setUp(self):
        self.model = ConvAutoencoder(dropout_rate=0.1)
        self.batch_size = 2
        self.seq_length = 1000

    def test_init(self):
        """Test initialization."""
        self.assertIsInstance(self.model, ConvAutoencoder)
        with self.assertRaises(DenoiserConfigError):
            ConvAutoencoder(dropout_rate=1.5)  # Invalid dropout rate

    def test_forward(self):
        """Test forward pass."""
        x = torch.randn(self.batch_size, 1, self.seq_length)
        output = self.model(x)
        self.assertEqual(output.shape, x.shape)

        # Test invalid inputs
        with self.assertRaises(DenoiserInputError):
            self.model(
                torch.randn(self.batch_size, self.seq_length)
            )  # Wrong dimensions


class TestECGDeepDenoiser(unittest.TestCase):
    """Test cases for ECGDeepDenoiser."""

    def setUp(self):
        self.input_length = 1000
        self.denoiser = ECGDeepDenoiser(input_length=self.input_length)

    def test_init(self):
        """Test initialization."""
        self.assertIsInstance(self.denoiser, ECGDeepDenoiser)
        with self.assertRaises(DenoiserConfigError):
            ECGDeepDenoiser(input_length=-1)  # Invalid input length

    def test_train(self):
        """Test training process."""
        # Generate dummy data
        x_train = np.random.randn(10, self.input_length)

        # Test training
        history = self.denoiser.train(
            x_train=x_train, epochs=2, batch_size=2, validation_split=0.2
        )

        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)

        # Test invalid inputs
        with self.assertRaises(DenoiserInputError):
            self.denoiser.train(x_train=[1, 2, 3])  # Invalid input type

    def test_denoise(self):
        """Test denoising process."""
        # Test single signal
        signal = np.random.randn(self.input_length)
        denoised = self.denoiser.denoise(signal)
        self.assertEqual(len(denoised), self.input_length)

        # Test batch of signals
        batch = np.random.randn(5, self.input_length)
        denoised_batch = self.denoiser.denoise(batch)
        self.assertEqual(denoised_batch.shape, batch.shape)

        # Test with reconstruction error
        denoised, error = self.denoiser.denoise(
            signal, return_reconstruction_error=True
        )
        self.assertIsInstance(error, float)


class TestECGAutoencoder(unittest.TestCase):
    """Test cases for ECGAutoencoder."""

    def setUp(self):
        self.input_size = 512
        self.latent_dim = 32
        self.model = ECGAutoencoder(self.input_size, self.latent_dim)

    def test_init(self):
        """Test initialization."""
        self.assertIsInstance(self.model, ECGAutoencoder)

    def test_forward(self):
        """Test forward pass."""
        x = torch.randn(10, self.input_size)
        reconstructed, latent = self.model(x)

        self.assertEqual(reconstructed.shape, x.shape)
        self.assertEqual(latent.shape, (10, self.latent_dim))

    def test_encode_decode(self):
        """Test encode and decode functions."""
        x = torch.randn(10, self.input_size)
        latent = self.model.encode(x)
        reconstructed = self.model.decode(latent)

        self.assertEqual(latent.shape, (10, self.latent_dim))
        self.assertEqual(reconstructed.shape, x.shape)


class TestECGAnomalyDetector(unittest.TestCase):
    """Test cases for ECGAnomalyDetector."""

    def setUp(self):
        self.input_size = 512
        self.detector = ECGAnomalyDetector(
            input_size=self.input_size, latent_dim=32, threshold_percentile=95
        )

    def test_init(self):
        """Test initialization."""
        self.assertIsInstance(self.detector, ECGAnomalyDetector)

    def test_fit_predict(self):
        """Test fitting and prediction."""
        # Generate dummy data
        X = np.random.randn(100, self.input_size)

        # Test fitting
        history = self.detector.fit(X, epochs=2, batch_size=32)
        self.assertIn("loss", history)

        # Test prediction
        anomalies, scores = self.detector.predict(X)
        self.assertEqual(len(anomalies), len(X))
        self.assertEqual(len(scores), len(X))
        self.assertTrue(np.all((anomalies == 0) | (anomalies == 1)))

    def test_get_latent_features(self):
        """Test latent feature extraction."""
        X = np.random.randn(100, self.input_size)
        self.detector.fit(X, epochs=2, batch_size=32)

        features = self.detector.get_latent_features(X)
        self.assertEqual(features.shape, (100, 32))

    def test_save_load_model(self):
        """Test model saving and loading."""
        import tempfile
        import os

        # Generate dummy data and fit model
        X = np.random.randn(100, self.input_size)
        self.detector.fit(X, epochs=2, batch_size=32)

        # Save model
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            self.detector.save_model(tmp.name)

            # Load model
            new_detector = ECGAnomalyDetector(input_size=self.input_size, latent_dim=32)
            new_detector.load_model(tmp.name)

        # Clean up
        os.unlink(tmp.name)

        # Test predictions match
        anomalies1, _ = self.detector.predict(X)
        anomalies2, _ = new_detector.predict(X)
        np.testing.assert_array_equal(anomalies1, anomalies2)


if __name__ == "__main__":
    unittest.main()
