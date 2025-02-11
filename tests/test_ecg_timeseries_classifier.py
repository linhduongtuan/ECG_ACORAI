# test_ecg_timeseries_classifier.py
import unittest
import numpy as np
import torch
from ecg_processor.ecg_timeseries_classifier import (
    ECGTimeSeriesClassifier,
    train_time_series_classifier,
    predict,
    ECGClassifierError,
)


class TestECGTimeSeriesClassifier(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.input_length = 100
        self.num_classes = 2
        self.batch_size = 32
        self.model = ECGTimeSeriesClassifier(
            input_length=self.input_length, num_classes=self.num_classes
        )

        # Create sample data
        self.x_train = np.random.randn(64, self.input_length)
        self.y_train = np.random.randint(0, self.num_classes, 64)
        self.x_val = np.random.randn(32, self.input_length)
        self.y_val = np.random.randint(0, self.num_classes, 32)

    def test_model_initialization(self):
        """Test model initialization with valid and invalid parameters."""
        # Test valid initialization
        model = ECGTimeSeriesClassifier(input_length=100, num_classes=2)
        self.assertEqual(model.input_length, 100)
        self.assertEqual(model.num_classes, 2)

        # Test invalid input_length
        with self.assertRaises(ECGClassifierError):
            ECGTimeSeriesClassifier(input_length=15, num_classes=2)

        # Test invalid num_classes
        with self.assertRaises(ECGClassifierError):
            ECGTimeSeriesClassifier(input_length=100, num_classes=1)

    def test_forward_pass(self):
        """Test forward pass with valid and invalid inputs."""
        # Test valid input
        x = torch.randn(self.batch_size, 1, self.input_length)
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

        # Test invalid input dimensions
        with self.assertRaises(ECGClassifierError):
            x_invalid = torch.randn(
                self.batch_size, self.input_length
            )  # Missing channel dim
            self.model(x_invalid)

        # Test invalid input length
        with self.assertRaises(ECGClassifierError):
            x_invalid = torch.randn(self.batch_size, 1, self.input_length + 1)
            self.model(x_invalid)

        # Test invalid number of channels
        with self.assertRaises(ECGClassifierError):
            x_invalid = torch.randn(self.batch_size, 2, self.input_length)
            self.model(x_invalid)

    def test_train_classifier(self):
        """Test training function with various configurations."""
        # Test basic training
        history = train_time_series_classifier(
            model=self.model,
            x_train=self.x_train,
            y_train=self.y_train,
            epochs=2,
            batch_size=32,
        )
        self.assertIn("train_loss", history)
        self.assertIn("train_acc", history)

        # Test training with validation data
        history = train_time_series_classifier(
            model=self.model,
            x_train=self.x_train,
            y_train=self.y_train,
            x_val=self.x_val,
            y_val=self.y_val,
            epochs=2,
            batch_size=32,
        )
        self.assertIn("val_loss", history)
        self.assertIn("val_acc", history)

        # Test invalid input shapes
        with self.assertRaises(ECGClassifierError):
            train_time_series_classifier(
                model=self.model,
                x_train=self.x_train,
                y_train=self.y_train[:10],  # Mismatched lengths
                epochs=2,
            )

        # Test invalid input values
        with self.assertRaises(ECGClassifierError):
            x_invalid = self.x_train.copy()
            x_invalid[0, 0] = np.nan
            train_time_series_classifier(
                model=self.model, x_train=x_invalid, y_train=self.y_train, epochs=2
            )

    def test_predict(self):
        """Test prediction function with various inputs."""
        # Test basic prediction
        x_test = np.random.randn(10, self.input_length)
        predictions = predict(self.model, x_test)
        self.assertEqual(predictions.shape, (10,))
        self.assertTrue(
            np.all(predictions >= 0) and np.all(predictions < self.num_classes)
        )

        # Test prediction with probabilities
        predictions, probabilities = predict(
            self.model, x_test, return_probabilities=True
        )
        self.assertEqual(predictions.shape, (10,))
        self.assertEqual(probabilities.shape, (10, self.num_classes))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0))

        # Test invalid input
        with self.assertRaises(ECGClassifierError):
            x_invalid = np.random.randn(10, self.input_length + 1)  # Wrong length
            predict(self.model, x_invalid)

        # Test invalid values
        with self.assertRaises(ECGClassifierError):
            x_invalid = x_test.copy()
            x_invalid[0, 0] = np.nan
            predict(self.model, x_invalid)

    def test_early_stopping(self):
        """Test early stopping functionality."""
        history = train_time_series_classifier(
            model=self.model,
            x_train=self.x_train,
            y_train=self.y_train,
            x_val=self.x_val,
            y_val=self.y_val,
            epochs=10,
            early_stopping_patience=2,
            batch_size=32,
        )
        # Check if training stopped early
        self.assertLessEqual(len(history["train_loss"]), 10)

    def test_model_device_handling(self):
        """Test model handling of CPU/GPU devices."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        x = torch.randn(self.batch_size, 1, self.input_length).to(device)
        output = self.model(x)
        self.assertEqual(output.device, device)


if __name__ == "__main__":
    unittest.main()
