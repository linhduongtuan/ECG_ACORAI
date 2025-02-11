# test_ecg_transformer_classifier.py
import unittest
import torch
import numpy as np
from ecg_processor.ecg_transformer_classifier import (
    PositionalEncoding,
    ECGTransformerClassifier,
    train_transformer_classifier,
    predict_transformer,
    TransformerConfigError,
    TransformerInputError,
)


class TestPositionalEncoding(unittest.TestCase):
    """Test cases for the PositionalEncoding class."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 64
        self.max_len = 1000
        self.batch_size = 32
        self.seq_len = 100

    def test_initialization(self):
        """Test PositionalEncoding initialization with valid and invalid parameters."""
        # Valid initialization
        pos_enc = PositionalEncoding(d_model=self.d_model, max_len=self.max_len)
        self.assertIsInstance(pos_enc, PositionalEncoding)

        # Invalid d_model
        with self.assertRaises(TransformerConfigError):
            PositionalEncoding(d_model=0)

        # Invalid dropout
        with self.assertRaises(TransformerConfigError):
            PositionalEncoding(d_model=self.d_model, dropout=1.5)

        # Invalid max_len
        with self.assertRaises(TransformerConfigError):
            PositionalEncoding(d_model=self.d_model, max_len=0)

    def test_forward_pass(self):
        """Test forward pass of PositionalEncoding."""
        pos_enc = PositionalEncoding(d_model=self.d_model, max_len=self.max_len)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = pos_enc(x)

        # Check output shape
        self.assertEqual(output.shape, x.shape)

        # Test invalid input dimensions
        with self.assertRaises(TransformerInputError):
            pos_enc(torch.randn(self.batch_size, self.seq_len))  # Missing feature dim

        # Test sequence length exceeding max_len
        with self.assertRaises(TransformerInputError):
            pos_enc(torch.randn(self.batch_size, self.max_len + 1, self.d_model))


class TestECGTransformerClassifier(unittest.TestCase):
    """Test cases for the ECGTransformerClassifier class."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_length = 1000
        self.d_model = 64
        self.nhead = 4
        self.num_layers = 2
        self.num_classes = 2
        self.batch_size = 32

    def test_initialization(self):
        """Test ECGTransformerClassifier initialization."""
        # Valid initialization
        model = ECGTransformerClassifier(
            input_length=self.input_length,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
        )
        self.assertIsInstance(model, ECGTransformerClassifier)

        # Test invalid parameters
        with self.assertRaises(TransformerConfigError):
            ECGTransformerClassifier(input_length=0)

        with self.assertRaises(TransformerConfigError):
            ECGTransformerClassifier(
                input_length=self.input_length,
                d_model=63,  # Not divisible by nhead
                nhead=4,
            )

    def test_forward_pass(self):
        """Test forward pass of ECGTransformerClassifier."""
        model = ECGTransformerClassifier(
            input_length=self.input_length, d_model=self.d_model, nhead=self.nhead
        )

        # Test with 2D input
        x = torch.randn(self.batch_size, self.input_length)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

        # Test with 3D input
        x = torch.randn(self.batch_size, self.input_length, 1)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))


class TestTrainingAndPrediction(unittest.TestCase):
    """Test cases for training and prediction functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_length = 1000
        self.num_samples = 100
        self.num_classes = 2
        self.batch_size = 32

        # Create synthetic data
        self.x_train = np.random.randn(self.num_samples, self.input_length)
        self.y_train = np.random.randint(0, self.num_classes, size=self.num_samples)
        self.x_val = np.random.randn(self.num_samples // 2, self.input_length)
        self.y_val = np.random.randint(0, self.num_classes, size=self.num_samples // 2)

        # Initialize model
        self.model = ECGTransformerClassifier(
            input_length=self.input_length, d_model=64, nhead=4
        )

    def test_train_transformer_classifier(self):
        """Test training function."""
        # Test basic training
        history = train_transformer_classifier(
            model=self.model,
            x_train=self.x_train,
            y_train=self.y_train,
            epochs=2,
            batch_size=self.batch_size,
        )

        self.assertIn("train_loss", history)
        self.assertIn("train_acc", history)

        # Test training with validation data
        history = train_transformer_classifier(
            model=self.model,
            x_train=self.x_train,
            y_train=self.y_train,
            x_val=self.x_val,
            y_val=self.y_val,
            epochs=2,
            batch_size=self.batch_size,
        )

        self.assertIn("val_loss", history)
        self.assertIn("val_acc", history)

        # Test invalid inputs
        with self.assertRaises(TransformerInputError):
            train_transformer_classifier(
                model=self.model,
                x_train=self.x_train,
                y_train=self.y_train[:50],  # Mismatched lengths
            )

    def test_predict_transformer(self):
        """Test prediction function."""
        # Test basic prediction
        x_test = np.random.randn(10, self.input_length)
        predictions = predict_transformer(self.model, x_test)
        self.assertEqual(predictions.shape, (10,))

        # Test prediction with probabilities
        predictions, probabilities = predict_transformer(
            self.model, x_test, return_probabilities=True
        )
        self.assertEqual(predictions.shape, (10,))
        self.assertEqual(probabilities.shape, (10, self.num_classes))

        # Test invalid inputs
        with self.assertRaises(TransformerInputError):
            predict_transformer(
                self.model,
                np.random.randn(10, self.input_length + 1),  # Wrong length
            )

    def test_early_stopping(self):
        """Test early stopping functionality."""
        history = train_transformer_classifier(
            model=self.model,
            x_train=self.x_train,
            y_train=self.y_train,
            x_val=self.x_val,
            y_val=self.y_val,
            epochs=10,
            early_stopping_patience=2,
            batch_size=self.batch_size,
        )
        # Check if training stopped early
        self.assertLessEqual(len(history["train_loss"]), 10)


if __name__ == "__main__":
    unittest.main()
