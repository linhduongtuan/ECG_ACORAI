# test_features_transforms.py
import unittest
import numpy as np
from scipy import stats
from ecg_processor.features_transforms import (
    FeatureTransformer,
    _is_heavily_skewed,
    _has_outliers,
)
from ecg_processor.config import ECGConfig


class TestFeatureTransformer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 100

        # Create synthetic feature data
        self.features = {
            "normal": np.random.normal(0, 1, self.n_samples),
            "skewed": np.random.exponential(2, self.n_samples),  # Increased skewness
            "outliers": np.concatenate(
                [
                    np.random.normal(0, 1, self.n_samples - 1),
                    [10.0],  # Clear outlier
                ]
            ),
            "missing": np.random.normal(0, 1, self.n_samples),
        }

        # Add missing values
        self.features["missing"][0] = np.nan

        # Create custom config for testing
        self.config = ECGConfig()
        self.config.USE_PCA = False  # Disable PCA for basic tests
        self.config.PCA_COMPONENTS = 2

        # Create transformer instance
        self.transformer = FeatureTransformer(config=self.config)

    def test_initialization(self):
        """Test FeatureTransformer initialization."""
        self.assertIsInstance(self.transformer, FeatureTransformer)
        self.assertFalse(self.transformer._is_fitted)
        self.assertEqual(len(self.transformer.feature_names), 0)

    def test_fit(self):
        """Test fit method."""
        self.transformer.fit(self.features)
        self.assertTrue(self.transformer._is_fitted)
        self.assertEqual(len(self.transformer.scalers), len(self.features))

    def test_transform(self):
        """Test transform method."""
        self.transformer.fit(self.features)
        transformed = self.transformer.transform(self.features)

        # Check output shape
        self.assertEqual(transformed.shape[0], self.n_samples)
        self.assertEqual(transformed.shape[1], len(self.features))

    def test_fit_transform(self):
        """Test fit_transform method."""
        transformed = self.transformer.fit_transform(self.features)
        self.assertEqual(transformed.shape[0], self.n_samples)
        self.assertEqual(transformed.shape[1], len(self.features))

    def test_inverse_transform(self):
        """Test inverse_transform method."""
        transformed = self.transformer.fit_transform(self.features)
        inverse_transformed = self.transformer.inverse_transform(transformed)

        # Check that we get back the original features (excluding missing values)
        for feature_name in self.features:
            original = self.features[feature_name]
            reconstructed = inverse_transformed[feature_name]
            mask = ~np.isnan(original)
            np.testing.assert_array_almost_equal(
                original[mask], reconstructed[mask], decimal=5
            )


class TestHelperFunctions(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.normal_data = np.random.normal(0, 1, 1000)
        self.skewed_data = np.random.exponential(2, 1000)  # Increased skewness
        self.outlier_data = np.concatenate(
            [
                np.random.normal(0, 1, 999),
                [100.0],  # Very clear outlier
            ]
        )

    def test_is_heavily_skewed(self):
        """Test _is_heavily_skewed function."""
        self.assertFalse(_is_heavily_skewed(self.normal_data))
        self.assertTrue(
            _is_heavily_skewed(self.skewed_data, threshold=0.5)
        )  # Adjusted threshold
        self.assertFalse(_is_heavily_skewed(np.array([])))
        self.assertFalse(_is_heavily_skewed(np.array([np.nan])))

    def test_has_outliers(self):
        """Test _has_outliers function."""
        self.assertFalse(_has_outliers(self.normal_data))
        self.assertTrue(_has_outliers(self.outlier_data))
        self.assertFalse(_has_outliers(np.array([])))
        self.assertFalse(_has_outliers(np.array([np.nan])))


class TestHelperFunctions(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with more extreme test cases."""
        np.random.seed(42)

        # Create normal distribution data
        self.normal_data = np.random.normal(0, 1, 1000)

        # Create heavily skewed data using exponential distribution
        # Multiply by 2 to make skewness more pronounced
        self.skewed_data = 2 * np.random.exponential(scale=1.0, size=1000)

        # Add some extreme values to ensure skewness
        self.skewed_data[0:10] = self.skewed_data[0:10] * 5

        # Create data with outliers
        self.outlier_data = np.random.normal(0, 1, 1000)
        self.outlier_data[0] = 20.0  # Add extreme outlier

        # Create clean data
        self.clean_data = np.clip(np.random.normal(0, 1, 1000), -2, 2)

    def test_is_heavily_skewed(self):
        """Test _is_heavily_skewed function with improved test cases."""
        # Test normal distribution (should not be heavily skewed)
        self.assertFalse(_is_heavily_skewed(self.normal_data, threshold=0.5))

        # Test skewed distribution
        self.assertTrue(
            _is_heavily_skewed(self.skewed_data, threshold=0.5),
            "Failed to detect heavily skewed data",
        )

        # Test edge cases
        self.assertFalse(_is_heavily_skewed(np.array([])))
        self.assertFalse(_is_heavily_skewed(np.array([np.nan])))

        # Test constant data
        self.assertFalse(_is_heavily_skewed(np.ones(100)))

        # Print diagnostic information
        skewness = stats.skew(self.skewed_data)
        print("\nDiagnostic Information:")
        print(f"Skewness of normal data: {stats.skew(self.normal_data):.3f}")
        print(f"Skewness of skewed data: {skewness:.3f}")


if __name__ == "__main__":
    unittest.main()
