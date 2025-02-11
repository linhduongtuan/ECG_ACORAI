"""
Feature transformation and normalization utilities for ECG signal processing.

This module provides functions for transforming and normalizing ECG features,
including scaling, standardization, and dimensionality reduction techniques.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats
import logging
from .config import ECGConfig

logger = logging.getLogger(__name__)


class FeatureTransformer:
    """
    A class for transforming and normalizing ECG features.

    This class provides methods for:
    - Feature scaling and normalization
    - Dimensionality reduction
    - Feature selection
    - Outlier detection and handling
    """

    def __init__(self, config: Optional[ECGConfig] = None):
        """
        Initialize the feature transformer.

        Parameters
        ----------
        config : Optional[ECGConfig]
            Configuration object containing transformation parameters
        """
        self.config = config or ECGConfig()
        self.scalers: Dict[str, Union[StandardScaler, MinMaxScaler, RobustScaler]] = {}
        self.pca: Optional[PCA] = None
        self.feature_names: List[str] = []
        self._is_fitted = False

    def fit(
        self, features: Dict[str, np.ndarray], feature_list: Optional[List[str]] = None
    ) -> None:
        """
        Fit the transformers to the feature data.

        Parameters
        ----------
        features : Dict[str, np.ndarray]
            Dictionary of feature arrays
        feature_list : Optional[List[str]]
            List of features to use. If None, use all features.

        Raises
        ------
        ValueError
            If features are invalid or empty
        """
        try:
            if not features:
                raise ValueError("Features dictionary is empty")

            # Select features to use
            self.feature_names = feature_list or list(features.keys())
            if not self.feature_names:
                raise ValueError("No features selected for transformation")

            # Initialize scalers based on feature characteristics
            for feature_name in self.feature_names:
                if feature_name not in features:
                    raise ValueError(f"Feature {feature_name} not found in data")

                feature_data = features[feature_name]
                if not isinstance(feature_data, np.ndarray):
                    raise ValueError(
                        f"Feature {feature_name} data must be a numpy array"
                    )

                # Choose scaler based on data distribution
                if np.any(np.isnan(feature_data)):
                    logger.warning(
                        f"NaN values found in {feature_name}, will be handled during transformation"
                    )

                if _is_heavily_skewed(feature_data):
                    self.scalers[feature_name] = RobustScaler()
                elif _has_outliers(feature_data):
                    self.scalers[feature_name] = RobustScaler()
                else:
                    self.scalers[feature_name] = StandardScaler()

                # Fit the scaler
                valid_data = feature_data[~np.isnan(feature_data)].reshape(-1, 1)
                if len(valid_data) > 0:
                    self.scalers[feature_name].fit(valid_data)

            # Initialize PCA if dimensionality reduction is enabled
            if hasattr(self.config, "USE_PCA") and self.config.USE_PCA:
                if len(self.feature_names) > self.config.PCA_COMPONENTS:
                    self.pca = PCA(n_components=self.config.PCA_COMPONENTS)

            self._is_fitted = True
            logger.info("Feature transformer successfully fitted")

        except Exception as e:
            logger.error(f"Error fitting feature transformer: {str(e)}")
            raise

    def transform(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Transform the features using fitted transformers.

        Parameters
        ----------
        features : Dict[str, np.ndarray]
            Dictionary of feature arrays

        Returns
        -------
        np.ndarray
            Transformed feature matrix

        Raises
        ------
        ValueError
            If transformer is not fitted or features are invalid
        """
        try:
            if not self._is_fitted:
                raise ValueError("Transformer must be fitted before transform")

            transformed_features = []
            for feature_name in self.feature_names:
                if feature_name not in features:
                    raise ValueError(f"Feature {feature_name} not found in data")

                feature_data = features[feature_name]
                if not isinstance(feature_data, np.ndarray):
                    raise ValueError(
                        f"Feature {feature_name} data must be a numpy array"
                    )

                # Handle missing values
                feature_data = np.nan_to_num(feature_data, nan=0.0)

                # Transform the feature
                transformed = (
                    self.scalers[feature_name]
                    .transform(feature_data.reshape(-1, 1))
                    .ravel()
                )

                transformed_features.append(transformed)

            # Combine all transformed features
            feature_matrix = np.column_stack(transformed_features)

            # Apply PCA if enabled
            if self.pca is not None:
                if not hasattr(self.pca, "components_"):
                    self.pca.fit(feature_matrix)
                feature_matrix = self.pca.transform(feature_matrix)

            return feature_matrix

        except Exception as e:
            logger.error(f"Error transforming features: {str(e)}")
            raise

    def fit_transform(
        self, features: Dict[str, np.ndarray], feature_list: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Fit the transformer and transform the features in one step.

        Parameters
        ----------
        features : Dict[str, np.ndarray]
            Dictionary of feature arrays
        feature_list : Optional[List[str]]
            List of features to use. If None, use all features.

        Returns
        -------
        np.ndarray
            Transformed feature matrix
        """
        self.fit(features, feature_list)
        return self.transform(features)

    def inverse_transform(self, feature_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Inverse transform the feature matrix back to original scale.

        Parameters
        ----------
        feature_matrix : np.ndarray
            Transformed feature matrix

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of inverse transformed features

        Raises
        ------
        ValueError
            If transformer is not fitted or input is invalid
        """
        try:
            if not self._is_fitted:
                raise ValueError("Transformer must be fitted before inverse transform")

            # Inverse PCA if applied
            if self.pca is not None:
                feature_matrix = self.pca.inverse_transform(feature_matrix)

            # Inverse transform each feature
            features = {}
            for i, feature_name in enumerate(self.feature_names):
                feature_data = feature_matrix[:, i].reshape(-1, 1)
                features[feature_name] = (
                    self.scalers[feature_name].inverse_transform(feature_data).ravel()
                )

            return features

        except Exception as e:
            logger.error(f"Error inverse transforming features: {str(e)}")
            raise


def _is_heavily_skewed(data: np.ndarray, threshold: float = 0.5) -> bool:
    """
    Check if data is heavily skewed using a robust skewness measure.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    threshold : float
        Threshold for determining heavy skewness

    Returns
    -------
    bool
        True if data is heavily skewed, False otherwise
    """
    try:
        valid_data = data[~np.isnan(data)]
        if len(valid_data) < 3:  # Need at least 3 points for skewness
            return False

        # Calculate skewness using scipy.stats
        skewness = stats.skew(valid_data)

        # Calculate quartiles for robust measure
        q1, median, q3 = np.percentile(valid_data, [25, 50, 75])
        iqr = q3 - q1

        if iqr == 0:  # Handle constant or near-constant data
            return False

        # Use both traditional skewness and quartile-based measure
        traditional_skewness = abs(skewness)
        quartile_skewness = abs((q3 + q1 - 2 * median) / iqr)

        # Combined measure
        is_skewed = (traditional_skewness > threshold * 2) or (
            quartile_skewness > threshold
        )

        return bool(is_skewed)

    except Exception as e:
        logger.warning(f"Error in skewness calculation: {str(e)}")
        return False


def _has_outliers(data: np.ndarray, threshold: float = 3.0) -> bool:
    """
    Check if data has significant outliers using IQR method.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    threshold : float
        Number of IQRs to use for outlier detection

    Returns
    -------
    bool
        True if outliers are detected, False otherwise
    """
    try:
        valid_data = data[~np.isnan(data)]
        if len(valid_data) < 2:
            return False

        # Calculate quartiles
        q1, q3 = np.percentile(valid_data, [25, 75])
        iqr = q3 - q1

        if iqr == 0:  # Handle constant or near-constant data
            return False

        # Define bounds
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        # Check for outliers
        return np.any((valid_data < lower_bound) | (valid_data > upper_bound))
    except Exception as e:
        logger.warning(f"Error in outlier detection: {str(e)}")
        return False


def test_feature_transformer():
    """
    Test the FeatureTransformer class with synthetic data.
    """
    # Create synthetic feature data
    np.random.seed(42)
    n_samples = 100
    features = {
        "mean": np.random.normal(0, 1, n_samples),
        "std": np.abs(np.random.normal(0, 1, n_samples)),
        "skewness": np.random.exponential(1, n_samples),  # Skewed distribution
        "kurtosis": np.random.normal(0, 1, n_samples),
    }

    # Add some outliers
    features["std"][0] = 100.0  # Add outlier

    # Add some missing values
    features["mean"][1] = np.nan

    try:
        # Test feature transformer
        transformer = FeatureTransformer()

        # Test fit
        transformer.fit(features)
        print("Fit successful")

        # Test transform
        transformed = transformer.transform(features)
        print("\nTransformed feature matrix shape:", transformed.shape)

        # Test inverse transform
        inverse_transformed = transformer.inverse_transform(transformed)
        print("\nInverse transform successful")

        # Compare original and reconstructed features
        for feature_name in features.keys():
            original = features[feature_name]
            reconstructed = inverse_transformed[feature_name]
            mse = np.mean(
                (original[~np.isnan(original)] - reconstructed[~np.isnan(original)])
                ** 2
            )
            print(f"\nMSE for {feature_name}: {mse:.6f}")

    except Exception as e:
        print(f"Test failed: {str(e)}")


if __name__ == "__main__":
    test_feature_transformer()
