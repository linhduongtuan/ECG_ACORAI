"""
Feature transformation and normalization utilities for ECG signal processing using PyTorch.

This module provides functions for transforming and normalizing ECG features,
including scaling, standardization, dimensionality reduction (via PCA), and
handling outliers and skewed feature distributions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from scipy import stats
from .config import ECGConfig  # Assumes you have a configuration module

logger = logging.getLogger(__name__)


# --- Scaler Implementations ---


class TorchStandardScaler:
    """
    A simple standard scaler that subtracts the mean and divides by the standard deviation.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: torch.Tensor):
        # Assume X is of shape (n_samples, 1)
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, unbiased=False, keepdim=True)
        # Avoid division by zero.
        self.std[self.std == 0] = 1.0

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X * self.std) + self.mean


class TorchRobustScaler:
    """
    A robust scaler that subtracts the median and scales according to the interquartile range.
    """

    def __init__(self):
        self.median = None
        self.iqr = None

    def fit(self, X: torch.Tensor):
        # Assume X is of shape (n_samples, 1)
        self.median = X.median(dim=0, keepdim=True)[0]
        q1 = torch.quantile(X, 0.25, dim=0, keepdim=True)
        q3 = torch.quantile(X, 0.75, dim=0, keepdim=True)
        self.iqr = q3 - q1
        self.iqr[self.iqr == 0] = 1.0

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.median) / self.iqr

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X * self.iqr) + self.median


# --- PCA Implementation ---


class TorchPCA:
    """
    A simple PCA implementation using torch.linalg.svd.
    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean = None
        self.components = (
            None  # Will hold the projection matrix (n_features x n_components)
        )

    def fit(self, X: torch.Tensor):
        # X shape: (n_samples, n_features)
        self.mean = X.mean(dim=0, keepdim=True)
        X_centered = X - self.mean
        # Use SVD on the centered data.
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        # Vh is (n_features, n_features); we take the first n_components rows and transpose.
        self.components = Vh[: self.n_components, :].T

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.components is None:
            self.fit(X)
        X_centered = X - self.mean
        return X_centered @ self.components

    def inverse_transform(self, X_transformed: torch.Tensor) -> torch.Tensor:
        return (X_transformed @ self.components.T) + self.mean


# --- Helper Functions for Data Distribution Checks ---


def _is_heavily_skewed(
    data: Union[torch.Tensor, np.ndarray], threshold: float = 0.5
) -> bool:
    """
    Check whether the data is heavily skewed.
    """
    try:
        if isinstance(data, np.ndarray):
            valid_data = data[~np.isnan(data)]
            if valid_data.size < 3:
                return False
            skewness = stats.skew(valid_data)
            q1, median, q3 = np.percentile(valid_data, [25, 50, 75])
            iqr = q3 - q1
            if iqr == 0:
                return False
            traditional_skewness = abs(skewness)
            quartile_skewness = abs((q3 + q1 - 2 * median) / iqr)
            return (traditional_skewness > threshold * 2) or (
                quartile_skewness > threshold
            )
        else:
            valid_data = data[~torch.isnan(data)]
            if valid_data.numel() < 3:
                return False
            mean_val = valid_data.mean()
            std_val = valid_data.std(unbiased=False)
            skewness = ((valid_data - mean_val) ** 3).mean() / (std_val**3 + 1e-10)
            skewness = abs(skewness.item())
            q1 = torch.quantile(valid_data, 0.25)
            median = torch.quantile(valid_data, 0.5)
            q3 = torch.quantile(valid_data, 0.75)
            iqr = q3 - q1
            if iqr.item() == 0:
                return False
            quartile_skewness = abs((q3 + q1 - 2 * median).item() / iqr.item())
            return (skewness > threshold * 2) or (quartile_skewness > threshold)
    except Exception as e:
        logger.warning(f"Error in skewness calculation: {str(e)}")
        return False


def _has_outliers(
    data: Union[torch.Tensor, np.ndarray], threshold: float = 3.0
) -> bool:
    """
    Check whether the data has significant outliers using the IQR method.
    """
    try:
        if isinstance(data, np.ndarray):
            valid_data = data[~np.isnan(data)]
            if valid_data.size < 2:
                return False
            q1, q3 = np.percentile(valid_data, [25, 75])
            iqr = q3 - q1
            if iqr == 0:
                return False
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            return np.any((valid_data < lower_bound) | (valid_data > upper_bound))
        else:
            valid_data = data[~torch.isnan(data)]
            if valid_data.numel() < 2:
                return False
            q1 = torch.quantile(valid_data, 0.25)
            q3 = torch.quantile(valid_data, 0.75)
            iqr = q3 - q1
            if iqr.item() == 0:
                return False
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            return torch.any(valid_data < lower_bound) or torch.any(
                valid_data > upper_bound
            )
    except Exception as e:
        logger.warning(f"Error in outlier detection: {str(e)}")
        return False


# --- Feature Transformer Class ---


class FeatureTransformer:
    """
    A class for transforming and normalizing ECG features using PyTorch.

    Provides methods for scaling (using either standard or robust scalers),
    dimensionality reduction (via PCA), and inverse transformation.
    """

    def __init__(self, config: Optional[ECGConfig] = None):
        """
        Initialize the feature transformer.

        Parameters
        ----------
        config : Optional[ECGConfig]
            Configuration object containing transformation parameters.
        """
        self.config = config or ECGConfig()
        self.scalers: Dict[str, Union[TorchStandardScaler, TorchRobustScaler]] = {}
        self.pca: Optional[TorchPCA] = None
        self.feature_names: List[str] = []
        self._is_fitted = False

    def fit(
        self,
        features: Dict[str, Union[np.ndarray, torch.Tensor]],
        feature_list: Optional[List[str]] = None,
    ) -> None:
        """
        Fit the scalers (and optionally PCA) to the feature data.

        Parameters
        ----------
        features : Dict[str, np.ndarray or torch.Tensor]
            Dictionary of feature arrays.
        feature_list : Optional[List[str]]
            List of features to use. If None, all features are used.
        """
        try:
            if not features:
                raise ValueError("Features dictionary is empty")

            # Select features to use.
            self.feature_names = feature_list or list(features.keys())
            if not self.feature_names:
                raise ValueError("No features selected for transformation")

            for feature_name in self.feature_names:
                if feature_name not in features:
                    raise ValueError(f"Feature {feature_name} not found in data")
                feature_data = features[feature_name]
                if not isinstance(feature_data, torch.Tensor):
                    feature_data = torch.tensor(feature_data, dtype=torch.float32)
                # Determine scaler type.
                if _is_heavily_skewed(feature_data) or _has_outliers(feature_data):
                    scaler = TorchRobustScaler()
                else:
                    scaler = TorchStandardScaler()
                valid_data = feature_data[~torch.isnan(feature_data)].view(-1, 1)
                if valid_data.numel() > 0:
                    scaler.fit(valid_data)
                self.scalers[feature_name] = scaler

            # Initialize PCA if enabled in the configuration.
            if hasattr(self.config, "USE_PCA") and self.config.USE_PCA:
                if len(self.feature_names) > self.config.PCA_COMPONENTS:
                    self.pca = TorchPCA(n_components=self.config.PCA_COMPONENTS)

            self._is_fitted = True
            logger.info("Feature transformer successfully fitted")

        except Exception as e:
            logger.error(f"Error fitting feature transformer: {str(e)}")
            raise

    def transform(
        self, features: Dict[str, Union[np.ndarray, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Transform the features using the fitted scalers (and PCA if applicable).

        Parameters
        ----------
        features : Dict[str, np.ndarray or torch.Tensor]
            Dictionary of feature arrays.

        Returns
        -------
        torch.Tensor
            Transformed feature matrix.
        """
        try:
            if not self._is_fitted:
                raise ValueError("Transformer must be fitted before transform")

            transformed_features = []
            for feature_name in self.feature_names:
                if feature_name not in features:
                    raise ValueError(f"Feature {feature_name} not found in data")
                feature_data = features[feature_name]
                if not isinstance(feature_data, torch.Tensor):
                    feature_data = torch.tensor(feature_data, dtype=torch.float32)
                feature_data = torch.nan_to_num(feature_data, nan=0.0)
                scaler = self.scalers[feature_name]
                transformed = scaler.transform(feature_data.view(-1, 1)).view(-1)
                transformed_features.append(transformed)

            # Combine features column-wise.
            feature_matrix = torch.stack(transformed_features, dim=1)
            if self.pca is not None:
                if self.pca.mean is None or self.pca.components is None:
                    self.pca.fit(feature_matrix)
                feature_matrix = self.pca.transform(feature_matrix)
            return feature_matrix

        except Exception as e:
            logger.error(f"Error transforming features: {str(e)}")
            raise

    def fit_transform(
        self,
        features: Dict[str, Union[np.ndarray, torch.Tensor]],
        feature_list: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Fit the transformer and transform the features in one step.

        Parameters
        ----------
        features : Dict[str, np.ndarray or torch.Tensor]
            Dictionary of feature arrays.
        feature_list : Optional[List[str]]
            List of features to use. If None, all features are used.

        Returns
        -------
        torch.Tensor
            Transformed feature matrix.
        """
        self.fit(features, feature_list)
        return self.transform(features)

    def inverse_transform(
        self, feature_matrix: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Inverse transform the feature matrix back to the original feature scales.

        Parameters
        ----------
        feature_matrix : torch.Tensor
            Transformed feature matrix.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of inverse transformed features.
        """
        try:
            if not self._is_fitted:
                raise ValueError("Transformer must be fitted before inverse transform")

            if self.pca is not None:
                feature_matrix = self.pca.inverse_transform(feature_matrix)
            features_inv = {}
            for i, feature_name in enumerate(self.feature_names):
                col = feature_matrix[:, i].view(-1, 1)
                scaler = self.scalers[feature_name]
                features_inv[feature_name] = scaler.inverse_transform(col).view(-1)
            return features_inv

        except Exception as e:
            logger.error(f"Error inverse transforming features: {str(e)}")
            raise


# --- Testing Function ---


def test_feature_transformer():
    """
    Test the FeatureTransformer class with synthetic data.
    """
    torch.manual_seed(42)
    n_samples = 100
    features = {
        "mean": torch.normal(0, 1, (n_samples,)),
        "std": torch.abs(torch.normal(0, 1, (n_samples,))),
        "skewness": torch.distributions.Exponential(1).sample((n_samples,)),
        "kurtosis": torch.normal(0, 1, (n_samples,)),
    }
    # Inject outlier and missing value.
    features["std"][0] = 100.0
    features["mean"][1] = float("nan")

    try:
        transformer = FeatureTransformer()
        transformer.fit(features)
        print("Fit successful")
        transformed = transformer.transform(features)
        print("\nTransformed feature matrix shape:", transformed.shape)
        inverse_transformed = transformer.inverse_transform(transformed)
        print("\nInverse transform successful")
        for feature_name in features.keys():
            original = features[feature_name]
            reconstructed = inverse_transformed[feature_name]
            valid_mask = ~torch.isnan(original)
            mse = (
                ((original[valid_mask] - reconstructed[valid_mask]) ** 2).mean().item()
            )
            print(f"\nMSE for {feature_name}: {mse:.6f}")
    except Exception as e:
        print(f"Test failed: {str(e)}")


if __name__ == "__main__":
    test_feature_transformer()
