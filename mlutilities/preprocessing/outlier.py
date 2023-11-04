from typing import Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
import numpy as np


# replace outliers with upper fence from IQR
class IQR(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    An Interquartile Range (IQR) Transformer.

    This transformer calculates the Interquartile Range (IQR) for each feature in the input data and replaces outliers
    with predefined thresholds. Outliers are values that fall outside the upper and lower fences defined by the IQR.

    Parameters:
    -----------
    threshold : float, optional (default=1.5)
        The multiplier for the IQR used to define the upper and lower thresholds for identifying outliers.

    Methods:
    --------
    fit(X, y=None):
        Calculate the IQR and thresholds for the input data.

    transform(X, y=None):
        Replace outliers in the input data with the defined upper and lower thresholds.

    Attributes:
    -----------
    iqr : pandas Series
        The calculated IQR for each feature.

    upper_fence : pandas Series
        The upper fence for each feature used to identify upper outliers.

    lower_fence : pandas Series
        The lower fence for each feature used to identify lower outliers.

    upper_threshold : pandas Series
        The upper threshold for each feature used to replace upper outliers.

    lower_threshold : pandas Series
        The lower threshold for each feature used to replace lower outliers.
    """

    def __init__(self, threshold: float = 1.5):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Calculate the IQR and thresholds for the input data.

        Parameters:
        -----------
            X : pandas DataFrame
                The input DataFrame containing the training data.
            y : pandas Series, optional
                The target variable (ignored).
        Returns:
        --------
        IQR
            The IQR transformer.
        """

        input_features = X.columns
        self.iqr = X[input_features].quantile([0.25, 0.75]).apply(lambda x: x[0.75] - x[0.25], axis=0)
        self.upper_fence = X[input_features].quantile(0.75) + self.iqr * 1.5
        self.lower_fence = X[input_features].quantile(0.25) - self.iqr * 1.5

        self.upper_threshold = X[input_features].quantile(0.75) + self.iqr * self.threshold
        self.lower_threshold = X[input_features].quantile(0.25) - self.iqr * self.threshold

        self.upper_fence = pd.Series([np.min([k, v]) for k, v in zip(self.upper_fence, X[input_features].max())], self.upper_fence.index)
        self.lower_fence = pd.Series([np.max([k, v]) for k, v in zip(self.lower_fence, X[input_features].min())], self.lower_fence.index)

        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Replace outliers in the input data with the defined upper and lower thresholds.

        Parameters:
        -----------
            X : pandas DataFrame
                The input DataFrame containing the training data.
            y : pandas Series, optional
                The target variable (ignored).
        Returns:
        --------
        pandas DataFrame
            The input DataFrame with outliers replaced by the defined thresholds.

        """

        return X.where(X < self.upper_threshold, self.upper_fence, axis=1).where(X > self.lower_threshold, self.lower_fence, axis=1)
