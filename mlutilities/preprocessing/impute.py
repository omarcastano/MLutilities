import numpy as np
import pandas as pd
from typing import Optional, List
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin


class LinearModelImputer(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    Custom transformer for imputing missing values based on a linear regression.
    It replaces missing values in the 'impute_feature' column of a DataFrame by imputing
    values predicted from a linear regression model trained with a set of given variables.

    Parameters:
    -----------
      impute_feature:
        The name of the feature to be imputed (i.e., the feature with missing values).
      regression_features:
        The names of the features to be used in the linear regression model.

    Raises:
    -------
        ValueError: If the feature column contains NaN values or no numeric columns are found in the input data.
    """

    def __init__(self, impute_feature: str, regression_features: List[str] = None):
        self.impute_feature = impute_feature
        self.regression_features = regression_features
        self.linear_model = LinearRegression()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the linear regression model using non-null instances from the 'feature' and 'impute_feature' columns of the input data.

        Parameters:
        -----------
            X:
              The input data containing the 'feature' and 'impute_feature' columns.
            y:
              The target variable (ignored).

        Returns:
            LinearModelImputer

        """
        # select numeric variables only
        X_copy = X.copy().select_dtypes(include=np.number)

        # if feature column is not provided, select the most correlated numeric column
        if self.regression_features is None:
            numeric_cols = X_copy.columns.copy().drop(self.impute_feature)

            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in the input data.")

            correlations = X_copy.corr(method="spearman")
            most_correlated_feature = correlations.loc[self.impute_feature].abs().iloc[1:].idxmax()
            self.regression_features = [most_correlated_feature]

        # check that feature column does not have missing values
        for feature in self.regression_features:
            if X_copy[feature].isnull().any():
                raise ValueError("The feature column contains NaN values.")

        # get instances with impute_feature missing values
        nan_instances = X_copy.loc[:, self.impute_feature].isnull()

        # use instances with no missing impute_feature values to train the linear model
        feature_train = X_copy.loc[~nan_instances, self.regression_features]
        target_train = X_copy.loc[~nan_instances, self.impute_feature]

        # train the linear model
        self.linear_model.fit(feature_train, target_train)

        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Replace missing values in the impute_feature column with predicted values from the trained linear regression model.

        Parameters:
        -----------
            X:
              The input data containing the feature and impute_feature columns.
            y:
              The target variable (ignored).

        Returns:
        --------
            The modified impute_feature column with missing values replaced by predictions.

        """
        X_copy = X.copy().select_dtypes(include=np.number)

        # get instances with impute_feature missing values
        nan_instances = X_copy.loc[:, self.impute_feature].isna()

        # extract instances with missing impute_feature values and their corresponding feature values
        feature_test = X_copy.loc[nan_instances, self.regression_features]

        # predict and replace the missing impute_feature values using the trained linear regression model
        target_predict = self.linear_model.predict(feature_test)
        X_copy.loc[nan_instances, self.impute_feature] = target_predict

        return X_copy.loc[:, [self.impute_feature]]


class GroupImputer(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    Custom transformer for imputing missing values based on group-wise statistics.
    It replaces missing values in the 'impute_feature' column of a DataFrame by imputing
    values based on the most common value if the 'impute_feature' is categorical
    (or median if the 'impute variable' is numerical) within each group defined
    by the 'group_features' columns.

    Parameters:
    -----------
      impute_feature:
        The name of the feature to be imputed (i.e., the feature with missing values).
      group_features:
        The names of the features used to define the groups for imputation.

    Returns:
    --------
      A pandas DataFrame containing the 'impute_feature' with imputed missing values
    """

    def __init__(self, impute_feature: str, group_features: List[str]):
        self.impute_feature = impute_feature
        self.group_features = group_features

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fits the imputer by calculating the most common value for the 'impute_feature' within each group.

        Parameters:
        -----------
          X:
            The input DataFrame containing the training data.
          y:
            The target variable (ignored).
        """
        X_copy = X.copy()

        # check that group features does not have missing values
        for feature in self.group_features:
            if X_copy.loc[:, feature].isna().any():
                raise ValueError(f"Column '{feature}' contains NaN values.")

        # build group_features -> impute_feature mapping
        self.group_mapping = X_copy.groupby(self.group_features)[self.impute_feature].describe()

        if X_copy[self.impute_feature].dtype in ["category", "object"]:
            self.group_mapping = self.group_mapping.top
        elif X_copy[self.impute_feature].dtype in ["int", "float", np.number]:
            self.group_mapping = self.group_mapping["50%"]

        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Imputes missing values in the 'impute_feature' by replacing them with the most common value within the group.

        Parameters:
        -----------
          X:
            The input DataFrame to be transformed.
          y:
            The target variable (ignored).

        Returns:
        --------
          The transformed DataFrame (only with 'impute_feature') with missing values imputed
        """
        X_copy = X.copy()

        X_copy.loc[:, self.impute_feature] = X_copy.set_index(self.group_features)[self.impute_feature].fillna(self.group_mapping).to_list()

        return X_copy.loc[:, [self.impute_feature]]
