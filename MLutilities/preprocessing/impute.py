import numpy as np
import pandas as pd
from typing import Optional, List
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin


class LinearModelImputer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for imputing missing values based on a linear regression.
    It replaces missing values in the 'impute_feature' column of a DataFrame by imputing
    values predicted from a linear regression model trained with a single feature.

    Parameters:
    -----------
      impute_feature:
        The name of the feature to be imputed (i.e., the feature with missing values).
      feature:
        The name of the feature column used for training the linear regression model.

    Raises:
    -------
        ValueError: If the feature column contains NaN values or no numeric columns are found in the input data.
    """

    def __init__(self, impute_feature: str, feature: str = None):
        self.impute_feature = impute_feature
        self.feature = feature
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
        if self.feature is None:
            numeric_cols = X_copy.columns.copy().drop(self.impute_feature)

            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in the input data.")

            correlations = X_copy.corr(method="spearman")
            most_correlated_feature = (
                correlations.loc[self.impute_feature].abs().iloc[1:].idxmax()
            )
            self.feature = most_correlated_feature

        # check that feature column does not have missing values
        if X_copy.loc[:, self.feature].isna().any():
            raise ValueError("The feature column contains NaN values.")

        # get instances with impute_feature missing values
        nan_instances = X_copy.loc[:, self.impute_feature].isna()

        # use instances with no missing impute_feature values to train the linear model
        feature_train = X_copy.loc[~nan_instances, [self.feature]]
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
            The modified impute_feature column with missing values replaced by predictions.

        """
        X_copy = X.copy().select_dtypes(include=np.number)

        # get instances with impute_feature missing values
        nan_instances = X_copy.loc[:, self.impute_feature].isna()

        # extract instances with missing impute_feature values and their corresponding feature values
        feature_test = X_copy.loc[nan_instances, [self.feature]]

        # predict and replace the missing impute_feature values using the trained linear regression model
        target_predict = self.linear_model.predict(feature_test)
        X_copy.loc[nan_instances, self.impute_feature] = target_predict

        return X_copy.loc[:, [self.impute_feature]]

    def get_feature_names_out(self):
        """
        Returns the output feature names after transformation (needed to use 'set_output(transform="pandas")')
        """
        pass


class GroupImputer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for imputing missing values based on group-wise statistics.
    It replaces missing values in the 'impute_feature' column of a DataFrame
    by imputing the most common value within each group defined by the specified 'group_features'

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
        self.group_mapping = (
            X_copy.groupby(self.group_features)[self.impute_feature].describe().top
        )

        # if there is missing values in the mapping, replace them by the 'impute_feature' mode
        self.impute_feature_mode = X_copy.loc[:, self.impute_feature].describe().top
        if self.group_mapping.isna().any():
            self.group_mapping.loc[self.group_mapping.isna()] = self.impute_feature_mode

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

        # get index to sample from 'group_mapping'
        index = (
            X_copy.loc[X_copy[self.impute_feature].isna()]
            .set_index(self.group_features)
            .index
        )

        # initialize series with imputed values
        imputed_values = pd.Series(index=index, dtype="object")

        # if index is not present in 'group_mapping' keys, replace its value by 'impute_feature_mode'
        diff_ids = index.difference(self.group_mapping.index)
        if not diff_ids.empty:
            imputed_values.loc[diff_ids.values] = self.impute_feature_mode

        # sample values from group_mapping
        common_ids = index.intersection(self.group_mapping.index)
        imputed_values.loc[common_ids] = self.group_mapping.get(common_ids)

        # impute missing values
        X_copy.loc[
            X_copy[self.impute_feature].isna(), self.impute_feature
        ] = imputed_values.values

        return X_copy.loc[:, [self.impute_feature]]

    def get_feature_names_out(self):
        pass
