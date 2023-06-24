import pandas as pd
from typing import Optional
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin


class LinearModelImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in a target column using a linear regression model trained with a single feature.

    Parameters:
    -----------
        target (str): The name of the target column to impute.
        feature (str): The name of the feature column used for training the linear regression model.

    Example:
    --------
        # Create an instance of the LinearModelImputer
        imputer = LinearModelImputer(target='target_column', feature='feature_column')
        imputer.set_output(transform="pandas")

        # Fit the imputer on the training data
        imputer.fit_transform(X_train[[target, feature]])

        # Impute missing values in the target column of the test data
        X_test_imputed = imputer.transform(X_test[[target, feature]])

    Raises:
    -------
        ValueError: If the feature column contains NaN values or no numeric columns are found in the input data.
    """

    def __init__(self, target: str, feature: str = None):
        self.target = target
        self.feature = feature
        self.linear_model = LinearRegression()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the linear regression model using non-null instances from the feature and target columns of the input data.

        Parameters:
        -----------
            X:
              The input data containing the feature and target columns.
            y:
              Label vector (default: None).

        Returns:
            LinearModelImputer

        """
        X_copy = X.copy()

        # if feature column is not provided, select the most correlated numeric column
        if self.feature is None:
            numeric_cols = (
                X_copy.select_dtypes(include=np.number).columns.copy().drop(self.target)
            )
            target_col = X_copy.loc[:, self.target]

            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in the input data.")

            correlation_matrix = X_train.corr(method="spearman", numeric_only=True)
            most_correlated_feature = (
                correlation_matrix.loc[self.target].abs().iloc[1:].idxmax()
            )
            self.feature = most_correlated_feature

        # check that feature column does not have missing values
        if X_copy.loc[:, self.feature].isna().any():
            raise ValueError("The feature column contains NaN values.")

        # use instances with no missing target values to train the linear model
        feature_train = X_copy.loc[~X_copy.loc[:, self.target].isna(), [self.feature]]
        target_train = X_copy.loc[~X_copy.loc[:, self.target].isna(), self.target]

        # train the linear model
        self.linear_model.fit(feature_train, target_train)

        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Replace missing values in the target column with predicted values from the trained linear regression model.

        Parameters:
        -----------
            X:
              The input data containing the feature and target columns.
            y:
              Label vector (default: None).

        Returns:
            The modified target column with missing values replaced by predictions.

        """
        X_copy = X.copy()
      
        # extract instances with missing target values and their corresponding feature values
        feature_test = X_copy.loc[X_copy.loc[:, self.target].isna(), [self.feature]]
      
        # predict and replace the missing target values using the trained linear regression model
        target_predict = self.linear_model.predict(feature_test)
        X_copy.loc[X_copy.loc[:, self.target].isna(), self.target] = target_predict

        return X_copy.loc[:, self.target]

    def get_feature_names_out(self):
        """
        Returns the output feature names after transformation (needed to use 'set_output(transform="pandas")')
        """
        pass
