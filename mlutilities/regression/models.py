from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from plotly import graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

class PolynomialRegression:
    """
    Fits a polynomial regression model.

    Arguments:
    ----------
    degree: int
        Degree of the polynomial.
    estimator: sklearn estimator, optional
        Estimator to use. Defaults to `LinearRegression()`.

    Example:
    ---------
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.linear_model import LinearRegression
    >>> from MLutilities.regression import PolynomialRegression
    >>> X, y = load_boston(return_X_y=True)
    >>> model = PolynomialRegression(degree=2, estimator=LinearRegression())
    >>> model.fit(X, y)
    >>> model.plot_fitted_model(X, y)
    """

    def __init__(self, degree, estimator=LinearRegression()):
        self.degree = degree
        self.model = Pipeline([("poly", PolynomialFeatures(degree=degree, include_bias=False)), ("linear", estimator)])

    def fit(self, X, y):
        """
        Fits the model.

        Arguments:
        ----------
        X: array-like
            Training data.
        y: array-like
            Training labels.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predicts the labels for the given data.

        Arguments:
        ----------
        X: array-like
            Data for which to predict labels.

        Returns:
        --------
        array-like
            Predicted labels.
        """
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def get_feature_names(self):
        return self.model.named_steps["poly"].get_feature_names()

    def get_coef(self):
        return self.model.named_steps["linear"].coef_

    def get_intercept(self):
        return self.model.named_steps["linear"].intercept_

    def plot_fitted_model(self, X_train, y_train, X_test=None, y_test=None):
        """
        Plot the fitted model.

        Arguments:
        ----------
            X_train: array-like
                Training data.
            y_train: array-like
                Training labels.
            X_test: array-like, optional
                Testing data.
            y_test: array-like, optional
                Testing labels.
        """
        mae_test = np.mean(np.abs((self.predict(X_test) - y_test)))
        r2_test = self.score(X_test, y_test)

        mae_train = np.mean(np.abs((self.predict(X_train) - y_train)))
        r2_train = self.score(X_train, y_train)

        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(
            go.Scatter(
                x=X_train.ravel(), y=y_train, mode="markers", name=f"Training data: <br>MAE: {mae_train:.3f} <br>R^2: {r2_train:.3f}"
            )
        )
        if X_test is not None and y_test is not None:
            fig.add_scatter(
                x=X_test.ravel(), y=y_test, mode="markers", name=f"Test data: <br>MAE: {mae_test:.3f} <br>R^2: {r2_test:.3f}"
            )

        x_dummy = np.linspace(X_train.min(), X_train.max(), 100)
        fig.add_trace(go.Scatter(x=x_dummy, y=self.predict(x_dummy.reshape(-1, 1)).ravel(), mode="lines", name="Fitted model"))

        # Second plot (Weights of the fitted curve)
        weights = self.get_coef()
        x_weights = np.arange(len(weights))

        fig.add_trace(go.Scatter(x=x_weights, y=weights, mode="lines+markers", name="Weights", line=dict(dash="dash")), row=1, col=2)

        fig.update_xaxes(title_text="X", row=1, col=1)
        fig.update_yaxes(title_text="y", row=1, col=1)
        fig.update_xaxes(title_text="Weights", row=1, col=2)
        fig.update_yaxes(title_text="Values", row=1, col=2)

        fig.update_layout(
            title={"text": "Polynomial regression", "x": 0.45, "xanchor": "center", "yanchor": "middle", "font": {"size": 24}},
            width=1300,
            height=600,
        )

        fig.show()
