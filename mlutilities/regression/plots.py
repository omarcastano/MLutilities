import numpy as np
from plotly import graph_objects as go
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from mlutilities.regression.models import PolynomialRegression
from mlutilities.regression.utils import generate_nonlinear_data
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


def plot_poly_reg(degree: int, N: int = 50) -> None:
    """
    helper visualization function to see the results of a fitted polynomial regression model on non-linear data

    Parameters:
    -----------
      degree:
        Degree of the polynomial regression model
      N:
        Number of instances of the generated non-linear data
    """
    X, y = generate_nonlinear_data(N)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = PolynomialRegression(degree=degree)
    model.fit(X_train, y_train)
    model.plot_fitted_model(X_train, y_train, X_test, y_test)


def compare_learning_curves(
    estimator: str = "Ridge",
    degree: int = 13,
    N: int = 50,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
) -> None:
    """
    Helper visualization function that compares the learning curves of two polynomial regression models:
    one using a linear estimator and the other using a regularized linear estimator.

    Parameters:
    -----------
    estimator:
      The estimator to use for the learning curve. Options: "Ridge", "Lasso", "ElasticNet".
    degree:
      The degree of polynomial regression.
    N:
      The number of data points to generate.
    alpha:
      The regularization strength for Ridge, Lasso, and ElasticNet estimators.
    l1_ratio:
      The ElasticNet mixing parameter.
    """
    # generate data
    X, y = generate_nonlinear_data(N=N)

    # definicion del modelo
    def poly_regression(degree=2, estimator="Linear"):
        estimators = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=alpha),
            "Lasso": Lasso(alpha=alpha),
            "ElasticNet": ElasticNet(alpha=alpha, l1_ratio=l1_ratio),
        }

        poly_transformer = PolynomialFeatures(degree=degree, include_bias=False)

        return make_pipeline(poly_transformer, estimators[estimator])

    # linear regression scores
    lr = poly_regression(degree, "Linear")
    train_sizes, train_scores, test_scores = learning_curve(
        lr,
        X,
        y,
        cv=10,
        n_jobs=1,
        scoring="r2",
        train_sizes=np.linspace(0.2, 1, 25),
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # regularized linear model scores
    reg_model = poly_regression(degree, estimator)
    _, rtrain_scores, rtest_scores = learning_curve(
        reg_model,
        X,
        y,
        cv=10,
        n_jobs=1,
        scoring="r2",
        train_sizes=np.linspace(0.2, 1, 25),
    )
    rtrain_scores_mean = np.mean(rtrain_scores, axis=1)
    rtest_scores_mean = np.mean(rtest_scores, axis=1)

    # plot learning curves
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_scores_mean,
            mode="lines",
            name="Linear train score",
            line=dict(color="blue", dash="solid"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=test_scores_mean,
            mode="lines",
            name="Linear test score",
            line=dict(color="blue", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=rtrain_scores_mean,
            mode="lines",
            name=f"{estimator} train score",
            line=dict(color="red", dash="solid"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=rtest_scores_mean,
            mode="lines",
            name=f"{estimator} test score",
            line=dict(color="red", dash="dash"),
        )
    )
    fig.update_yaxes(range=[0, 1.2])

    fig.update_layout(
        title="Learning Curves",
        xaxis_title="Training Set Size",
        yaxis_title="Score",
        width=1300,
        height=600,
    )
    fig.show()


def plot_regularized_poly_reg(estimator: str = "Linear", degree: int = 2, N: int = 50, alpha: int = 1, l1_ratio: float = 0.5) -> None:
    """
    Perform regularized polynomial regression using the specified estimator and plot the fitted model.

    Parameters:
    -----------
        estimator:
          The name of the estimator to use for regularized polynomial regression.
          Available options: "Linear", "Ridge", "Lasso", "ElasticNet". Default is "Linear".

        degree:
          The degree of the polynomial regression model. Default is 2.

        N:
          The number of data points to generate for training and testing. Default is 50.

        alpha:
          Regularization strength (alpha) for Ridge, Lasso, and ElasticNet regressions. Default is 1.

        l1_ratio:
          ElasticNet mixing parameter (l1_ratio) between L1 and L2 regularization. Default is 0.5.
    """
    # model training
    estimators = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=alpha),
        "Lasso": Lasso(alpha=alpha),
        "ElasticNet": ElasticNet(alpha=alpha, l1_ratio=l1_ratio),
    }
    # generate data
    X, y = generate_nonlinear_data(N=N)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = PolynomialRegression(degree=degree, estimator=estimators[estimator])
    model.fit(X_train, y_train)
    model.plot_fitted_model(X_train, y_train, X_test, y_test)
