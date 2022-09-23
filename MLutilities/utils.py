import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from yellowbrick.model_selection import LearningCurve


sns.set()


def cramerv_relationship_strength(degrees_of_freedom: int, cramerv: float):
    """
    returns the strength of the relationship of two categorical variables
    source: https://www.statology.org/interpret-cramers-v/

    Arguments:
    ----------
    degrees_of_freedom:  degrees of freedom obtained from a contingency
                         table as:
                            min(number of rows - 1, number of columns - 1)
    cramerv:             Cramer's V coefficient
    """
    values = {
        "1": [0.10, 0.50],
        "2": [0.07, 0.35],
        "3": [0.06, 0.29],
        "4": [0.05, 0.25],
        "5": [0.04, 0.22],
    }

    if np.round(cramerv, 2) <= values[str(degrees_of_freedom)][0]:
        return "small"
    elif np.round(cramerv, 2) >= values[str(degrees_of_freedom)][-1]:
        return "high"
    else:
        return "medium"


def scaler(
    dataset: pd.DataFrame = None,
    variables: List[str] = None,
    kind: str = "standar_scaler",
):

    """
    Helper function to visualize the efect of scaling and normalization over continuos variables

    Arguments:
    ----------
    dataset:   pandas dataframe or dict with the format {'col1':np.array, 'col2':np.array}
    variables: list with the name of the features to scale
    kind:      name of the transformation to perform. Options ["standar_scaler", "minmax"]
    """

    scale = {
        "standar_scaler": StandardScaler().fit_transform,
        "minmax": MinMaxScaler().fit_transform,
    }

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    for i, var in enumerate(variables):
        sns.kdeplot(dataset[var], ax=ax[0], label=var)
        sns.kdeplot(scale[kind](dataset[[var]]).ravel(), ax=ax[1], label=var)

    ax[0].set_title("Original")
    ax[1].set_title("Transform")
    ax[0].legend()
    ax[1].legend()


def generate_nonlinear_data(N: int, seed: int = 1) -> Tuple:
    """
    generate N (x, y) pairs with a non-linear relationship

    Arguments:
    ----------
    N:     number of instances
    seed:  seed for the random numbers generator
    """
    np.random.seed(seed)
    X = np.random.rand(N) ** 2
    y = 10 - 1 / (X + 0.1) + 2 * np.random.rand(N)
    return X.reshape(-1, 1), y


def poly_reg(estimator, degree: int = 2):
    """
    returns a polinomial regression estimator

    Arguments:
    ----------
    estimator (sklearn estimator): estimator to include in the model pipeline
    degree:                        degree of the PolynomialFeatures transformation
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    scaler = StandardScaler()
    return make_pipeline(poly, scaler, estimator)


def plot_learning_curve(linear_estimator, X, y, degree, ax):
    visualizer = LearningCurve(
        estimator=poly_reg(linear_estimator, degree),
        scoring="r2",
        train_sizes=np.linspace(0.2, 1.0, 12),
        ax=ax,
    )

    visualizer.fit(X, y)
    ax.set_ylim(0, 1.1)
    visualizer.show()


def plot_polyreg(
    N=50, estimator="LinearRegression", degree=2, alpha=1, l1_ratio=1, helper_viz=False
):
    """
    plot a polynomial regression model with and a helper visualization

    Arguments:
    --------
    N:           number of instances
    degree:      degree to use in a PolynomialFeatures transformation
    estimator:   estimator to use in the model pipeline ("LinearRegression", "Ridge", "Lasso" or "ElasticNet")
    alpha:       regularization parameter for Ridge, Lasso and ElasticNet estimators
    l1_ration:   regularization parameter for the ElasticNet estimator
    helper_viz:  helper visualization (weights plot or Learning Curve)
    """
    # train and test data
    X, y = generate_nonlinear_data(N=N, seed=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # model training
    estimators = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=alpha),
        "Lasso": Lasso(alpha=alpha),
        "ElasticNet": ElasticNet(alpha=alpha, l1_ratio=l1_ratio),
    }
    model = poly_reg(estimators[estimator], degree)
    model.fit(X_train, y_train)

    # predictions
    x_pred = np.linspace(0, 1, 100).reshape(-1, 1)
    y_pred = model.predict(x_pred)

    # visualization
    fig, ax = plt.subplots(1, 2, figsize=(25, 8))

    # train and validation scores
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_test, y_test)
    ax[0].scatter(X_train, y_train, label=f"train score: {train_score:.2f}")
    ax[0].scatter(X_test, y_test, label=f"validation score: {val_score:.2f}")

    # model predictions
    ax[0].plot(x_pred, y_pred, "r", label="Model")

    # helper visualization
    if helper_viz == "learning curve":
        plot_learning_curve(estimators[estimator], X, y, degree, ax[1])
    elif helper_viz == "weights":
        ax[1].plot(model[f"{estimator.lower()}"].coef_, "ro--")
        ax[1].set(xlabel="Weight", ylabel="Value")

    ax[0].set_title(f"Estimator: {estimator}\nDegree: {degree}")
    ax[0].legend()


def plot_log_reg(
    threshold: float = 0.5,
    regression: str = "none",
    point_position: int = 12,
) -> None:
    """
    helper visualization to illustrate linear and logistic regressions in a 1D binary classification problem

    Arguments:
    ----------
    threshold:      threshold for the decision region
    regression:     regression to use ("none", "linear", "logistic")
    point_position: position of a positive class point
    """
    n_instances_negative_class = 6
    n_instances_positive_class = 12
    start = 1
    end = 12
    classes_intercept = 5

    instances_negative_class = np.linspace(
        start, classes_intercept + 1, n_instances_negative_class
    )
    instances_positive_class = np.concatenate(
        [
            np.linspace(classes_intercept, end, n_instances_positive_class),
            [point_position],
        ]
    )
    data = np.concatenate([instances_negative_class, instances_positive_class])

    labels_negative_class = np.zeros_like(instances_negative_class)
    labels_positive_class = np.ones_like(instances_positive_class)
    labels = np.concatenate([labels_negative_class, labels_positive_class])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x=data, y=labels, c=labels, cmap="coolwarm")
    ax.set(ylabel="Label", xlabel="Feature")

    if regression != "none":
        # model training
        estimators = {
            "linear": LinearRegression(),
            "logistic": LogisticRegression(),
        }
        model = estimators[regression]
        model.fit(data.reshape(-1, 1), labels)

        # predictions and decision frontier
        x = np.linspace(start, point_position)

        if regression == "linear":
            decision_frontier = (threshold - model.intercept_) / model.coef_[0]
            y_pred = model.predict(x.reshape(-1, 1))
            y_label = "$y=w_0 + w_1 x_1$"
        elif regression == "logistic":
            decision_frontier = (-1 / model.coef_[0]) * (
                model.intercept_ + np.log(1 / threshold - 1)
            )
            y_pred = model.predict_proba(x.reshape(-1, 1))[:, 1]
            y_label = "$\hat P (C_1 | \textbf{x})$"

        ymin = min(y_pred) if min(y_pred) <= 0 else 0
        sns.lineplot(x=x, y=y_pred, color="k", label=y_label)

        # decision regions
        if not decision_frontier <= 0:
            ax.fill_betweenx(
                y=np.linspace(ymin, max(y_pred)),
                x1=start,
                x2=decision_frontier,
                alpha=0.2,
                color="b",
            )
        ax.fill_betweenx(
            y=np.linspace(ymin, max(y_pred)),
            x1=decision_frontier,
            x2=point_position,
            alpha=0.2,
            color="r",
        )

        # threshold line
        ax.hlines(
            threshold,
            start,
            point_position,
            linestyle="--",
            alpha=0.5,
            label=f"Threshold",
        )
        ax.legend(fontsize=15)
