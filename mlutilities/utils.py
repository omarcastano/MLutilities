import logging
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.integrate as integrate
from IPython.display import display
from scipy.stats import gaussian_kde
from typing import List, Tuple, Any, Dict
from plotly.subplots import make_subplots
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from yellowbrick.model_selection import LearningCurve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
)

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





def generate_nonlinear_data(N: int, seed: int = 1) -> Tuple:
    logging.warning(
        """
    ---------------------------------------------------------------------------------
    This function (MLutilities.utils.generate_nonlinear_data) will be deprecated use 
    MLutilities.regression.utils.generate_nonlinear_data instead
    ---------------------------------------------------------------------------------
    ---------
    >>> from MLutilities.regression.utils import generate_nonlinear_data
    >>> X, y = generate_nonlinear_data(N=50)
    >>> print(X.shape)
    (50, 1)
    """
    )

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
    logging.warning(
        """
    ---------------------------------------------------------------------------------
    This function (MLutilities.utils.poly_reg) will be deprecated use 
    MLutilities.regression.PolynomialRegression  instead
    ---------------------------------------------------------------------------------
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
    )

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
    logging.warning(
        """
    ---------------------------------------------------------------------------------
    This function (MLutilities.utils.plot_learning_curve) will be deprecated, use 
    MLutilities.model_selection.plot_learning_curve   instead
    ---------------------------------------------------------------------------------
    Example:
    ---------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from MLutilities.model_selection import plot_learning_curve
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    >>> clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    >>> plot_learning_curve(clf, X_train, y_train, scoring='accuracy')
    """
    )
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
    N=50,
    estimator: str = "LinearRegression",
    degree: int = 2,
    alpha: float = 1.0,
    l1_ratio: float = 1.0,
    helper_viz: bool = False,
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

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
    logging.warning(
        """
        ---------------------------------------------------------------------
        This function (MLutilities.utils.plot_log_reg) will be deprecated, 
        use MLutilities.classification.plot_1d_classification instead
        ---------------------------------------------------------------------
        """
    )
    
    n_instances_negative_class = 6
    n_instances_positive_class = 12
    start = 1
    end = 12
    classes_intercept = 5

    instances_negative_class = np.linspace(start, classes_intercept + 1, n_instances_negative_class)
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
            decision_frontier = (-1 / model.coef_[0]) * (model.intercept_ + np.log(1 / threshold - 1))
            y_pred = model.predict_proba(x.reshape(-1, 1))[:, 1]
            y_label = r"$\hat{p} = \sigma(z)$"

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
            label="Threshold",
        )
        ax.legend(fontsize=15)


def highlight_quadrant(quadrant: int, width: int = 8, color: str = "r", ax=None):
    """
    higlight confusion matrix quadrant
    """
    x_left, x_right = ax.get_xlim()
    y_down, y_up = ax.get_ylim()
    x_middle = (x_left + x_right) / 2
    y_middle = (y_down + y_up) / 2

    if quadrant == 0:
        ax.vlines(x_left, y_middle, y_up, color=color, linewidth=width)
        ax.vlines(x_middle, y_middle, y_up, color=color, linewidth=width / 2)
        ax.hlines(y_up, x_left, x_middle, color=color, linewidth=width)
        ax.hlines(y_middle, x_left, x_middle, color=color, linewidth=width / 2)
    if quadrant == 1:
        ax.vlines(x_right, y_middle, y_up, color=color, linewidth=width)
        ax.vlines(x_middle, y_middle, y_up, color=color, linewidth=width / 2)
        ax.hlines(y_up, x_middle, x_right, color=color, linewidth=width)
        ax.hlines(y_middle, x_middle, x_right, color=color, linewidth=width / 2)
    if quadrant == 2:
        ax.vlines(x_left, y_down, y_middle, color=color, linewidth=width)
        ax.vlines(x_middle, y_down, y_middle, color=color, linewidth=width / 2)
        ax.hlines(y_down, x_left, x_middle, color=color, linewidth=width)
        ax.hlines(y_middle, x_left, x_middle, color=color, linewidth=width / 2)
    if quadrant == 3:
        ax.vlines(x_right, y_down, y_middle, color=color, linewidth=width)
        ax.vlines(x_middle, y_down, y_middle, color=color, linewidth=width / 2)
        ax.hlines(y_down, x_middle, x_right, color=color, linewidth=width)
        ax.hlines(y_middle, x_middle, x_right, color=color, linewidth=width / 2)


def highlight_quadrants(metric: str, ax=None):
    """
    highlight confusion matrix quadrants associated with a metric
    """
    metrics_data = get_metrics_data()
    quadrant = metrics_data[metric]["quadrant"]
    for i in quadrant:
        highlight_quadrant(quadrant=i, ax=ax)


def get_metrics_data() -> dict:
    """
    returns a dictionary with metrics functions, formulas and confusion matrix quadrants
    """
    metrics_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "tpr",
        "tnr",
        "fnr",
        "fpr",
        "npv",
    ]

    # functions to compute each metric in terms of the confusion matrix entries
    def accuracy(tn, fp, fn, tp):
        return (tp + tn) / (tp + fp + tn + fn + 10e-8)

    def precision(tn, fp, fn, tp):
        return tp / (tp + fp + 10e-8)

    def recall(tn, fp, fn, tp):
        return tp / (tp + fn + 10e-8)

    def f1_score(tn, fp, fn, tp):
        num = (tp / (tp + fp + 10e-8)) * (tp / (tp + fn + 10e-8))
        den = (tp / (tp + fp)) + (tp / (tp + fn))
        return 2 * num / den

    def tpr(tn, fp, fn, tp):
        return tp / (tp + fn + 10e-8)

    def tnr(tn, fp, fn, tp):
        return tn / (tn + fp + 10e-8)

    def fnr(tn, fp, fn, tp):
        return fn / (fn + tp + 10e-8)

    def fpr(tn, fp, fn, tp):
        return fp / (fp + tn + 10e-8)

    def npv(tn, fp, fn, tp):
        return tn / (tn + fn + 10e-8)

    metrics_functions = [
        accuracy,
        precision,
        recall,
        f1_score,
        tpr,
        tnr,
        fnr,
        fpr,
        npv,
    ]

    # confusion matrix quadrants asociated with each metric
    metric_quadrants = [
        [0, 1, 2, 3],
        [1, 3],
        [3, 2],
        [3, 1, 2],
        [3, 2],
        [0, 1],
        [3, 2],
        [0, 1],
        [0, 2],
    ]

    # mathematical expression for each metric
    metrics_formulas = [
        r"$\frac{TP + {TN}}{{TP}+{TN}+{FP}+{FN}}$",
        r"$\frac{TP}{TP + FP}$",
        r"$\frac{TP}{TP+FN}$",
        r"$2 \frac{P*R}{P+R}$",
        r"$\frac{TP}{TP+FN}$",
        r"$\frac{TN}{TN + FP}$",
        r"$\frac{FN}{FN + TP}$",
        r"$\frac{FP}{FP + TN}$",
        r"$\frac{TN}{TN + FN}$",
    ]

    metrics_data = {}
    for i, key in enumerate(metrics_keys):
        metrics_data[key] = {}
        metrics_data[key]["function"] = metrics_functions[i]
        metrics_data[key]["formula"] = metrics_formulas[i]
        metrics_data[key]["quadrant"] = metric_quadrants[i]

    return metrics_data


def get_xlims(mean, sigma=0.082):
    """compute normal distribution x limits"""
    xmin = mean - 3 * sigma
    xmax = mean + 3 * sigma
    return xmin, xmax


def get_normal_dist(mean, sigma=0.082):
    """compute normal distribution coordinates"""
    xmin, xmax = get_xlims(mean, sigma)
    x = np.linspace(xmin, xmax, 100)
    y = stats.norm.pdf(x, mean, sigma)
    return x, y


def get_area(min, max, mean, sigma=0.082):
    """compute the area under a normal distribution"""

    def f(x, mean):
        return stats.norm.pdf(x, mean, sigma)

    area = integrate.quad(f, min, max, args=(mean))[0]
    return area


def fill_region(mean, min, max, sigma=0.082, label=None, color=None, ax=None):
    """fill some region of a normal distribution between min and max"""
    x = np.linspace(min, max, 100)
    y = stats.norm.pdf(x, mean, sigma)
    ax.fill_between(x, y, facecolor=color, alpha=0.2, label=label)


def fill_false_regions(pos_dist_mean, neg_dist_mean, threshold=0.5, ax=None):
    """plot false and positive regions"""
    _, pos_dist_max = get_xlims(pos_dist_mean)
    neg_dist_min, _ = get_xlims(neg_dist_mean)

    pos_dist_x, pos_dist = get_normal_dist(pos_dist_mean)
    neg_dist_x, neg_dist = get_normal_dist(neg_dist_mean)

    if neg_dist_min < threshold:
        fill_region(
            mean=neg_dist_mean,
            min=neg_dist_min,
            max=threshold,
            label="FP",
            color="r",
            ax=ax,
        )

    if pos_dist_max > threshold:
        fill_region(
            mean=pos_dist_mean,
            min=threshold,
            max=pos_dist_max,
            label="FN",
            color="b",
            ax=ax,
        )


def plot_probability_distributions(pos_dist_mean, neg_dist_mean, threshold=0.5, ax=None):
    """
    plot one-dimensional probability distributions for a binary classifier
    """
    pos_dist_x, pos_dist = get_normal_dist(pos_dist_mean)
    neg_dist_x, neg_dist = get_normal_dist(neg_dist_mean)

    ax.plot(pos_dist_x, pos_dist, label="Positive class")
    ax.plot(neg_dist_x, neg_dist, label="Negative class")
    ax.set_title("Probability distributions")
    fill_false_regions(pos_dist_mean, neg_dist_mean, threshold, ax)


@np.vectorize
def get_classification_results(pos_dist_mean, neg_dist_mean, threshold):
    """compute true and false positive/negative values"""
    pos_dist_min, pos_dist_max = get_xlims(pos_dist_mean)
    neg_dist_min, neg_dist_max = get_xlims(neg_dist_mean)

    # area under a normal curve
    auc = get_area(0, 0.5, 0.25)

    # positive class scenarios
    if threshold <= pos_dist_min:
        TP = 0
        FN = auc
    elif threshold >= pos_dist_max:
        TP = auc
        FN = 0
    else:
        TP = get_area(pos_dist_min, threshold, pos_dist_mean)
        FN = get_area(threshold, pos_dist_max, pos_dist_mean)

    # negative class scenarios
    if threshold <= neg_dist_min:
        TN = auc
        FP = 0
    elif threshold >= neg_dist_max:
        TN = 0
        FP = auc
    else:
        TN = get_area(threshold, neg_dist_max, neg_dist_mean)
        FP = get_area(neg_dist_min, threshold, neg_dist_mean)

    return TP, TN, FP, FN


@np.vectorize
def get_positive_rates(pos_dist_mean, neg_dist_mean, threshold):
    """compute false and positive rates"""
    TP, TN, FP, FN = get_classification_results(pos_dist_mean, neg_dist_mean, threshold)
    fpr = FP / (FP + TN + 1e-5)
    tpr = TP / (TP + FN + 1e-5)
    return fpr, tpr


def plot_roc_curve(pos_dist_mean, neg_dist_mean, ax):
    # false and positive rates
    threshold = np.linspace(0, 1)
    fpr, tpr = get_positive_rates(pos_dist_mean, neg_dist_mean, threshold)

    # area under the curve
    auc = np.trapz(tpr, fpr)

    ax.set(title="ROC Curve", xlabel="False positive rate", ylabel="True positive rate")
    ax.plot([0, 1], [0, 1], "k--")
    ax.plot(fpr, tpr, "r", label=f"AUC {auc:.2f}")
    ax.legend()


def roc_curve_viz(pos_dist_mean=0.25, neg_dist_mean=0.75):
    # probability distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 8))
    plot_probability_distributions(pos_dist_mean, neg_dist_mean, ax=ax1)

    # threshold and decision regions
    threshold = 0.5
    ax1.vlines(threshold, 0, 5)
    ax1.axvspan(0, threshold, alpha=0.1, color="blue")
    ax1.axvspan(threshold, 1, alpha=0.1, color="darkorange")
    ax1.axis([0, 1, 0, 5])
    ax1.legend()

    # ROC curve
    plot_roc_curve(pos_dist_mean, neg_dist_mean, ax=ax2)


def plot_iris_decision_tree(max_depth: int = 1):
    """
    trains a decision tree on the iris dataset and shows the tree and decision regions

    Arguments:
    ----------
        max_depth:
            Decision tree maximum depth
    """
    data = sns.load_dataset("iris")
    X = data[["petal_length", "petal_width"]]
    y = data["species"].map({"setosa": 0, "versicolor": 1, "virginica": 2})

    dt_clf = DecisionTreeClassifier(random_state=42, max_depth=max_depth).fit(X.values, y.values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    sns.scatterplot(data=data, x="petal_length", y="petal_width", hue="species", ax=ax1)
    plot_decision_regions(X.values, y.values, dt_clf, legend=0, ax=ax1)
    plot_tree(dt_clf, feature_names=["petal_length", "petal_width"], filled=True, ax=ax2)


def plot_kernel_pca(
    df: pd.DataFrame,
    target: str,
    n_components: int = 2,
    standarized: bool = True,
    kernel: str = "linear",
    coef0: float = 1,
    gamma: str = None,
    degree: int = 2,
):
    """
    Reduce the dimensionality of a dataset using a kernelized PCA and plot the result.

    Arguments:
    ----------
    df:
        dataset
    target:
        target variable
    n_components:
        Number of principal components (2 or 3)
    standarized:
        If True standarizes the features before the PCA
    kernel:
        kernel to use in PCA {linear, poly, rbf, sigmoid, cosine}
    coef0:
        Independent term in poly and sigmoid kernels. Ignored by other kernels
    gamma:
        Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other kernels
    degree:
        Degree for poly kernels. Ignored by other kernels
    """
    X = df.drop(columns=target).select_dtypes(exclude="object")
    y = df[target]

    pca = KernelPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
    )

    X_pca = make_pipeline(StandardScaler(), pca).fit_transform(X) if standarized else pca.fit_transform(X)

    labels = {"x": "PC1", "y": "PC2"} if n_components == 2 else {"x": "PC1", "y": "PC2", "z": "PC3"}

    if n_components == 2:
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=y, labels=labels)
    elif n_components == 3:
        fig = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], color=y, labels=labels)
    fig.show()


def show_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_color: str = "yellow",
    right_color: str = "orange",
    **merge_kwargs: Dict[str, Any],
) -> None:
    """
    Helper function to visualize the merge of two Pandas DataFrames.

    Arguments:
    ----------
        left : pandas.DataFrame
            The left DataFrame to merge.
        right : pandas.DataFrame
            The right DataFrame to merge.
        left_color : str, optional
            The background color for the left DataFrame, by default "yellow".
        right_color : str, optional
            The background color for the right DataFrame, by default "orange".
        **merge_kwargs
            Additional keyword arguments to pass to `pd.merge()`.

    Example:
    --------
    >>> left = pd.DataFrame({"key": ["A", "B", "C"], "value": [1, 2, 3]})
    >>> right = pd.DataFrame({"key": ["B", "C", "D"], "value": [4, 5, 6]})
    >>> show_merge(left, right, how="inner", on="key")
    """
    # display left and right DataFrames
    colors = {"left": left_color, "right": right_color}
    print(f"left:")
    display(left.style.set_properties(**{"background-color": colors["left"]}))
    print(f"\nright:")
    display(right.style.set_properties(**{"background-color": colors["right"]}))

    # determine which columns are unique to each DataFrame
    # and assign them to left_col and right_col respectively
    if not set(left.columns).intersection(right.columns):
        left_col = list(left.columns)
        right_col = list(right.columns)
    else:
        left_col = set(left.columns) - set(right.columns)
        right_col = set(right.columns) - set(left.columns)

    # create a string representation for merge_kwargs
    str_kwargs = ""
    for key, value in merge_kwargs.items():
        str_kwargs += f"{key}='{value}', "
    str_kwargs = str_kwargs.rstrip(", ")

    # create a dictionary of styles for the merged DataFrame
    style_keys = {}
    for lcol, rcol in zip(left_col, right_col):
        style_keys[lcol] = [{"selector": "td", "props": f"background-color:{colors['left']}"}]
        style_keys[rcol] = [{"selector": "td", "props": f"background-color:{colors['right']}"}]

    # display left and right merge
    merge_df = pd.merge(left, right, **merge_kwargs)
    print(f"\npd.merge(left, right, {str_kwargs})")
    display(merge_df.style.set_table_styles({**style_keys}))


def show_join(left: pd.DataFrame, right: pd.DataFrame, **join_kwargs: Dict[str, Any]) -> None:
    """
    Helper function to visualize the join of two Pandas DataFrames.

    Arguments
    ----------
        left : pandas.DataFrame
            The left DataFrame to join.
        right : pandas.DataFrame
            The right DataFrame to join.
        **join_kwargs
            Additional keyword arguments to pass to `pd.DataFrame.join()`.
    """
    # display left and right DataFrames
    print(f"left:")
    display(left)
    print(f"\nright:")
    display(right)

    # create a string representation for join_kwargs
    str_kwargs = ""
    for key, value in join_kwargs.items():
        str_kwargs += f"{key}='{value}', "
    str_kwargs = str_kwargs.rstrip(", ")

    # display left and right join
    print(f"\nleft.join(right, {str_kwargs})")
    display(left.join(right, **join_kwargs))
