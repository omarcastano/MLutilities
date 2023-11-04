import pandas as pd
import ipywidgets as widgets
import numpy.typing as npt
from functools import partial
from IPython.display import display
from mlutilities.utils import (
    plot_log_reg,
    roc_curve_viz,
    plot_iris_decision_tree,
    plot_kernel_pca,
)
from mlutilities.Classification import metrics
import logging


logging.warning(
    """
    ---------------------------------------------------------------------------
    The module Classification will be remove. Please use classification instead.     
    ----------------------------------------------------------------------------
    """
)


def logistic_regression_widget():
    """
    helper widget to illustrate linear and logistic regressions in a 1D binary classification problem
    """
    threshold = widgets.FloatSlider(
        description="Threshold:",
        min=0.01,
        max=0.99,
        value=0.5,
        step=0.01,
        continuous_update=False,
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )

    regression = widgets.Dropdown(
        options=["none", "linear", "logistic"],
        description="Regression:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )

    point_position = widgets.IntSlider(
        description="Point position",
        min=12,
        max=40,
        value=12,
        step=4,
        continuous_update=False,
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )

    w = widgets.interactive_output(
        partial(plot_log_reg),
        {
            "threshold": threshold,
            "regression": regression,
            "point_position": point_position,
        },
    )

    display(widgets.VBox([regression, threshold, point_position]), w)


def metrics_evaluation_widget(y_true: npt.ArrayLike, y_score: npt.ArrayLike, label: str = None) -> None:
    """
    Widget that plot the value of a given metric for several probability thresholds.
    This function only work for a binary classification problem.

    Arguments:
          y_true: (1D)
              true labels
          y_score: (2D)
              predicted scores for positive and negative class
          label: (optional) Name of the model
    """
    threshold = widgets.FloatSlider(
        description="Threshold:",
        min=0.0,
        max=1,
        value=0.5,
        step=0.01,
        continuous_update=True,
        layout=widgets.Layout(width="20%", height="50px"),
    )

    metric = widgets.Dropdown(
        options=[
            "accuracy",
            "precision",
            "recall",
            "npv",
            "tnr",
            "f1_score",
            "fpr",
            "fnr",
        ],
        description="Metric:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )

    w = widgets.interactive_output(
        partial(
            metrics.threshold_metric_evaluation,
            y_true=y_true,
            y_score=y_score,
            label=label,
        ),
        {"metric": metric, "threshold": threshold},
    )

    display(widgets.VBox([metric, threshold]), w)


def precision_recall_tradeoff_widget(y_true: npt.ArrayLike, y_predict_proba: npt.ArrayLike) -> None:
    """
    This function allows you to evaluate the precision-recall tradeoff

      Arguments:
          y_true: (1D)
              true labels
          y_score: (2D)
              predicted scores such as probability
    """
    threshold = widgets.FloatSlider(
        description="Threshold:",
        min=0.0,
        max=1,
        value=0.5,
        step=0.01,
        continuous_update=True,
        layout=widgets.Layout(width="20%", height="50px"),
    )
    w = widgets.interactive_output(
        partial(metrics.precision_recall_tradeoff, y_true=y_true, y_score=y_predict_proba),
        {"threshold": threshold},
    )
    display(widgets.VBox([threshold]), w)


def precision_recall_widget(y_true: npt.ArrayLike, y_score: npt.ArrayLike):
    """
    Plots Preciison recall curve and PR AUC

    Arguments:
        y_true: (1D)
            true labels
        y_score: (2D)
            predicted scores such as probability
    """
    threshold = widgets.FloatSlider(
        description="Threshold:",
        min=0.0,
        max=1,
        value=0.5,
        step=0.01,
        continuous_update=True,
        layout=widgets.Layout(width="20%", height="50px"),
    )
    w = widgets.interactive_output(
        partial(metrics.precision_recall_curve, y_true=y_true, y_score=y_score),
        {"threshold": threshold},
    )
    display(widgets.VBox([threshold]), w)


def roc_curve_widget(y_true: npt.ArrayLike, y_score: npt.ArrayLike):
    """
    Compute ROC curve and AUC

    Arguments:
    ----------
        y_true: (1D)
            true labels
        y_score: (2D)
            predicted scores such as probability
    """
    threshold = widgets.FloatSlider(
        description="Threshold:",
        min=0.0,
        max=1,
        value=0.5,
        step=0.01,
        continuous_update=True,
        layout=widgets.Layout(width="20%", height="50px"),
    )
    w = widgets.interactive_output(
        partial(metrics.ROC_curve, y_true=y_true, y_score=y_score),
        {"threshold": threshold},
    )
    display(widgets.VBox([threshold]), w)


def roc_widget():
    pos_dist_mean = widgets.FloatSlider(
        description="Positive class mean:",
        min=0.25,
        max=0.75,
        value=0.25,
        step=0.01,
        continuous_update=False,
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )

    neg_dist_mean = widgets.FloatSlider(
        description="Negative class mean:",
        min=0.25,
        max=0.75,
        value=0.75,
        step=0.01,
        continuous_update=False,
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )

    w = widgets.interactive_output(
        partial(roc_curve_viz),
        {"pos_dist_mean": pos_dist_mean, "neg_dist_mean": neg_dist_mean},
    )
    display(widgets.HBox([pos_dist_mean, neg_dist_mean]), w)


def iris_decision_tree_widget():
    """
    Helper widget to visualize a decision tree classifier on the iris dataset
    """
    max_depth = widgets.IntSlider(
        description="Max Depth:",
        min=1,
        max=10,
        value=1,
        step=1,
        continuous_update=False,
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )

    w = widgets.interactive_output(
        partial(
            plot_iris_decision_tree,
        ),
        {"max_depth": max_depth},
    )

    display(widgets.VBox([max_depth]), w)


def kernelPCA_widget(dataset: pd.DataFrame, target: str):
    """
    Helper widget to visualize the effect of a kernelized PCA transformation

    Arguments:
    ----------
        dataset:
            pandas dataframe having the input data
        target:
            name of the target variable
    """
    n_components = widgets.Dropdown(
        description="Number of components:",
        options=[2, 3],
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    standarized = widgets.Dropdown(
        options=[True, False],
        description="Standarize:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    kernel = widgets.Dropdown(
        options=[
            "linear",
            "poly",
            "rbf",
            "sigmoid",
            "cosine",
        ],
        description="kernel:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    coef0 = widgets.FloatSlider(
        description="coef0:",
        min=-5,
        max=5,
        value=1,
        step=0.5,
        continuous_update=False,
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    gamma = widgets.FloatSlider(
        description="gamma:",
        min=0.01,
        max=10,
        value=0.01,
        step=0.01,
        continuous_update=False,
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    degree = widgets.IntSlider(
        description="degree:",
        min=2,
        max=10,
        value=2,
        step=1,
        continuous_update=False,
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )

    w = widgets.interactive_output(
        partial(plot_kernel_pca, dataset, target),
        {
            "n_components": n_components,
            "standarized": standarized,
            "kernel": kernel,
            "coef0": coef0,
            "gamma": gamma,
            "degree": degree,
        },
    )

    display(widgets.VBox([n_components, standarized, kernel, coef0, gamma, degree]), w)
