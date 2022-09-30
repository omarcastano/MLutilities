import ipywidgets as widgets
import numpy.typing as npt
from functools import partial
from IPython.display import display
from MLutilities.utils import plot_log_reg
from MLutilities.Classification import metrics


def logistic_regression_widget():
    threshold = widgets.FloatSlider(
        description="Threshold",
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


def threshold_metric_widget(
    y_true: npt.ArrayLike, y_predict_proba: npt.ArrayLike
) -> None:
    """
    Widget that plot the value of a given metric for several probability thresholds. This function only work for a binary classification problem.

    Arguments:
          y_true: (1D)
              true labels
          y_score: (2D)
              predicted scores for positive and negative class
    """
    threshold = widgets.FloatSlider(
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
            metrics.threshold_metric_evaluation, y_true=y_true, y_score=y_predict_proba
        ),
        {"metric": metric, "threshold": threshold},
    )

    display(widgets.VBox([metric, threshold]), w)


def precision_recall_tradeoff_widget(
    y_true: npt.ArrayLike, y_predict_proba: npt.ArrayLike
) -> None:
    """
    This function allows you to evaluate the precision-recall tradeoff

      Arguments:
          y_true: (1D)
              true labels
          y_score: (2D)
              predicted scores such as probability
    """
    threshold = widgets.FloatSlider(
        min=0.0,
        max=1,
        value=0.5,
        step=0.01,
        continuous_update=True,
        layout=widgets.Layout(width="20%", height="50px"),
    )
    w = widgets.interactive_output(
        partial(
            metrics.precision_recall_tradeoff, y_true=y_true, y_score=y_predict_proba
        ),
        {"threshold": threshold},
    )
    display(widgets.VBox([threshold]), w)
