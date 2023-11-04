import ipywidgets as widgets
from functools import partial
from mlutilities.utils import plot_polyreg
from IPython.display import display
import logging


logging.warning(
    """
    ------------------------------------------------------------------------------------
    The module Regression will be remove. Pleas use regression (with lower case) instead.     
    ------------------------------------------------------------------------------------
    """
)


def regularization_widget():
    """
    this function helps to visualize the effect of regularization in a regression model
    """
    logging.warning(
        """
        ------------------------------------------------------------------------------------
        This function (MLutilities.Regression.regression_widgets.regularization_widget) is deprecated. 
        Please use MLutilities.regression.regression_widget.polynomial_regression_widget  instead.          
        ------------------------------------------------------------------------------------
        """
    )
    N = widgets.IntSlider(
        description="Number of points",
        min=50,
        max=500,
        value=50,
        continuous_update=False,
        layout=widgets.Layout(width="20%", height="50px"),
        style={"description_width": "initial"},
    )

    degree = widgets.IntSlider(
        description="Polynomial degree",
        min=1,
        max=15,
        value=2,
        continuous_update=False,
        layout=widgets.Layout(width="20%", height="50px"),
        style={"description_width": "initial"},
    )

    estimator = widgets.Dropdown(
        options=["LinearRegression", "Ridge", "Lasso", "ElasticNet"],
        description="Estimator:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )

    alpha = widgets.FloatSlider(
        description="Alpha",
        min=0.01,
        max=10,
        value=1,
        step=0.01,
        continuous_update=False,
        layout=widgets.Layout(width="20%", height="50px"),
        style={"description_width": "initial"},
    )

    l1_ratio = widgets.FloatSlider(
        description="l1_ratio",
        min=0.1,
        max=1,
        value=0.5,
        continuous_update=False,
        layout=widgets.Layout(width="20%", height="50px"),
        style={"description_width": "initial"},
    )
    helper_viz = widgets.Dropdown(
        options=["learning curve", "weights"],
        value="weights",
        description="Second viz:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )

    w = widgets.interactive_output(
        partial(plot_polyreg),
        {
            "N": N,
            "degree": degree,
            "estimator": estimator,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "helper_viz": helper_viz,
        },
    )

    display(
        widgets.VBox(
            [
                widgets.HBox([estimator, helper_viz]),
                widgets.HBox([N, degree]),
                widgets.HBox([alpha, l1_ratio]),
            ]
        ),
        w,
    )
