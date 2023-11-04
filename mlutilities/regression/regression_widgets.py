import logging
import ipywidgets as widgets
from functools import partial
from . import PolynomialRegression
from IPython.display import display
from mlutilities.utils import plot_polyreg
from sklearn.model_selection import train_test_split
from mlutilities.regression.plots import compare_learning_curves
from mlutilities.regression.utils import generate_nonlinear_data
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


def regularization_widget():
    logging.warning(
        """
        ------------------------------------------------------------------------------------
        This function (MLutilities.Regression.regression_widgets.regularization_widget) is deprecated. 
        Please use MLutilities.regression.regression_widget.polynomial_regression_widget  instead.          
        ------------------------------------------------------------------------------------
        """
    )

    """
    this function helps to visualize the effect of regularization in a regression model
    """
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


def polynomial_regression_widget():
    """
    Interactive polynomial regression
    """

    N = widgets.IntSlider(
        description="Number of points",
        min=50,
        max=500,
        value=50,
        continuous_update=False,
        layout=widgets.Layout(width="25%", height="50px"),
        style={"description_width": "initial"},
    )

    degree = widgets.IntSlider(
        description="Polynomial degree",
        min=1,
        max=30,
        value=2,
        continuous_update=False,
        layout=widgets.Layout(width="25%", height="50px"),
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
        layout=widgets.Layout(width="25%", height="50px"),
        style={"description_width": "initial"},
    )

    l1_ratio = widgets.FloatSlider(
        description="l1_ratio",
        min=0.01,
        max=1,
        value=0.5,
        continuous_update=False,
        layout=widgets.Layout(width="25%", height="50px"),
        style={"description_width": "initial"},
    )

    def poly_reg(degree, N, estimator_name, alpha, l1_ratio):
        # model training
        estimators = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=alpha),
            "Lasso": Lasso(alpha=alpha),
            "ElasticNet": ElasticNet(alpha=alpha, l1_ratio=l1_ratio),
        }

        # generate data
        X, y = generate_nonlinear_data(N=N)

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = PolynomialRegression(degree=degree, estimator=estimators[estimator_name])
        model.fit(X_train, y_train)
        model.plot_fitted_model(X_train, y_train, X_test, y_test)

    w = widgets.interactive_output(
        poly_reg,
        {
            "degree": degree,
            "N": N,
            "estimator_name": estimator,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
        },
    )

    display(
        widgets.VBox(
            [
                widgets.HBox([estimator]),
                widgets.HBox([N, degree]),
                widgets.HBox([alpha, l1_ratio]),
            ]
        ),
        w,
    )


def compare_learning_curves_widget():
    """
    Helper widget that compares the learning curves of two polynomial regression models:
    one using a linear estimator and the other using a regularized linear estimator.
    """

    N = widgets.IntSlider(
        description="Number of points",
        min=50,
        max=500,
        value=50,
        continuous_update=False,
        layout=widgets.Layout(width="25%", height="50px"),
        style={"description_width": "initial"},
    )

    degree = widgets.IntSlider(
        description="Polynomial degree",
        min=1,
        max=20,
        value=13,
        continuous_update=False,
        layout=widgets.Layout(width="25%", height="50px"),
        style={"description_width": "initial"},
    )

    estimator = widgets.Dropdown(
        options=["Ridge", "Lasso", "ElasticNet"],
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
        layout=widgets.Layout(width="25%", height="50px"),
        style={"description_width": "initial"},
    )

    l1_ratio = widgets.FloatSlider(
        description="l1_ratio",
        min=0.01,
        max=1,
        value=0.5,
        continuous_update=False,
        layout=widgets.Layout(width="25%", height="50px"),
        style={"description_width": "initial"},
    )

    w = widgets.interactive_output(
        compare_learning_curves,
        {
            "degree": degree,
            "N": N,
            "estimator": estimator,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
        },
    )

    display(
        widgets.VBox(
            [
                widgets.HBox([estimator]),
                widgets.HBox([N, degree]),
                widgets.HBox([alpha, l1_ratio]),
            ]
        ),
        w,
    )
