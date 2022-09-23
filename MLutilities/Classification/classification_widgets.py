import ipywidgets as widgets
from functools import partial
from IPython.display import display
from MLutilities.utils import plot_log_reg


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
