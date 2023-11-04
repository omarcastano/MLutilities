import ipywidgets as widgets
from functools import partial
from IPython.display import display
from mlutilities.classification.plots import plot_1d_binary_classification


def binary_1d_widget():
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
    instance_position = widgets.IntSlider(
        description="Instance position",
        min=12,
        max=40,
        value=12,
        step=1,
        continuous_update=False,
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    w = widgets.interactive_output(
        partial(plot_1d_binary_classification),
        {
            "threshold": threshold,
            "regression": regression,
            "instance_position": instance_position,
        },
    )
    display(widgets.VBox([regression, threshold, instance_position]), w)
