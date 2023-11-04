import ipywidgets as widgets
from mlutilities.stats.distributions import example_histogram


def widget_example_histogram_matplotlib():
    """
    Interactively plots the boxplot and the histogram for a normally distributed variable
    and a skewed distributed variabl.
    """

    mean = widgets.FloatSlider(
        description="mean",
        min=-10,
        max=10,
        value=0.0,
        continuous_update=False,
        layout=widgets.Layout(width="20%", height="100px"),
        style={"description_width": "initial"},
    )
    std = widgets.FloatSlider(
        description="std",
        min=0.0,
        max=10,
        value=1.0,
        continuous_update=False,
        layout=widgets.Layout(width="20%", height="100px"),
        style={"description_width": "initial"},
    )
    alpha = widgets.FloatSlider(
        description="alpha",
        min=-30,
        max=30,
        value=0.0,
        continuous_update=False,
        layout=widgets.Layout(width="20%", height="100px"),
        style={"description_width": "initial"},
    )

    w = widgets.interactive_output(example_histogram, {"mean": mean, "std": std, "alpha": alpha})
    display(widgets.HBox([mean, std, alpha]), w)
