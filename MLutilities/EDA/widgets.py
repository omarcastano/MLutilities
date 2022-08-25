import ipywidgets as widgets
import pandas as pd
import numpy as np
from functools import partial
from MLutilities.EDA import kolmogorov_test
from IPython.display import display


def kolmogorov_test_widget(dataset: pd.DataFrame):

    num_vars = dataset.select_dtypes([np.number]).columns
    cat_vars = dataset.select_dtypes([object]).columns.tolist()

    """
    This function computes Kolmogorov test to check if the variable
    is normaly distributed

    H0: The variable follows a normal distribution
    H1: The variable do not follow a normal distribution

    if p_value < 0.05 you can reject the null hypohesis

    Arguments:
        dataset: pandas dataframe or dict with de format {'col1':np.array, 'col2':np.array}
    """

    variable = widgets.Dropdown(
        options=num_vars,
        description="Variable:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    apply_log_transform = widgets.Dropdown(
        options=[False, True],
        description="Apply Log Transform:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    color = widgets.Dropdown(
        options=[None] + cat_vars,
        description="Color:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    bins = widgets.IntSlider(
        description="Bins",
        min=5,
        max=50,
        value=30,
        continuous_update=False,
        layout=widgets.Layout(width="20%", height="100px"),
        style={"description_width": "initial"},
    )

    w = widgets.interactive_output(
        partial(kolmogorov_test, dataset=dataset, plot_histogram=True),
        {
            "variable": variable,
            "apply_log_transform": apply_log_transform,
            "color": color,
            "bins": bins,
        },
    )

    display(
        widgets.VBox([widgets.HBox([variable, apply_log_transform, color]), bins]), w
    )
