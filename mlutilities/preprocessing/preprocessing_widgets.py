import pandas as pd
import numpy as np
from IPython.display import display
from ipywidgets import widgets
from functools import partial
from mlutilities.preprocessing.utils import scaler


def scaler_widget(dataset: pd.DataFrame, plotly_renderer: str = "notebook"):
    """
    Helper function to visualize the efect of scaling and normalization over continuos variables

    Arguments:
        dataset: pandas dataframe or dict with the format {'col1':np.array, 'col2':np.array}
        plotly_renderer: renderer to use when plotting plotly figures. Options:
            - notebook: render plotly figures in a jupyter notebook
            - colab: render plotly figures in a google colab notebook
    """

    num_vars = dataset.select_dtypes([np.number]).columns

    num_variable = widgets.SelectMultiple(
        options=num_vars,
        description="Numerical Variable:",
        value=[num_vars[0]],
        disabled=False,
        layout=widgets.Layout(width="20%", height="100px"),
        style={"description_width": "initial"},
    )

    kind = widgets.Dropdown(
        options=["standard_scaler", "minmax_scaler"],
        description="kind:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )

    w = widgets.interactive_output(
        partial(scaler, dataset, plotly_renderer=plotly_renderer),
        {"kind": kind, "variables": num_variable},
    )

    display(widgets.HBox([num_variable, kind]), w)
