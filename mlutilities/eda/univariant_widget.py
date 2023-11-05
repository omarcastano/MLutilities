"""
This module contains the widget for the univariant tests
"""

from functools import partial
import pandas as pd
import numpy as np
import ipywidgets as widgets
from functools import partial
from IPython.display import display
from mlutilities.eda import kolmogorov_test, shapiro_test
import plotly.express as px
from typing import Union, Dict


def kolmogorov_test_widget(dataset: pd.DataFrame, plotly_renderer: str = "notebook"):
    """
    This function computes Kolmogorov test to check if the variable
    is normaly distributed

    H0: The variable follows a normal distribution
    H1: The variable do not follow a normal distribution

    if p_value < 0.05 you can reject the null hypohesis

    Arguments:
    ---------
        dataset: pandas dataframe or dict with de format {'col1':np.array, 'col2':np.array}
        plotly_renderer: renderer to use when plotting plotly figures. Options:
            - notebook: render plotly figures in a jupyter notebook
            - colab: render plotly figures in a google colab notebook
    """

    num_vars = dataset.select_dtypes([np.number]).columns
    cat_vars = dataset.select_dtypes([object, "category"]).columns.tolist()

    variable = widgets.Dropdown(
        options=num_vars,
        description="Variable:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    transformation = widgets.Dropdown(
        options=["None", "yeo_johnson", "log"],
        description="Transformation:",
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
        partial(kolmogorov_test, dataset=dataset, plot_histogram=True, plotly_renderer=plotly_renderer),
        {
            "variable": variable,
            "transformation": transformation,
            "color": color,
            "bins": bins,
        },
    )

    display(widgets.VBox([widgets.HBox([variable, transformation, color]), bins]), w)


def shapiro_test_widget(dataset: pd.DataFrame):
    """
    This function computes Shapiro test to check if the variable
    is normaly distributed

    H0: The variable follows a normal distribution
    H1: The variable do not follow a normal distribution

    if p_value < 0.05 you can reject the null hypothesis

    Arguments:
        dataset: pandas dataframe or dict with de format {'col1':np.array, 'col2':np.array}
    """

    num_vars = dataset.select_dtypes([np.number]).columns
    cat_vars = dataset.select_dtypes([object, "category"]).columns.tolist()

    variable = widgets.Dropdown(
        options=num_vars,
        description="Variable:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    transformation = widgets.Dropdown(
        options=["None", "yeo_johnson", "log"],
        description="Transformation:",
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
        partial(shapiro_test, dataset=dataset, plot_histogram=True),
        {
            "variable": variable,
            "transformation": transformation,
            "color": color,
            "bins": bins,
        },
    )

    display(widgets.VBox([widgets.HBox([variable, transformation, color]), bins]), w)


def countplot_widget(dataset: pd.DataFrame, plotly_renderer: str = "notebook"):
    """
    Show the counts of observations in each categorical bin using bars. A count plot can be
    thought of as a histogram across a categorical, instead of quantitative variable.
    This function will infer data types, so it is highly recommended to set categorical variables
    as string or pd.Categorical

    Arguments:
    ---------
        dataset: pandas dataframe or dict with de format {'col1':np.array, 'col2':np.array}
        plotly_renderer: renderer to use when plotting plotly figures. Options:
            - notebook: render plotly figures in a jupyter notebook
            - colab: render plotly figures in a google colab notebook
    """

    cat_vars = dataset.select_dtypes([object, "category"]).columns.tolist()

    variable = widgets.Dropdown(
        options=cat_vars,
        description="Variable:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    color = widgets.Dropdown(
        options=[None] + cat_vars,
        description="Color:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )

    def hist(dataset, **kwargs):
        fig = px.histogram(data_frame=dataset, barmode="group", histnorm="percent", **kwargs)
        fig.update_layout(width=1500, height=500)
        fig.show(renderer=plotly_renderer)

    w = widgets.interactive_output(
        partial(
            hist,
            dataset=dataset,
        ),
        {"x": variable, "color": color},
    )
    display(widgets.HBox([variable, color]), w)


def kde_widget(dataset: Union[pd.DataFrame, Dict[str, np.ndarray]]):
    """
    Generate a interactive widget for a kernel density estimate (KDE) plot for a given variable in the dataset. Optionally applies a
    transformation to the variable before generating the plot.

    Args:
        dataset (pd.DataFrame or dict with format {'col1': np.array, 'col2': np.array}): The input dataset to use
            for generating the KDE plot.
    """
    num_vars = dataset.select_dtypes([np.number]).columns
    cat_vars = dataset.select_dtypes([object, "category"]).columns.tolist()

    variable = widgets.Dropdown(
        options=num_vars,
        description="Variable:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    transformation = widgets.Dropdown(
        options=["None", "yeo_johnson", "log"],
        description="Transformation:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    color = widgets.Dropdown(
        options=[None] + cat_vars,
        description="Color:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    plot_boxplot = widgets.Checkbox(
        value=False,
        description="boxplot",
        layout=widgets.Layout(width="5%", height="5px"),
        style={"description_width": "initial"},
    )
    w = widgets.interactive_output(
        partial(kde_plot, dataset=dataset),
        {
            "variable": variable,
            "transformation": transformation,
            "color": color,
            "plot_boxplot": plot_boxplot,
        },
    )
    display(widgets.HBox([variable, color, transformation, plot_boxplot]), w)
