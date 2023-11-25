import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from typing import List
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_kde(
    data,
):
    """
    Perform kernel density estimation (KDE) on a given dataset.

    This function calculates the kernel density estimation of a dataset using a Gaussian kernel.
    It returns the x-axis and y-axis values representing the KDE curve.

    Arguments:
    -----------
    data : array-like
        One-dimensional array-like object containing the dataset on which to perform KDE.

    Returns:
    --------
        x : ndarray
            One-dimensional array of x-axis values for the KDE curve.
        y : ndarray
            One-dimensional array of y-axis values representing the estimated density values.

    Example:
    --------
    data = [1.2, 1.5, 2.1, 1.8, 3.2]
    x, y = get_kde(data)
    # x contains the x-axis values for plotting the KDE curve
    # y contains the corresponding y-axis values representing the estimated density
    """
    # Create the kernel density estimation
    kde = gaussian_kde(data)

    # Generate the x-axis values for the plot
    x = np.linspace(data.min(), data.max(), 200)

    # Calculate the y-axis values by evaluating the KDE at each x value
    y = kde(x)

    return x, y


def scaler(dataset: pd.DataFrame = None, variables: List[str] = None, kind: str = "standard_scaler", plotly_renderer="notebook"):
    """
    Helper function to visualize the effect of scaling and normalization over continuous variables

    Arguments:
    ----------
        dataset:   pandas dataframe or dict with the format {'col1':np.array, 'col2':np.array}
        variables: list with the name of the features to scale
        kind:      name of the transformation to perform. Options ["standard_scaler", "minmax_scaler"]
        plotly_renderer: renderer to use when plotting plotly figures. Options:
            - notebook: render plotly figures in a jupyter notebook
            - colab: render plotly figures in a google colab notebook
    """
    # drop nan values
    dataset = dataset.dropna()

    scale = {
        "standard_scaler": StandardScaler().fit_transform,
        "minmax_scaler": MinMaxScaler().fit_transform,
    }

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Original", "Transformed"))

    for var in variables:
        original_data = dataset[var]
        x, y = get_kde(original_data)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=var), row=1, col=1)

        scaled_data = pd.Series(scale[kind](dataset[[var]]).reshape(-1))
        x, y = get_kde(scaled_data)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"scaled {var}"), row=1, col=2)

    fig.update_layout(
        xaxis=dict(title="Value"),
        yaxis=dict(title="Count"),
        width=1000,
        height=500,
        # legend=dict(orientation="h", y=-0.25),
    )

    fig.show(renderer=plotly_renderer)
