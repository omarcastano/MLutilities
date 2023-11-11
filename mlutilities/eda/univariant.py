"""
This module contains functions to perform univariant tests
"""

import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
from typing import Union, Optional, Dict


def kolmogorov_test(
    dataset,
    variable: str,
    transformation: str = None,
    plot_histogram: bool = False,
    bins: int = 30,
    color: str = None,
    print_test: bool = True,
    return_test: bool = False,
    plotly_renderer: str = "notebook",
):
    """
    This function computes Kolmogorov test to check if the variable
    is normaly distributed

    H0: The variable follows a normal distribution
    H1: The variable do not follow a normal distribution

    if p_value < 0.05 you can reject the null hypohesis

    Arguments:
    ----------
        dataset: pandas dataframe or dict with de format {'col1':np.array, 'col2':np.array}
        variable: variable to performe the Kolmogorov test
        transformation: kind of transformation to apply. Options:
             - yeo_johnson: appy yeo johnson transformation to the input variable
             - log: apply logarithm transformation to the input variable
        plot_histogram:If True plot a histogram of the variable
        bins: Number of bins to use when plotting the histogram
        color: Name of column in dataset. Values from this column are used to assign color to marks.
        plotly_renderer: renderer to use when plotting plotly figures. Options:
            - notebook: render plotly figures in a jupyter notebook
            - colab: render plotly figures in a google colab notebook
        return_test: If True return the test statistic and p_value

    Returns:
    --------
        ktest: tuple with the test statistic and p_value
        conclusion: string with the conclusion of the test. Options:
            - Not normal distribution
            - Normal distribution
    """

    if type(dataset) == dict:
        dataset = pd.DataFrame(dataset)

    dataset = dataset.dropna(subset=[variable]).copy()

    if transformation == "yeo_johnson":
        x = stats.yeojohnson(dataset[variable].to_numpy())[0]
    elif transformation == "log":
        x = np.log1p(dataset[variable].to_numpy())
    else:
        x = dataset[variable].to_numpy()

    x_scale = (x - x.mean()) / x.std()

    ktest = stats.kstest(x_scale, "norm")

    if print_test:
        print(f"------------------------- Kolmogorov test fot the variable {variable} --------------------")
        print(f"statistic={ktest[0]:.3f}, p_value={ktest[1]:.3f}\n")
        if ktest[1] < 0.05:
            print(
                f"Since {ktest[1]:.3f} < 0.05 you can reject the null hypothesis, so the variable {variable} \ndo not follow a normal distribution"  # noqa: E501
            )
            conclusion = "Not normal distribution"
        else:
            print(
                f"Since {ktest[1]:.3f} > 0.05 you cannot reject the null hypothesis, so the variable {variable} \nfollows a normal distribution"
            )
            conclusion = "Normal distribution"
        print("-------------------------------------------------------------------------------------------\n")

    else:
        if ktest[1] < 0.05:
            conclusion = "Not normal distribution"
        else:
            conclusion = "Normal distribution"

    if plot_histogram:
        fig = px.histogram(dataset, x=x, nbins=bins, marginal="box", color=color, barmode="overlay")
        fig.update_traces(marker_line_width=1, marker_line_color="white", opacity=0.8)
        fig.update_layout(xaxis_title=variable, width=1500, height=500)
        fig.show(renderer=plotly_renderer)

    if return_test:
        return (
            ktest[0],
            ktest[1],
            conclusion,
        )


def shapiro_test(
    dataset,
    variable: str,
    transformation: str = None,
    plot_histogram: bool = False,
    bins: int = 30,
    color: str = None,
    plotly_renderer: str = "notebook",
):
    """
    This function computes Shapiro test to check if the variable
    is normaly distributed

    H0: The variable follows a normal distribution
    H1: The variable do not follow a normal distribution

    if p_value < 0.05 you can reject the null hypothesis

    Arguments:
    ----------
        dataset: pandas dataframe or dict with de format {'col1':np.array, 'col2':np.array}
        variable: variable to perform the Shapiro test
        transformation: kind of transformation to apply. Options:
             - yeo_johnson: apply yeo johnson transformation to the input variable
             - log: apply logarithm transformation to the input variable
        plot_histogram:If True plot a histogram of the variable
        bins: Number of bins to use when plotting the histogram
        color: Name of column in dataset. Values from this column are used to assign color to marks.
        plotly_renderer: renderer to use when plotting plotly figures. Options:
            - notebook: render plotly figures in a jupyter notebook
            - colab: render plotly figures in a google colab notebook
    """

    if type(dataset) == dict:
        dataset = pd.DataFrame(dataset)

    dataset = dataset.dropna(subset=[variable]).copy()

    if transformation == "yeo_johnson":
        x = stats.yeojohnson(dataset[variable].to_numpy())[0]
    elif transformation == "log":
        x = np.log1p(dataset[variable].to_numpy())
    else:
        x = dataset[variable].to_numpy()

    x_scale = (x - x.mean()) / x.std()

    ktest = stats.shapiro(x_scale)
    print(f"------------------------- Shapiro test fot the variable {variable} --------------------")
    print(f"statistic={ktest[0]:.3f}, p_value={ktest[1]:.3f}\n")
    if ktest[1] < 0.05:
        print(
            f"Since {ktest[1]:.3f} < 0.05 you can reject the null hypothesis, so the variable {variable} \ndo not follow a normal distribution"  # noqa: E501
        )
    else:
        print(
            f"Since {ktest[1]:.3f} > 0.05 you cannot reject the null hypothesis, so the variable {variable} \nfollows a normal distribution"
        )
    print("-------------------------------------------------------------------------------------------\n")
    if plot_histogram:
        fig = px.histogram(dataset, x=x, nbins=bins, marginal="box", color=color, barmode="overlay")
        fig.update_traces(marker_line_width=1, marker_line_color="white", opacity=0.8)
        fig.update_layout(xaxis_title=variable, width=1500, height=500)
        fig.show(renderer=plotly_renderer)


def kde_plot(
    dataset: Union[pd.DataFrame, Dict[str, np.ndarray]],
    variable: str,
    transformation: Optional[str] = None,
    color: Optional[str] = None,
    plot_boxplot: bool = False,
):
    """
    Generate a kernel density estimate (KDE) plot for a given variable in the dataset. Optionally applies a
    transformation to the variable before generating the plot.
    Parameters:
    -----------
        dataset: (pd.DataFrame or dict with format {'col1': np.array, 'col2': np.array}): The input dataset to use
            for generating the KDE plot.
        variable: (str): The name of the variable to use in the KDE plot.
        transformation: (str, optional): The kind of transformation to apply to the input variable. Default is None.
            Valid options are:
                - "yeo_johnson": apply Yeo-Johnson transformation to the input variable.
                - "log": apply logarithmic transformation to the input variable.
        color: (str, optional): The name of the column in the dataset to use for assigning colors to the marks in
            the plot. Default is None.
        plot_boxplot: (bool, optional): Whether to add a boxplot in the margin of the KDE plot. Default is False.
    """

    if type(dataset) == dict:
        dataset = pd.DataFrame(dataset)
    dataset = dataset.dropna(subset=[variable]).copy()

    if transformation == "yeo_johnson":
        x = stats.yeojohnson(dataset[variable].to_numpy())[0]
    elif transformation == "log":
        x = np.log1p(dataset[variable].to_numpy())
    else:
        x = dataset[variable].to_numpy()
    x_scale = (x - x.mean()) / x.std()

    if plot_boxplot:
        mosaic = """
    aaaaa
    AAAAA
    AAAAA
    AAAAA
    AAAAA
    """
        fig, ax = plt.subplot_mosaic(mosaic, figsize=(20, 10), sharex=True)
        sns.kdeplot(x=x, hue=dataset[color] if color else None, ax=ax["A"])
        sns.boxplot(x=x, y=dataset[color] if color else None, ax=ax["a"])
        ax["A"].set_ylabel("Density", size=15)
        ax["A"].set_xlabel(variable, size=15)
    else:
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.kdeplot(x=x, hue=dataset[color] if color else None, ax=ax)
        ax.set_ylabel("Density", size=15)
        ax.set_xlabel(variable, size=15)
