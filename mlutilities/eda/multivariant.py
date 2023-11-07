import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from IPython.display import display
from typing import Union, Dict, Optional


def _cramerv_relationship_strength(degrees_of_freedom: int, cramerv: float):
    """
    returns the strength of the relationship of two categorical variables
    source: https://www.statology.org/interpret-cramers-v/

    Arguments:
    ----------
        degrees_of_freedom: degrees of freedom obtained from a contingency
            table as: min(number of rows - 1, number of columns - 1)
        cramerv: Cramer's V coefficient
    """
    values = {
        "1": [0.10, 0.50],
        "2": [0.07, 0.35],
        "3": [0.06, 0.29],
        "4": [0.05, 0.25],
        "5": [0.04, 0.22],
    }

    if np.round(cramerv, 2) <= values[str(degrees_of_freedom)][0]:
        return "small"
    elif np.round(cramerv, 2) >= values[str(degrees_of_freedom)][-1]:
        return "high"
    else:
        return "medium"


# Creamers V Correlation
def cramersv(
    dataset,
    target_feature: str,
    input_feature: str,
    show_crosstab: bool = False,
    plot_histogram: bool = False,
    histnorm: str = "percent",
    return_test: bool = True,
    print_test: bool = False,
    plotly_renderer: str = "notebook",
):
    """
    This function computes Cramer's V correlation coefficient which is a measure of association between two nominal variables.

    H0: there is not a relationship between the variables.
    H1: there is a relationship between the variables..

    If p_value < 0.5 you can reject the null hypothesis

    Arguments:
    ----------
        dataset: pandas dataframe or dict with the format {'col1':np.array, 'col2':np.array}
        target_variable: string
            Name of the target variable
        input_variable: string
            Name of the input variable
        show_crosstab: bool:
            if True prints the crosstab used to compute Cramer's V
        plot_histogram: bool
            If True plot the histogram of input_variable
        histnorm: string (default='percentage')
            It can be either 'percent' or 'count'. If 'percent'
            show the percentage of each category, if 'count' show
            the frequency of each category.
        print_test: bool (default=False)
            If True prints the test statistic and p-value
        return_test: bool (default=False)
            If True returns the test statistic and p-value
        plotly_renderer: renderer to use when plotting plotly figures. Options:
            - notebook: render plotly figures in a jupyter notebook
            - colab: render plotly figures in a google colab notebook
    """

    if type(dataset) == dict:
        dataset = pd.DataFrame(dataset)

    dataset = dataset.dropna(subset=[input_feature, target_feature])

    obs = pd.crosstab(dataset[input_feature], dataset[target_feature], margins=False)
    chi2, p, dof, ex = stats.chi2_contingency(obs, correction=False)

    if show_crosstab:
        print("----------------------- Contingency Table -------------------------")
        display(pd.crosstab(dataset[input_feature], dataset[target_feature], margins=True).style.background_gradient(cmap="Blues"))
        print("------------------------------------------------------------------\n")

    dimension = obs.to_numpy().sum()
    cramer = np.sqrt((chi2 / dimension) / (np.min(obs.shape) - 1))

    # interpretation
    n_rows = dataset[target_feature].nunique()
    n_cols = dataset[input_feature].nunique()
    degrees_of_freedom = min(n_rows - 1, n_cols - 1)

    strength = _cramerv_relationship_strength(5 if degrees_of_freedom > 4 else degrees_of_freedom, cramer)

    if print_test:
        print("---------------------------------------------- Cramer's V --------------------------------------------")
        print(f"CramersV: {cramer:.3f}, chi2:{chi2:.3f}, p_value:{p:.5f}\n")
        if p < 0.05:
            print(f"Since {p:.5f} < 0.05 you can reject the null hypothesis, \nThere is a {strength} relationship between the variables.")
        else:
            print(f"Since {p:.5f} > 0.05 you cannot reject the null hypothesis, \nso there is not a relationship between the variables.")
        print("------------------------------------------------------------------------------------------------------\n")

    if plot_histogram:
        fig = px.histogram(
            dataset,
            x=input_feature,
            histnorm=histnorm,
            color=target_feature,
            barmode="group",
            width=1500,
            height=500,
        )
        fig.show(renderer=plotly_renderer)

    if return_test:
        return cramer, p


def cramersv_heatmap(dataset: pd.DataFrame):
    """
    This function plots a heatmap of Cramer's V correlation coefficient. Variables that will be
    used must be categorical or object dtype

    Arguments:
    ----------
        dataset: pandas dataframe

    Example:
    --------
        cramersv_heatmap(dataset)
    """

    cat_vars = dataset.select_dtypes([object, "category"]).columns.tolist()

    # define empty corr dataframe
    corr = pd.DataFrame(columns=cat_vars, index=cat_vars)

    # compute cramersV
    for i in cat_vars:
        for j in cat_vars:
            corr.loc[i, j] = cramersv(dataset, i, j, print_test=False)[0]

    # plot correlaton matrix
    plt.figure(figsize=(15, 8))
    sns.heatmap(corr.astype("float"), annot=True, cbar=False, cmap="Blues")
