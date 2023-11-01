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

"""
This module provides some exploratory data analysis tools.
"""


def kolmogorov_test(
    dataset,
    variable: str,
    transformation: str = None,
    plot_histogram: bool = False,
    bins: int = 30,
    color: str = None,
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
    print(f"------------------------- Kolmogorov test fot the variable {variable} --------------------")
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


def biserial_correlation(
    dataset,
    categorical_variable: str,
    numerical_variable: str,
    transformation: str = None,
    box_plot: bool = True,
    test_assumptions: bool = True,
    plotly_renderer: str = "notebook",
):
    """
    A point-biserial correlation is used to measure the correlation between
    a continuous variable and a binary variable.
    Assumption: continuous data within each group created by the binary variable
    are normally distributed with equal variances and possibly different means.

    H0: variables are not correlated
    H1: variables are correlated

    If p_value < 0.05 reject the null hypothesis

    Arguments:
    ----------
        dataset: pandas dataframe or dict with de format {'col1':np.array, 'col2':np.array}
        categorical_variable: string
            Name of the binary categorical variable
        numerical_varaible: string
            Name of the numercial variable
        transformation: kind of transformation to apply. Options:
            - yeo_johnson: appy yeo johnson transformation to the input variable
            - log: apply logarithm transformation to the input variable
        box_plot:bool
            If Ture display a boxplot
        test_assumptions: bool
            If True test the assuptioms for the continuos variable
        plotly_renderer: renderer to use when plotting plotly figures. Options:
            - notebook: render plotly figures in a jupyter notebook
            - colab: render plotly figures in a google colab notebook
    """

    if type(dataset) == dict:
        dataset = pd.DataFrame(dataset)

    if test_assumptions:
        y_unique = dataset[categorical_variable].unique()
        x1 = dataset.loc[dataset[categorical_variable] == y_unique[0], [numerical_variable]]
        x2 = dataset.loc[dataset[categorical_variable] == y_unique[1], [numerical_variable]]

        print(f"------------------------Kolmogorov Test for {categorical_variable}:{y_unique[0]}---------------------------")
        kolmogorov_test(
            x1,
            numerical_variable,
            transformation=transformation,
            plot_histogram=False,
        )
        print(f"------------------------Kolmogorov Test for {categorical_variable}:{y_unique[1]}---------------------------")
        kolmogorov_test(
            x2,
            numerical_variable,
            transformation=transformation,
            plot_histogram=False,
        )

        print("--------------------------------Levene Test-----------------------------------")
        levene_test(dataset, categorical_variable, numerical_variable)

    # Point Biserial correlation Test
    y = LabelEncoder().fit_transform(dataset[categorical_variable])

    if transformation == "yeo_johnson":
        x = stats.yeojohnson(dataset[numerical_variable].to_numpy())[0]
    elif transformation == "log":
        x = np.log1p(dataset[numerical_variable].to_numpy())
    else:
        x = dataset[numerical_variable].to_numpy()

    biserial = stats.pointbiserialr(y, x)
    print("---------------------------Point Biserial Test--------------------------------")
    print(f"statistic={biserial[0]:.3f}, p_value={biserial[1]:.3f}\n")
    if biserial[1] < 0.05:
        print(f"Since {biserial[1]:.3f} < 0.05 you can reject the null hypothesis, \nso variables are correlated")
    else:
        print(f"Since {biserial[1]:.3f} > 0.05 you cannot reject the null hypothesis, \nso variables are not correlated")
    print("------------------------------------------------------------------------------\n")

    if box_plot:
        fig = px.box(dataset, x=categorical_variable, y=numerical_variable)
        fig.show(renderer=plotly_renderer)


def levene_test(dataset, categorical_variable, numerical_variable):
    """

    Levenes test is used to check that variances are equal for all
    samples when your data comes from a non normal distribution. This
    function is created to work only when categorical variable is binary

    H0: variances_1 = variances_2 = variances.
    H2: variances_1 != variances_2.

    if p_values < 0.05 rejecct the null hypothesis

    Arguments:
        dataset: pandas dataframe or dict with de format {'col1':np.array, 'col2':np.array}
        categorical_variable: string
            Name of the categorical variable
        input_varaible: string
            Name of the numerical variable

    """

    if type(dataset) == dict:
        dataset = pd.DataFrame(dataset)

    y_unique = dataset[categorical_variable].unique()

    x1 = dataset.loc[dataset[categorical_variable] == y_unique[0], numerical_variable].to_numpy()
    x2 = dataset.loc[dataset[categorical_variable] == y_unique[1], numerical_variable].to_numpy()

    levene = stats.levene(x1, x2)

    print("------------------------------------------------------------------------------")
    print(f"statistic={levene[0]:.3f}, p_value={levene[1]:.3f}\n")
    if levene[1] < 0.05:
        print(f"Since {levene[1]:.3f} < 0.05 you can reject the null hypothesis, \nso variances_1 != variances_2")
    else:
        print(f"Since {levene[1]:.3f} > 0.05 you cannot reject the null hypothesis, \nso variances_1 = variances_2")
    print("------------------------------------------------------------------------------\n")


# Kruskall-Wallas Test
def kruskal_test(
    dataset,
    target_variable: str,
    input_variable: str,
    plot_boxplot: bool = False,
    show_shapes: bool = False,
    plotly_renderer: str = "notebook",
):
    """
    The Kruskal-Wallis H test is a rank-based nonparametric test
    that can be used to determine if there are statistically significant
    differences between two or more groups of an independent variable on
    a continuous or ordinal dependent variable.

    Assumption:
        - Continuoues variable not need to follow a normal distribution
        - The distributions in each group should have the same shape.

    H0: medians_1 = medians_2 = .... = medians.
    H2: medians_1 != medians_2 != .....

    If p_values < 0.05 rejecct the null hypothesis

    Arguments:
    ----------
        dataset: pandas dataframe or dict with de format {'col1':np.array, 'col2':np.array}
        target_variable: string
            Name of the categorical variable
        input_varaible: string
            Name of the numerical variable
        plot_boxplot: bool
            If True display a boxplot
        show_shapes: bool
            If True print the skewness and kurtosis of the input_variable
        plotly_renderer: renderer to use when plotting plotly figures. Options:
            - notebook: render plotly figures in a jupyter notebook
            - colab: render plotly figures in a google colab notebook
    """

    if type(dataset) == dict:
        dataset = pd.DataFrame(dataset)

    dataset = dataset.dropna(subset=[target_variable, input_variable]).copy()

    y_unique = dataset[target_variable].unique()

    x = [dataset.loc[dataset[target_variable] == unique, input_variable] for unique in y_unique]

    if show_shapes:
        print("--------------------------------Skewness and Kurtosis-------------------------")

        for xi, yi in zip(x, y_unique):
            print(f"Skweness and kurtosis for {target_variable}:{yi}. Skweness={xi.skew():.3f}, Kurtosis={xi.kurtosis():.3f}")

        print("------------------------------------------------------------------------------\n")

    kruskal = stats.kruskal(*x)
    print("------------------------------------------------------------------------------")
    print(f"statistic={kruskal[0]:.3f}, p_value={kruskal[1]:.3f}\n")
    if kruskal[1] < 0.05:
        print(f"Since {kruskal[1]:.3f} < 0.05 you can reject the null hypothesis, \nso we have that medians_1 != medians_2 != ....")
    else:
        print(f"Since {kruskal[1]:.3f} > 0.05 you cannot reject the null hypothesis \nso we have that medians_1 = medians_2 = ....")
    print("------------------------------------------------------------------------------\n")

    if plot_boxplot:
        fig = px.box(
            data_frame=dataset,
            x=input_variable,
            y=target_variable,
            width=1500,
            height=500,
        )
        fig.show(renderer=plotly_renderer)


def _cramerv_relationship_strength(degrees_of_freedom: int, cramerv: float):
    """
    returns the strength of the relationship of two categorical variables
    source: https://www.statology.org/interpret-cramers-v/

    Arguments:
    ----------
    degrees_of_freedom:  degrees of freedom obtained from a contingency
                         table as:
                            min(number of rows - 1, number of columns - 1)
    cramerv:             Cramer's V coefficient
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


def breusch_pagan_test(dataset, target_variable: str, input_variable: str):
    """
    Breusch-Pagan test is a way to check whether heteroscedasticity exists in regression analysis.
    A Breusch-Pagan test follows the below hypotheses:

    H0: Homoscedasticity is present.
    H1: Homoscedasticity is not present (i.e. heteroscedasticity exists)

    Arguments:
        dataset: pandas dataframe or dict with de format {'col1':np.array, 'col2':np.array}
        target_variable: string
            Name of one numercial variable
        input_variable: string
            Name of one numercial variable

    """

    if type(dataset) == dict:
        dataset = pd.DataFrame(dataset)

    # fit regression model
    fit_lr = smf.ols(f"{target_variable} ~ {input_variable}", data=dataset).fit()

    # perform Bresuch-Pagan test
    statistic, p_value, _, _ = sms.het_breuschpagan(fit_lr.resid, fit_lr.model.exog)

    print("------------------------------------ Breusch-Pagan test ----------------------------------")
    print(f"statistic={statistic:.3f}, p_value={p_value:.3f}\n")
    if p_value < 0.05:
        print(f"Since {p_value:.3f} < 0.05 you can reject the null hypothesis, so homoscedasticity is not present")
    else:
        print(f"Since {p_value:.3f} > 0.05 you cannot reject the null hypothesis, so homoscedasticity is present")
    print("-------------------------------------------------------------------------------------------\n")


def correlation_coef(
    dataset,
    target_variable: str,
    input_variable: str,
    kind: str = "pearson",
    kolmogorov: bool = False,
    breusch_pagan: bool = False,
    scatter_plot: bool = False,
    apply_log_transform: bool = False,
    plotly_renderer: str = "notebook",
):
    """
    This function computes the correlation between two numerical variables.

    H0: variables are not correlated
    H1: varaibles are correlated

    Arguments:
        dataset: pandas dataframe or dict with de format {'col1':np.array, 'col2':np.array}
        target_variable: string
            Name of one numercial variable
        input_variable: string
            Name of one numercial variable
        kind: string
            kind of correlation you want to compute, possible potions are
            pearson, spearman or kendall.
        kolmogorov_test: bool
            wheter or not to compute kolmogorov test to check if variables
            are normally distributed. This test is only relevant if kint = 'spearman'
            correleation is  used.
        breusch_pagan:
            wheter or not to compute breusch pagan test to check if homoscedasticity
            is present. This test is only relevant if kind = 'spearman'
            correleation is  used.
        scatter_plot: bool
            If True a scatter plot is display
        apply_log_transform: bool
            If Ture apply a logarithm transformation to input and target variables
        plotly_renderer: renderer to use when plotting plotly figures. Options:
            - notebook: render plotly figures in a jupyter notebook
            - colab: render plotly figures in a google colab notebook
    """

    assert kind in [
        "spearman",
        "kendall",
        "pearson",
    ], "kind must be one of the options in the list ['spearmn', 'kendall', 'pearson']"

    if type(dataset) == dict:
        dataset = pd.DataFrame(dataset).copy()

    if (apply_log_transform) and (target_variable != input_variable):
        dataset = np.log1p(dataset[[target_variable, input_variable]].dropna().copy())
        print(dataset.head())

    if kind == "pearson":
        df = dataset[[target_variable, input_variable]].dropna().copy()
        if kolmogorov:
            kolmogorov_test(df, target_variable)
            kolmogorov_test(df, input_variable)
        if breusch_pagan:
            breusch_pagan_test(df, target_variable, input_variable)
        corr, p_value = stats.pearsonr(
            df.iloc[:, 0],
            df.iloc[:, 1],
        )
    elif kind == "spearman":
        corr, p_value = stats.spearmanr(
            dataset[target_variable],
            dataset[input_variable],
            nan_policy="omit",
        )
    elif kind == "kendall":
        corr, p_value = stats.kendalltau(
            dataset[target_variable],
            dataset[input_variable],
            nan_policy="omit",
        )

    print(f"------------------------------------ {kind} correlation ---------------------------------")
    print(f"statistic={corr:.3f}, p_value={p_value:.3f}\n")
    if p_value < 0.05:
        print(
            f"Since {p_value:.3f} < 0.05 you can reject the null hypothesis, so the variables {target_variable} \nand {input_variable} are correlated"  # noqa: E501
        )
    else:
        print(
            f"Since {p_value:.3f} > 0.05 you cannot reject the null hypothesis, so the variables {target_variable} \nand {input_variable} are not correlated"  # noqa: E501
        )
    print("-------------------------------------------------------------------------------------------\n")

    if scatter_plot:
        fig = px.scatter(
            dataset,
            x=input_variable,
            y=target_variable,
            # marginal_x="histogram",
            # marginal_y="histogram",
            trendline="ols",
            width=1200,
            height=600,
        )
        fig.update_traces(marker_line_width=1, marker_line_color="white", opacity=0.8)
        fig.show(renderer=plotly_renderer)


def contingency_table(dataset, target_variable: str, input_variable: str, table_size: int = 30) -> None:
    """
    This function computes the contingency table of the given varaibles

    Arguments:
        dataset: pandas dataframe or dict with de format {'col1':np.array, 'col2':np.array}
        target_variable: string
            Name of one categorical variable
        input_varaible: string
            Name of one categorical variable
        table_size: integet
            size of the displayed table in pixels
    """

    if type(dataset) == dict:
        dataset = pd.DataFrame(dataset)

    obs = pd.crosstab(dataset[input_variable], dataset[target_variable], margins=True)

    print("----------------------- Contingency Table -------------------------")
    display(obs.style.background_gradient(cmap="Blues").set_table_attributes(f'style="font-size: {table_size}px"'))
    print("------------------------------------------------------------------\n")


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
    Args:
        dataset (pd.DataFrame or dict with format {'col1': np.array, 'col2': np.array}): The input dataset to use
            for generating the KDE plot.
        variable (str): The name of the variable to use in the KDE plot.
        transformation (str, optional): The kind of transformation to apply to the input variable. Default is None.
            Valid options are:
                - "yeo_johnson": apply Yeo-Johnson transformation to the input variable.
                - "log": apply logarithmic transformation to the input variable.
        color (str, optional): The name of the column in the dataset to use for assigning colors to the marks in
            the plot. Default is None.
        plot_boxplot (bool, optional): Whether to add a boxplot in the margin of the KDE plot. Default is False.
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
