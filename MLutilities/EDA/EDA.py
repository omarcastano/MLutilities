import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
import numpy as np
from statsmodels.compat import lzip
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from IPython.display import display

"""
This module provides some exploratory data analysis tools.
"""


def kolmogorov_test(
    dataset,
    variable: str,
    apply_yeo_johnson: bool = False,
    apply_log_transform: bool = False,
    plot_histogram: bool = False,
    bins: int = 30,
    color: str = None,
):

    """

    This function computes Kolmogorov test to check if the variable
    is normaly distributed

    H0: The variable follows a normal distribution
    H1: The variable do not follow a normal distribution

    if p_value < 0.05 you can reject the null hypohesis

    Arguments:
        dataset: pandas dataframe or dict with de format {'col1':np.array, 'col2':np.array}
        variable: string
            variable to performe the Kolmogorov test
        apply_yeo_johnson: bool
            If True appy yeo johnson transformation to the input variable
        apply_log_transform: bool
            If True apply logarithm transformation to the input variable
        plot_histogram: bool
            If True plot a histogram of the variable
        bins: int
            Number of bins to use when plotting the histogram
        color: string
            Name of column in dataset. Values from this column are used to
            assign color to marks.

    """

    if type(dataset) == dict:
        dataset = pd.DataFrame(dataset)

    if apply_yeo_johnson:
        x = stats.yeojohnson(dataset[variable].to_numpy())[0]
    elif apply_log_transform:
        x = np.log1p(dataset[variable].to_numpy())
    else:
        x = dataset[variable].to_numpy()

    ktest = stats.kstest(x, "norm")
    print(
        f"------------------------- Kolmogorov test fot the variable {variable} --------------------"
    )
    print(f"statistic={ktest[0]:.3f}, p_value={ktest[1]:.3f}\n")
    if ktest[1] < 0.05:
        print(
            f"Since {ktest[1]:.3f} < 0.05 you can reject the null hypothesis, so the variable {variable} \ndo not follow a normal distribution"
        )
    else:
        print(
            f"Since {ktest[1]:.3f} > 0.05 you cannot reject the null hypothesis, so the variable {variable} \nfollows a normal distribution"
        )
    print(
        "-------------------------------------------------------------------------------------------\n"
    )
    if plot_histogram:
        fig = px.histogram(
            dataset, x=x, nbins=bins, marginal="box", color=color, barmode="overlay"
        )
        fig.update_traces(marker_line_width=1, marker_line_color="white", opacity=0.8)
        fig.update_layout(xaxis_title=variable)
        fig.show()


def biserial_correlation(
    dataset,
    categorical_variable: str,
    numerical_variable: str,
    apply_yeo_johnson: bool = False,
    apply_log_transform: bool = False,
    box_plot: bool = False,
    test_assumptions: bool = False,
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
    dataset: pandas dataframe or dict with de format {'col1':np.array, 'col2':np.array}
    categorical_variable: string
        Name of the binary categorical variable
    numerical_varaible: string
        Name of the numercial variable
    apply_yeo_johnson: bool
        If True appy yeo johnson transformation to the input variable
    apply_log_transform: bool
        If True apply logarithm transformation to the input variable
    box_plot:bool
        If Ture display a boxplot
    test_assumptions: bool
        If True test the assuptioms for the continuos variable
    """

    assert (
        not apply_log_transform or not apply_yeo_johnson
    ), "apply_log_transform and apply_yeo_johnson cannot be True at the same time"

    if type(dataset) == dict:
        dataset = pd.DataFrame(dataset)

    if test_assumptions:
        y_unique = dataset[categorical_variable].unique()
        x1 = dataset.loc[
            dataset[categorical_variable] == y_unique[0], [numerical_variable]
        ]
        x2 = dataset.loc[
            dataset[categorical_variable] == y_unique[1], [numerical_variable]
        ]

        print(
            f"------------------------Kolmogorov Test for {categorical_variable}:{y_unique[0]}---------------------------"
        )
        kolmogorov_test(
            x1,
            numerical_variable,
            apply_yeo_johnson=apply_yeo_johnson,
            apply_log_transform=apply_log_transform,
            plot_histogram=False,
        )
        print(
            f"------------------------Kolmogorov Test for {categorical_variable}:{y_unique[1]}---------------------------"
        )
        kolmogorov_test(
            x1,
            numerical_variable,
            apply_yeo_johnson=apply_yeo_johnson,
            apply_log_transform=apply_log_transform,
            plot_histogram=False,
        )

        print(
            "--------------------------------Levene Test-----------------------------------"
        )
        levene_test(dataset, categorical_variable, numerical_variable)

    # Point Biserial correlation Test
    y = LabelEncoder().fit_transform(dataset[categorical_variable])

    if apply_yeo_johnson:
        x = stats.yeojohnson(dataset[numerical_variable].to_numpy())[0]
    elif apply_log_transform:
        x = np.log1p(dataset[numerical_variable].to_numpy())
    else:
        x = dataset[numerical_variable].to_numpy()

    biserial = stats.pointbiserialr(y, x)
    print(
        "---------------------------Point Biserial Test--------------------------------"
    )
    print(f"statistic={biserial[0]:.3f}, p_value={biserial[1]:.3f}\n")
    if biserial[1] < 0.05:
        print(
            f"Since {biserial[1]:.3f} < 0.05 you can reject the null hypothesis, \nso variables are correlated"
        )
    else:
        print(
            f"Since {biserial[1]:.3f} > 0.05 you cannot reject the null hypothesis, \nso variables are not correlated"
        )
    print(
        "------------------------------------------------------------------------------\n"
    )

    if box_plot:
        fig = px.box(dataset, x=categorical_variable, y=numerical_variable)
        fig.show()


def levene_test(dataset, categorical_variable, numerical_variable):

    """

    Levene???s test is used to check that variances are equal for all
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

    x1 = dataset.loc[
        dataset[categorical_variable] == y_unique[0], numerical_variable
    ].to_numpy()
    x2 = dataset.loc[
        dataset[categorical_variable] == y_unique[1], numerical_variable
    ].to_numpy()

    levene = stats.levene(x1, x2)

    print(
        "------------------------------------------------------------------------------"
    )
    print(f"statistic={levene[0]:.3f}, p_value={levene[1]:.3f}\n")
    if levene[1] < 0.05:
        print(
            f"Since {levene[1]:.3f} < 0.05 you can reject the null hypothesis, \nso variances_1 != variances_2"
        )
    else:
        print(
            f"Since {levene[1]:.3f} > 0.05 you cannot reject the null hypothesis, \nso variances_1 = variances_2"
        )
    print(
        "------------------------------------------------------------------------------\n"
    )

#Kruskall-Wallas Test
def kruskal_test(dataset, target_variable: str, input_variable: str, plot_histogram:bool=False):

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
        dataset: pandas dataframe or dict with de format {'col1':np.array, 'col2':np.array}
        target_variable: string
            Name of the categorical variable
        input_varaible: string
            Name of the numerical variable

    """

    if type(dataset) == dict:
        dataset = pd.DataFrame(dataset)

    y_unique = dataset[target_variable].unique()
    
    x = [dataset.loc[dataset[target_variable] == unique, input_variable] for unique in y_unique]

    print(
        "--------------------------------Skewness and Kurtosis-------------------------"
    )
    
    for xi, yi in zip(x, y_unique):
        
        print(
        f"Skweness and kurtosis for {target_variable}:{yi}. Skweness={xi.skew():.3f}, Kurtosis={xi.kurtosis():.3f}"
        )
    
    print(
        "------------------------------------------------------------------------------\n"
    )

    kruskal = stats.kruskal(*x)
    print(
        "------------------------------------------------------------------------------"
    )
    print(f"statistic={kruskal[0]:.3f}, p_value={kruskal[1]:.3f}\n")
    if kruskal[1] < 0.05:
        print(
            f"Since {kruskal[1]:.3f} < 0.05 you can reject the null hypothesis, \nso we have that medians_1 != medians_2 != ...."
        )
    else:
        print(
            f"Since {kruskal[1]:.3f} > 0.05 you cannot reject the null hypothesis \nso we have that medians_1 = medians_2 != ...."
        )
    print(
        "------------------------------------------------------------------------------\n"
    )
    
    if plot_histogram:
        
        fig = px.histogram(data_frame=dataset, x=input_variable, color=target_variable, marginal='box', nbins=40)
        fig.update_traces(marker_line_width=1, marker_line_color="white", opacity=0.8)
        fig.show()

# Creamers V Correlation
def cramersv(
    dataset,
    target_feature: str,
    input_feature: str,
    show_crosstab: bool = False,
    plot_histogram: bool = False,
    histnorm: str = "percent",
    color: str = None,
):
    """
    This function computes cramer's V correlation coefficient which is a measure of association between two nominal variables.

    H0: there is not a relationship between the variables.
    H1: there is a relationship between the variables.

    Arguments:
        dataset: pandas dataframe or dict with the format {'col1':np.array, 'col2':np.array}
        target_variable: string
            Name of the target variable
        input_varaible: string
            Name of the input variable
        show_crosstab: bool:
            if True prints the crosstab used to compute Cramer's V
        plot_histogram: bool
            If True plot the histogram of input_variable
        histnorm: string (default='percentage')
            It can be either 'percent' or 'count'. If 'percent'
            show the percengate of each category, if 'count' show
            the frequency of each category.
        color: string
            Name of column in dataset. Values from this column are used to
            assign color to markers.

    If p_value < 0.5 you can reject the null hypothesis
    """

    if type(dataset) == dict:
        dataset = pd.DataFrame(dataset)

    obs = pd.crosstab(dataset[input_feature], dataset[target_feature], margins=True)
    chi2, p, dof, ex = stats.chi2_contingency(obs, correction=False)

    if show_crosstab:
        print("----------------------- Contingency Table -------------------------")
        display(
            pd.crosstab(
                dataset[input_feature], dataset[target_feature], margins=True
            ).style.background_gradient(cmap="Blues")
        )
        print("------------------------------------------------------------------\n")

    dimension = dataset[[input_feature, target_feature]].notnull().prod(axis=1).sum()
    cramer = np.sqrt((chi2 / dimension) / (np.min(obs.shape) - 1))
    print(
        "---------------------------------------------- Cramer's V --------------------------------------------"
    )
    print(f"CramersV: {cramer:.3f}, chi2:{chi2:.3f}, p_value:{p:.5f}\n")
    if p < 0.05:
        print(
            f"Since {p:.5f} < 0.05 you can reject the null hypothesis, \nso there is a relationship between the variables."
        )
    else:
        print(
            f"Since {p:.5f} > 0.05 you cannot reject the null hypothesis, \nso there is not a relationship between the variables."
        )
    print(
        "------------------------------------------------------------------------------------------------------\n"
    )

    if plot_histogram:
        fig = px.histogram(
            dataset, x=input_feature, histnorm=histnorm, color=color, barmode="group"
        )
        fig.show()


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

    print(
        f"------------------------------------ Breusch-Pagan test ----------------------------------"
    )
    print(f"statistic={statistic:.3f}, p_value={p_value:.3f}\n")
    if p_value < 0.05:
        print(
            f"Since {p_value:.3f} < 0.05 you can reject the null hypothesis, so homoscedasticity is not present"
        )
    else:
        print(
            f"Since {p_value:.3f} > 0.05 you cannot reject the null hypothesis, so homoscedasticity is present"
        )
    print(
        "-------------------------------------------------------------------------------------------\n"
    )


def correlation_coef(
    dataset,
    target_variable: str,
    input_variable: str,
    kind: str = "pearson",
    kolmogorov: bool = False,
    breusch_pagan: bool = False,
    scatter_plot: bool = False,
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
    """

    assert kind in [
        "spearman",
        "kendall",
        "pearson",
    ], "kind must be one of the options in the list ['spearmn', 'kendall', 'pearson']"

    if type(dataset) == dict:
        dataset = pd.DataFrame(dataset)

    if kind == "pearson":
        if kolmogorov:
            kolmogorov_test(dataset, target_variable)
            kolmogorov_test(dataset, input_variable)
        if breusch_pagan:
            breusch_pagan_test(dataset, target_variable, input_variable)
        corr, p_value = stats.pearsonr(
            dataset[target_variable], dataset[input_variable]
        )
    elif kind == "spearman":
        corr, p_value = stats.spearmanr(
            dataset[target_variable], dataset[input_variable]
        )
    elif kind == "kendall":
        corr, p_value = stats.kendalltau(
            dataset[target_variable], dataset[input_variable]
        )

    if scatter_plot:
        fig = px.scatter(dataset, x=input_variable, y=target_variable, marginal_x='histogram', marginal_y='histogram', width=1200, height=600)
        fig.update_traces(marker_line_width=1, marker_line_color="white", opacity=0.8)

    print(
        f"------------------------------------ {kind} correlation ---------------------------------"
    )
    print(f"statistic={corr:.3f}, p_value={p_value:.3f}\n")
    if p_value < 0.05:
        print(
            f"Since {p_value:.3f} < 0.05 you can reject the null hypothesis, so the variables {target_variable} \nand {input_variable} are correlated"
        )
    else:
        print(
            f"Since {p_value:.3f} > 0.05 you cannot reject the null hypothesis, so the variables {target_variable} \nand {input_variable} are not correlated"
        )
    print(
        "-------------------------------------------------------------------------------------------\n"
    )
    
    fig.show()


def contingency_table(
    dataset, target_variable: str, input_variable: str, table_size: int = 30
) -> None:

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
    display(
        obs.style.background_gradient(cmap="Blues").set_table_attributes(
            f'style="font-size: {table_size}px"'
        )
    )
    print("------------------------------------------------------------------\n")
