import ipywidgets as widgets
import pandas as pd
import numpy as np
from functools import partial
from MLutilities.EDA import kolmogorov_test, correlation_coef, kruskal_test
from IPython.display import display
import plotly.express as px


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
        partial(kolmogorov_test, dataset=dataset, plot_histogram=True),
        {
            "variable": variable,
            "transformation": transformation,
            "color": color,
            "bins": bins,
        },
    )

    display(widgets.VBox([widgets.HBox([variable, transformation, color]), bins]), w)


def correlation_coef_widget(dataset: pd.DataFrame):

    num_vars = dataset.select_dtypes([np.number]).columns
    cat_vars = dataset.select_dtypes([object]).columns.tolist()

    """
    This function computes the correlation between two numerical variables.

    H0: variables are not correlated
    H1: varaibles are correlated

    Arguments:
        dataset: pandas dataframe or dict with de format {'col1':np.array, 'col2':np.array}
    
    """

    variable1 = widgets.Dropdown(
        options=num_vars,
        description="Variable 1:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    variable2 = widgets.Dropdown(
        options=num_vars,
        description="Variable 2:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    kind = widgets.Dropdown(
        options=["pearson", "spearman", "kendall"],
        description="Method:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )

    w = widgets.interactive_output(
        partial(correlation_coef, dataset=dataset, scatter_plot=True),
        {
            "input_variable": variable1,
            "target_variable": variable2,
            "kind": kind,
        },
    )

    display(widgets.HBox([variable1, variable2, kind]), w)



def countplot_widget(dataset: pd.DataFrame):

    """
    Show the counts of observations in each categorical bin using bars. A count plot can be
    thought of as a histogram across a categorical, instead of quantitative variable.
    This function will infer data types, so it is highly recomended to set categorical variables
    as string or pd.Categorical
    Arguments:
    ---------
        dataset: pandas dataframe or dict with de format {'col1':np.array, 'col2':np.array}
    """

    num_vars = dataset.select_dtypes([np.number]).columns
    cat_vars = dataset.select_dtypes([object, pd.Categorical]).columns.tolist()

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

        fig = px.histogram(data_frame=dataset, **kwargs)
        fig.update_layout(width=1500, height=500)
        fig.show()

    w = widgets.interactive_output(
        partial(hist, dataset=dataset), 
        {
            "x": variable,
            "color": color
         }
        )
    display(widgets.HBox([variable, color]), w)


def kruskal_test_widget(dataset: pd.DataFrame):
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
    """

    num_vars = dataset.select_dtypes([np.number]).columns
    cat_vars = dataset.select_dtypes([object]).columns.tolist()

    num_variable = widgets.Dropdown(
        options=num_vars,
        description="Numerical Variable:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    cat_variable = widgets.Dropdown(
        options=cat_vars,
        description="Categorical Variable:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )

    w = widgets.interactive_output(
        partial(kruskal_test, dataset, plot_boxplot=True),
        {"target_variable": cat_variable, "input_variable": num_variable},
    )

    display(widgets.HBox([num_variable, cat_variable]), w)


def barplot_widget(dataset: pd.DataFrame):

    """
    A bar plot represents an estimate of central tendency for a numeric variable with the height
    of each rectangle and provides some indication of the uncertainty around that estimate using
    error bars. Bar plots include 0 in the quantitative axis range, and they are a good choice when
    0 is a meaningful value for the quantitative variable, and you want to make comparisons against it.

    Arguments:
    ----------
        dataset: pandas dataframe or dict with de format {'col1':np.array, 'col2':np.array}

    """
    num_vars = dataset.select_dtypes([np.number]).columns
    cat_vars = dataset.select_dtypes([object]).columns.tolist()

    num_variable = widgets.Dropdown(
        options=num_vars,
        description="Numerical Variable:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    cat_variable = widgets.Dropdown(
        options=cat_vars,
        description="Categorical Variable:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    func = widgets.Dropdown(
        options=["mean", "median", "sum", "min", "max", "std"],
        description="Agg Gunction:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )

    def barplot(dataset, cat_var=None, num_var=None, func="mean"):

        df = dataset.groupby(cat_var, as_index=False).agg({f"{num_var}": [func]})
        df.columns = ["_".join(col) for col in df.columns.values]

        fig = px.bar(df, x=cat_var + "_", y=f"{num_var}_{func}")
        fig.show()

    w = widgets.interactive_output(
        partial(barplot, dataset=dataset),
        {
            "cat_var": cat_variable,
            "num_var": num_variable,
            "func": func,
        },
    )

    display(widgets.HBox([cat_variable, num_variable, func]), w)
