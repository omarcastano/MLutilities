# imporotamos librerias
import seaborn as sns
import pandas as pd
import plotly.express as px

sns.set()


def outlier_detection(
    dataset: pd.DataFrame,
    variable: str,
    factor: float = 1.5,
    replace_outliers: bool = False,
    plot_boxplot: bool = False,
):

    """
    Dedect outliers based on Inter Quantile Range
    Arguments:
    ---------
        dataset: DataFrame
        varaible: varaible to detect outliers
        factor: factor to detect outliers usgin the expresions Q3 + factor*IQR
                and Q3 - factor*IQR (Default factor=1.5)
        replace_outlier: if True replace outliers with the upper and lower fence
        plot_boxplot: if Ture plots a boxplot
    """

    q1 = dataset[variable].quantile(q=0.25)
    q3 = dataset[variable].quantile(q=0.75)
    IQR = q3 - q1
    upper = q3 + factor * IQR
    lower = q1 - factor * IQR

    idx = dataset.query(f"({variable} > {upper}) or ({variable} < {lower})")

    if replace_outliers:

        dataset.loc[dataset[variable] > upper, variable] = upper
        dataset.loc[dataset[variable] < lower, variable] = upper

    if plot_boxplot:

        fig = px.box(dataset, x=variable)
        fig.show()
