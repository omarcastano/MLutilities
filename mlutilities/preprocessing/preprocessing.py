# imporotamos librerias
import nltk
import re
import seaborn as sns
import pandas as pd
import plotly.express as px

nltk.download("stopwords")

sns.set()


def outlier_detection(
    dataset: pd.DataFrame,
    variable: str,
    factor: float = 1.5,
    replace_outliers: bool = False,
    plot_boxplot: bool = False,
):
    """
    Detect outliers based on Inter Quantile Range

    Arguments:
    ---------
        dataset: DataFrame
        variable: variable to detect outliers
        factor: factor to detect outliers unsign the expressions Q3 + factor*IQR
                and Q3 - factor*IQR (Default factor=1.5)
        replace_outlier: if True replace outliers with the upper and lower fence
        plot_boxplot: if True plots a boxplot
    """

    q1 = dataset[variable].quantile(q=0.25)
    q3 = dataset[variable].quantile(q=0.75)
    IQR = q3 - q1
    upper = q3 + factor * IQR
    lower = q1 - factor * IQR

    if replace_outliers:
        dataset.loc[dataset[variable] > upper, variable] = upper
        dataset.loc[dataset[variable] < lower, variable] = upper

    if plot_boxplot:
        fig = px.box(dataset, x=variable)
        fig.show()


def text_cleaning(text: str) -> str:
    """
    Basic text preprocessing

    Arguments:
    ----------
    text: str
        input string to be preprocessed

    Returns
    -------
    str
        preprocessed text
    """

    stopwords = nltk.corpus.stopwords.words("english")
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = " ".join([word for word in text.split() if word not in stopwords])
    text = re.sub(r"[^\w\s\']", " ", text)
    text = re.sub(r" +", " ", text)

    return text
