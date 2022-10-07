from lofo import LOFOImportance, Dataset, plot_importance
from typing import List
import pandas as pd


def plot_lofo_importance(
    dataset: pd.DataFrame,
    target: str,
    features: List[str],
    cv,
    scoring: str = "f1",
    figsize: tuple = (12, 20),
):
    """
    plot LOFO importance

    Arguments:
    ----------
        dataset:
            Pandas DataFrame with data
        target:
            target feature
        features:
            List of features
        cv:
            Cross validation scheme. Same as cv in Sklearn API
        scoring: (default f1)
            Same as scoring in in Sklearn API
        figsize: (default (12, 20))
            Size of figure
    """
    # define the binary target and features
    dataset = Dataset(df=dataset, target=target, features=features)

    # define the validation scheme and scorer (default model is LightGBM)
    lofo_imp = LOFOImportance(dataset, cv=cv, scoring=scoring)

    # get the mean and standard deviation of the importances in pandas format
    importance_df = lofo_imp.get_importance()

    # plot the means and standard deviations of the importances
    plot_importance(importance_df, figsize=figsize)
