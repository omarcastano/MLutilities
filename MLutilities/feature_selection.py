import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from lofo import LOFOImportance, FLOFOImportance, Dataset, plot_importance


def plot_lofo_importance(
    df: pd.DataFrame,
    target: str,
    scoring: str = "f1",
    cv=None,
    figsize: tuple = (12, 20),
):
    """
    plot LOFO importance (binary classification)

    Arguments:
    ----------
        df:
            Pandas DataFrame with data
        target:
            target feature
        scoring: (default f1)
            Same as scoring in in Sklearn API
        cv:
            Cross validation scheme. Same as cv in Sklearn API
        figsize: (default (12, 20))
            Size of figure

    """
    if not cv:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # define the binary target and features
    dataset = Dataset(
        df=df, target=target, features=[col for col in df.columns if col != target]
    )

    # define the validation scheme and scorer (default model is LightGBM)
    lofo_imp = LOFOImportance(dataset, cv=cv, scoring=scoring)

    # get the mean and standard deviation of the importances in pandas format
    importance_df = lofo_imp.get_importance()

    # plot the means and standard deviations of the importances
    plot_importance(importance_df, figsize=figsize)


def plot_flofo_importance(
    df: pd.DataFrame,
    target: str,
    scoring: str = "f1_macro",
    cv=None,
    figsize: tuple = (12, 20),
    model=None,
):
    """
    plot fast LOFO (FLOFO) importance.
    Applies a trained model on validation set by noising one feature each time

    Arguments:
    ----------
        df:
            Pandas DataFrame with data
        target:
            target feature
        scoring: (default f1_macro)
            Same as scoring in in Sklearn API
        cv:
            Cross validation scheme. Same as cv in Sklearn API
        model: (default RandomForestClassifier)
            Sklearn API model
        figsize: (default (12, 20))
            Size of figure
    """
    X = df.drop(columns=target)
    y = df[target]

    # check if n_samples > 1000
    if df.shape[0] < 1000:
        # repeat more data since FLOFO needs > 1000 samples
        repeats = 2000 / df.shape[0]
        df = pd.DataFrame(
            np.repeat(df.values, repeats=repeats, axis=0), columns=df.columns
        )

    # define the validation scheme, scorer, target, features and trained model
    if not model:
        model = RandomForestClassifier()
    model.fit(X, y)

    lofo_imp = FLOFOImportance(
        validation_df=df,
        target=target,
        features=[col for col in df.columns if col != target],
        scoring=scoring,
        trained_model=model,
    )

    # get the mean and standard deviation of the importances in pandas format
    importance_df = lofo_imp.get_importance()

    # plot the means and standard deviations of the importances
    plot_importance(importance_df, figsize=figsize)
