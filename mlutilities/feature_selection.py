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

    github.com/aerdem4/lofo-importance

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
    dataset = Dataset(
        df=df, target=target, features=[col for col in df.columns if col != target]
    )
    # default model is LightGBM
    lofo_imp = LOFOImportance(dataset, cv=cv, scoring=scoring)

    importance_df = lofo_imp.get_importance()
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

    github.com/aerdem4/lofo-importance

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

    if df.shape[0] < 1000:
        # repeat more data since FLOFO needs > 1000 samples
        repeats = 2000 / df.shape[0]
        df = pd.DataFrame(
            np.repeat(df.values, repeats=repeats, axis=0), columns=df.columns
        )

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

    importance_df = lofo_imp.get_importance()
    plot_importance(importance_df, figsize=figsize)
