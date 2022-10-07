import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lofo import LOFOImportance, FLOFOImportance, Dataset, plot_importance


def plot_lofo_importance(
    dataset: pd.DataFrame,
    target: str,
    cv,
    scoring: str,
    figsize: tuple = (12, 20),
    model=None,
):
    """
    plot LOFO (binary classification) or FLOFO (multiclass classification) importance

    Arguments:
    ----------
        df:
            Pandas DataFrame with data
        target:
            target feature
        cv:
            Cross validation scheme. Same as cv in Sklearn API
        scoring: (default f1)
            Same as scoring in in Sklearn API
        figsize: (default (12, 20))
            Size of figure
        model:
            Sklearn API model (used in FLOFO)
    """
    X = df.drop(target)
    y = df[target]

    n_labels = y.nunique()
    if n_labels == 2:
        if not scoring:
            scoring = "f1"

        # define the binary target and features
        dataset = Dataset(
            df=df, target=target, features=[col for col in df.columns if col != target]
        )

        # define the validation scheme and scorer (default model is LightGBM)
        lofo_imp = LOFOImportance(dataset, cv=cv, scoring=scoring)

    else:
        if not scoring:
            scoring = "f1_macro"

        # check if n_samples > 1000
        if dataset.shape[0] < 1000:
            # repeat more data since FLOFO needs > 1000 samples
            repeats = 2000 / dataset.shape[0]
            df = pd.DataFrame(
                np.repeat(df.values, repeats=repeats, axis=0), columns=df.columns
            )

        # train model
        if not model:
            model = RandomForestClassifier()
        model.fit(X, y)

        # define the validation scheme, scorer, target, features and trained model
        lofo_imp = FLOFOImportance(
            validation_df=df,
            target=target,
            features=[col for col in df.columns if col != target],
            cv=cv,
            scoring=scoring,
            trained_model=model,
        )

    # get the mean and standard deviation of the importances in pandas format
    importance_df = lofo_imp.get_importance()

    # plot the means and standard deviations of the importances
    plot_importance(importance_df, figsize=figsize)