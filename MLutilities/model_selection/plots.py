import numpy as np
from sklearn.model_selection import learning_curve, validation_curve, KFold
from plotly import graph_objs as go


# function that plots a learning curve using plotly given an estimator X, y
def plot_learning_curve(estimator, X, y, scoring=None):
    """
    Plots a learning curve using the learning_curve function from sklearn.model_selection.

    Arguments:
    ----------
    estimator: sklearn estimator
        The estimator to use for the learning curve.
    X: numpy array
        The data to fit.
    y: numpy array
        The target variable to fit.
    scoring: string
        The scoring method to use. This parameter work as the scoring parameter
        in sklearn.model_selection.learning_curve.

    Example:
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from MLutilities.model_selection.plots import plot_learning_curve
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    >>> clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    >>> plot_learning_curve(clf, X_train, y_train, scoring='accuracy')
    """

    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=10,
        n_jobs=1,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # plot learning curve using plotly
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_sizes, y=train_scores_mean, mode="lines", name="Training score"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=train_sizes, y=test_scores_mean, mode="lines", name="Testing score"
        )
    )

    fig.update_layout(
        title="Learning Curve",
        xaxis_title="Training Set Size",
        yaxis_title=scoring,
        width=800,
        height=600,
        yaxis_range=[0, 1.1]
    )

    fig.show()
    
def plot_validation_curve(
    estimator, X, y, param_name, param_range, cv=10, scoring=None
):
    """
    Plots a validation curve using the 'validation_curve' function from sklearn.model_selection.

    Arguments:
    ----------
    estimator: sklearn estimator
        The estimator to use for the learning curve.
    X: numpy array
        The data to fit.
    y: numpy array
        The target variable to fit.
    param_name: string
        Name of the parameter that will be varied.
    param_range: numpy array
        The values of the parameter that will be evaluated.
    cv: integer
        Determines the number of folds in a `KFold(n_splits=cv, shuffle=True)`
        CV splitter
    scoring: string
        The scoring method to use. This parameter work as the scoring parameter
        in sklearn.model_selection.validation_curve.
    """
    # get train and test scores
    train_scores, test_scores = validation_curve(
        estimator,
        X,
        y,
        cv=KFold(n_splits=cv, shuffle=True, random_state=42),
        n_jobs=1,
        scoring=scoring,
        param_name=param_name,
        param_range=param_range,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # plot validation curve using plotly
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=param_range, y=train_scores_mean, mode="lines", name="Training score"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=param_range, y=test_scores_mean, mode="lines", name="Testing score"
        )
    )
    fig.update_layout(
        title="Validation Curve",
        xaxis_title=param_name,
        yaxis_title=f"Score",
        width=800,
        height=600,
        yaxis_range=[0, 1.1],
    )
    fig.show()
