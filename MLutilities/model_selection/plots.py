import numpy as np

from sklearn.model_selection import learning_curve
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
        estimator, X, y, cv=10, n_jobs=1, scoring=scoring, train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # plot learning curve using plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean, mode="lines", name="Training score"))
    fig.add_trace(go.Scatter(x=train_sizes, y=test_scores_mean, mode="lines", name="Testing score"))

    fig.update_layout(title="Learning Curve", xaxis_title="Training Set Size", yaxis_title=scoring, width=800, height=600)

    fig.show(renderer="notebook")
