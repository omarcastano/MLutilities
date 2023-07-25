import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)

def plot_1d_binary_classification(
    threshold: float = 0.5,
    regression: str = "none",
    instance_position: int = 12,
) -> None:
    """
    helper visualization function to illustrate linear and logistic regressions in a 1D binary classification problem

    Parameters:
    ----------
        threshold:
            threshold for the decision region
        regression:
            regression to use ("none", "linear", "logistic")
        instance_position:
            position of a positive class point
    """
    n_instances_negative_class = 6
    n_instances_positive_class = 12
    start = 1
    end = 12
    classes_intercept = 5

    instances_negative_class = np.linspace(
        start, classes_intercept + 1, n_instances_negative_class
    )
    instances_positive_class = np.concatenate(
        [
            np.linspace(classes_intercept, end, n_instances_positive_class),
            [instance_position],
        ]
    )
    data = np.concatenate([instances_negative_class, instances_positive_class])

    labels_negative_class = np.zeros_like(instances_negative_class)
    labels_positive_class = np.ones_like(instances_positive_class)
    labels = np.concatenate([labels_negative_class, labels_positive_class])

    fig = go.Figure()
  
    # Create scatter plot
    fig.add_trace(
        go.Scatter(
            x=data,
            y=labels,
            mode="markers",
            marker=dict(color=labels, colorscale="picnic"),
            showlegend=False,
        )
    )
    yaxis_title = "Label"
    if regression != "none":
        # model training
        estimators = {
            "linear": LinearRegression(),
            "logistic": LogisticRegression(),
        }
        model = estimators[regression]
        model.fit(data.reshape(-1, 1), labels)

        # predictions and decision frontier
        x = np.linspace(start, instance_position)

        if regression == "linear":
            decision_frontier = (threshold - model.intercept_) / model.coef_[0]
            y_pred = model.predict(x.reshape(-1, 1))
            yaxis_title = "$y(x)$"
            y_label = r"$\hat{y}=w_0 + w_1 x_1$"
        elif regression == "logistic":
            decision_frontier = (
                (-1 / model.coef_[0]) * (model.intercept_ + np.log(1 / threshold - 1))
            )[0]
            y_pred = model.predict_proba(x.reshape(-1, 1))[:, 1]
            yaxis_title = "$p(x)$"
            y_label = r"$\hat{p} = \sigma(w_0 + w_1 x_1) = \sigma (z)$"
        ymin = min(y_pred) if min(y_pred) <= 0 else 0

        # Create line plot for predictions
        fig.add_trace(
            go.Scatter(
                x=x, y=y_pred, mode="lines", name=y_label, line=dict(color="black")
            )
        )

        # Create fill areas for decision regions
        y_limit = max(y_pred) if regression == "linear" else 1

        fill_below = fig.add_trace(
            go.Scatter(
                x=[start, decision_frontier],
                y=[y_limit, y_limit],
                fill="tozeroy",
                marker=dict(color="LightSkyBlue"),
                name="Below Threshold",
                showlegend=False,
            )
        )
        fill_above = fig.add_trace(
            go.Scatter(
                x=[decision_frontier, instance_position],
                y=[y_limit, y_limit],
                fill="tozeroy",
                marker=dict(color="lightpink"),
                name="Above Threshold",
                showlegend=False,
            )
        )

        # Create threshold line
        threshold_line = fig.add_trace(
            go.Scatter(
                x=[start, instance_position],
                y=[threshold, threshold],
                mode="lines",
                name="Threshold",
                line=dict(color="dimgray", dash="dash"),
            )
        )
    fig.update_layout(
        xaxis_title="$x$",
        yaxis_title=yaxis_title,
        width=1300,
        height=700,
        legend=dict(x=0.1, y=1.15, font=dict(size=14)),
        legend_traceorder="reversed",
    )
    fig.show()

def plot_gaussian_distributions(X: pd.DataFrame, y: pd.Series):
    """
    Plot gaussian distributions for features conditioned on class labels.

    This function visualizes Gaussian distributions for each feature in a dataset 'X',
    conditioned on their respective class labels in 'y'. The function creates a scatter plot
    of the data points colored by their class labels and overlays contour lines for the
    Gaussian probability density functions (PDFs) corresponding to each class. It also
    plots 1D Gaussian PDFs for each feature conditioned on the class labels.

    Parameters:
    -----------
        X:
          The input dataset with feature columns.
        y:
          The class labels corresponding to each data point in X.

    Example:
    --------
      # load penguins dataset
      penguins = sns.load_dataset("penguins").dropna()

      # define the feature matrix and label vector
      X = df[["bill_depth_mm", "flipper_length_mm"]]
      y = df.species

      # plot gaussians
      plot_gaussians(X, y)
    """

    def gaussian(X: pd.Series):
        """
        Compute Gaussian Probability Density Function (PDF) for a 1D array
        """
        x = np.linspace(X.min(), X.max(), 100)
        mu = X.mean()
        std = X.std()
        pdf = np.exp(-0.5 * (x - mu) ** 2 / std ** 2)
        return x, pdf

    # define figure layout
    mosaic = """
           AAA.
           BBBC
           BBBC
           BBBC
           """
    fig, ax = plt.subplot_mosaic(mosaic, figsize=(10, 10), tight_layout=True)

    # scatterplot of the data points
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y, ax=ax["B"])

    # compute the limits of the 'x' and 'y' axes for the "B" subplot and creates
    # a grid of points 'Xgrid' that spans the range of these axes.
    xlim = ax["B"].get_xlim()
    ylim = ax["B"].get_ylim()
    xg = np.linspace(xlim[0], xlim[1], 60)
    yg = np.linspace(ylim[0], ylim[1], 40)
    xx, yy = np.meshgrid(xg, yg)
    Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T

    for label, color in zip(y.unique(), ["blue", "darkorange"]):
        # for each unique class label in 'y', compute the joint probability density function (PDF)
        # and plot the contour lines of the PDF on the "B" subplot,
        # creating regions of equal probability density for each class.
        mask = y == label
        mu, std = X[mask].mean().values, X[mask].std().values
        P = np.exp(-0.5 * (Xgrid - mu) ** 2 / std ** 2).prod(1)
        ax["B"].contour(xx, yy, P.reshape(xx.shape), levels=10, colors=color, alpha=0.4)

        # call the 'gaussian' function to compute the 1D Gaussian PDFs for each feature conditioned on the current class.
        # plot these Gaussian PDFs on the "A" and "C" subplots for the first and second features, respectively.
        # These plots represent the conditional probabilities of each feature given the class label.
        x1, p1 = gaussian(X.loc[mask, X.columns[0]])
        ax["A"].plot(x1, p1, label=f"$P(x^1 | y=${label})")
        ax["A"].legend()

        x2, p2 = gaussian(X.loc[mask, X.columns[1]])
        ax["C"].plot(p2, x2, label=f"$P(x^2 | y=${label})")
        ax["C"].legend()
