import numpy as np
import pandas as pd
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
