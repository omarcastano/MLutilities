import ipywidgets as widgets
from functools import partial
from IPython.display import display
from .utils import plot_blobs_clustering


def blobs_clustering_widget():
    """
    Helper widget to visualize the clustering labels for the make_blobs dataset
    """
    model = widgets.Dropdown(
        description="Model:",
        options=["kmeans", "gmm"],
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )
    transform = widgets.Dropdown(
        options=[True, False],
        value=False,
        description="Transform:",
        layout=widgets.Layout(width="20%", height="30px"),
        style={"description_width": "initial"},
    )

    w = widgets.interactive_output(
        partial(plot_blobs_clustering),
        {
            "model": model,
            "transform": transform,
        },
    )
    display(widgets.VBox([model, transform]), w)
