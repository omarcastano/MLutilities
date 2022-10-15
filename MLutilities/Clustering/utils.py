import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def plot_blobs_clustering(model: str = "kmeans", transform: bool = False):
    """
    plot clustering labels for the make_blobs dataset

    Arguments:
    ----------
    model:
        kmeans or gmm (Gaussian Mixture)
    transform:
        If True transform dataset
    """
    X, _ = make_blobs(random_state=170, n_samples=600)
    rng = np.random.RandomState(74)

    transformation = rng.normal(size=(2, 2))
    X_transform = np.dot(X, transformation)

    X = X_transform if transform else X

    labels = {
        "kmeans": KMeans(3).fit(X).labels_,
        "gmm": GaussianMixture(n_components=3).fit(X).predict(X),
    }

    plt.figure(figsize=(10, 7))
    plt.scatter(X[:, 0], X[:, 1], c=labels[model], cmap="copper")
