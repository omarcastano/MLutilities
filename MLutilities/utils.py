import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


def cramerv_relationship_strength(degrees_of_freedom, cramerv):
    """
    returns the strength of the relationship of two categorical variables

    source: https://www.statology.org/interpret-cramers-v/
    """
    values = {
        "1": [0.10, 0.50],
        "2": [0.07, 0.35],
        "3": [0.06, 0.29],
        "4": [0.05, 0.25],
        "5": [0.04, 0.22],
    }

    if np.round(cramerv, 2) <= values[str(degrees_of_freedom)][0]:
        return "small"
    elif np.round(cramerv, 2) >= values[str(degrees_of_freedom)][-1]:
        return "high"
    else:
        return "medium"


def scaler(
    dataset: pd.DataFrame = None,
    variables: List[str] = None,
    kind: str = "standar_scaler",
):

    """
    Helper function to visualize the efect of scaling and normalization over continuos variables

    Arguments:
        dataset: pandas dataframe or dict with the format {'col1':np.array, 'col2':np.array}
        varaibles: list with the name of the features to scale
        kind: name of the transformation to perform. Options ["standar_scaler", "minmax"]
    """

    scale = {
        "standar_scaler": StandardScaler().fit_transform,
        "minmax": MinMaxScaler().fit_transform,
    }

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    for i, var in enumerate(variables):
        sns.kdeplot(dataset[var], ax=ax[0], label=var)
        sns.kdeplot(scale[kind](dataset[[var]]).ravel(), ax=ax[1], label=var)

    ax[0].set_title("Original")
    ax[1].set_title("Transform")
    ax[0].legend()
    ax[1].legend()
