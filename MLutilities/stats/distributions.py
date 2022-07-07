import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
from scipy import stats

def example_histogram(mean:float=0.0, std:float=1.0, alpha:float=0.0):
    
    """
    Plots the boxplot and the histogram for a normally distributed variable
    and a skewed distributed variable

    Arguments:
        mean: Mean (“centre”) of the distributions
        std: Standard deviation (spread or width) of the distribution. Must be non-negative.
        alpha: skewness factor
    """
    
    plt.style.use('seaborn')

    rvs = skewnorm.rvs(alpha, loc=mean, scale=std, size=50000)
    normal = stats.norm.rvs(loc=mean, scale=std, size=50000)

    fig , ax = plt.subplots(2, 2, figsize=(28,8)) 

    ax[1,0].hist(normal, bins=50, edgecolor='k', label=f"mean={normal.mean().round(2)}\n" +  
                                                        f"std={normal.std().round(2)}\n" +
                                                        f"skewness={round(stats.skew(normal), 2)}\n" +
                                                        f"kurtosis={round(stats.kurtosis(normal), 2)}")

    ax[1,1].hist(rvs, bins=50, edgecolor='k', label=f"mean={rvs.mean().round(2)}\n" +  
                                                        f"std={rvs.std().round(2)}\n" +
                                                        f"skewness={round(stats.skew(rvs), 2)}\n" +
                                                        f"kurtosis={round(stats.kurtosis(rvs), 2)}")

    ax[0,1].boxplot(rvs, patch_artist=True, vert=False) 
    ax[0,0].boxplot(normal, patch_artist=True, vert=False)

    ax[0,0].set_title("Distribucion Normal", fontsize=20)
    ax[0,1].set_title("Distribucion Sesgada", fontsize=20)
    ax[0,1].legend([f"median={np.mean(rvs).round()}"], fontsize=20)
    ax[0,0].legend([f"median={np.mean(normal).round()}"], fontsize=20)
    ax[1,0].legend(fontsize=20)
    ax[1,1].legend(fontsize=20)