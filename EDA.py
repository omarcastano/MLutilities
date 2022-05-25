import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
import numpy as np

"""
This module provides some exploratory data analysis tools.
"""

def KolmogorovTest(dataset, variable, apply_yeo_johnson=False, apply_log_transform=False ,plot_histogram=False, color=None):

    """
    This function computes Kolmogorov test to check if the variable
    is normaly distributed

    H0: The variable follows a normal distribution
    H1: The variable do not follow a normal distribution

    if p_value < 0.05 you can reject the null hypohesis

    Arguments:
        dataset: pandas dataframe
        variable: string
            variable to performe the Kolmogorov test
        apply_yeo_johnson: bool
            If True appy yeo johnson transformation to the input variable 
        apply_log_transform: bool
            If True apply logarithm transformation to the input variable
        plot_histogram: bool
            If True plot a histogram of the variable
        color: string
            Name of column in dataset. Values from this column are used to
            assign color to marks.
    """
    

    if apply_yeo_johnson:
        x = stats.yeojohnson(dataset[variable].to_numpy())[0]
    elif apply_log_transform:
        x = np.log1p(dataset[variable].to_numpy())
    else:
        x = dataset[variable].to_numpy()

    ktest = stats.kstest(x, 'norm')
    print('------------------------------------------------------------------------------')
    print(f'statistic={ktest[0]}, p_value={ktest[1]}\n')
    if ktest[1] < 0.05:
        print(f'Since {ktest[1]} < 0.05 you can reject the null hypothesis, so the variable \ndo not follow a normal distribution')
    else:
        print(f'Since {ktest[1]} > 0.05 you cannot reject the null hypothesis, so the variable \nfollows a normal distribution')
    print('------------------------------------------------------------------------------\n')
    if plot_histogram:
        fig = px.histogram(dataset, x=x, nbins=30, marginal='box', color=color, barmode='overlay')
        fig.update_traces(marker_line_width=1, marker_line_color="white", opacity=0.8)
        fig.update_layout(xaxis_title=variable)
        fig.show()


def BiserialCorrelation(dataset, target_variable, input_variable, apply_yeo_johnson=False, apply_log_transform=False, test_assumptions=False):
    
    """
        A point-biserial correlation is used to measure the correlation between
        a continuous variable and a binary variable.

        Assumption: continuous data within each group created by the binary variable
        are normally distributed with equal variances and possibly different means.

        H0: variables are not correlated
        H1: variables are correlated

        If p_value < 0.05 reject the null hypothesis

        Arguments:
        dataset: pandas dataframe
        target_variable: string
            Name of the target variable  
        input_varaible: string
            Name of the input variable
        apply_yeo_johnson: bool
            If True appy yeo johnson transformation to the input variable
        apply_log_transform: bool
            If True apply logarithm transformation to the input variable
        test_assumptions: bool
            If True test the assuptioms for the continuos variable
    """

    if test_assumptions:
        y_unique = dataset[target_variable].unique()
        x1 = dataset.loc[dataset[target_variable] == y_unique[0] ,[input_variable]]
        x2 = dataset.loc[dataset[target_variable] == y_unique[1] ,[input_variable]]

        print(f'------------------------Kolmogorov Test for y:{y_unique[0]}---------------------------')
        KolmogorovTest(x1, input_variable, apply_yeo_johnson=apply_yeo_johnson, apply_log_transform=apply_log_transform, plot_histogram=False)
        print(f'------------------------Kolmogorov Test for y:{y_unique[1]}---------------------------')
        KolmogorovTest(x1, input_variable, apply_yeo_johnson=apply_yeo_johnson, apply_log_transform=apply_log_transform, plot_histogram=False)

        print('--------------------------------Levene Test-----------------------------------')
        LeveneTest(dataset, target_variable, input_variable)

    #Point Biserial correlation Test
    y = LabelEncoder().fit_transform(dataset[target_variable])

    if apply_box_cox:
        x = stats.yeojohnson(dataset[input_variable].to_numpy())[0]
    elif apply_log_transform:
        x = np.log1p(dataset[variable].to_numpy())
    else:
        x = dataset[input_variable].to_numpy()

    biserial = stats.pointbiserialr(y, x)
    print('---------------------------Point Biserial Test--------------------------------')
    print(f'statistic={biserial[0]}, p_value={biserial[1]}\n')
    if biserial[1] < 0.05:
        print(f'Since {biserial[1]} < 0.05 you can reject the null hypothesis, \nso variables are correlated')
    else:
        print(f'Since {biserial[1]} > 0.05 you cannot reject the null hypothesis, \nso variables are not correlated')
    print('------------------------------------------------------------------------------\n')


def LeveneTest(dataset, target_variable, input_variable):

    """
    Levene’s test is used to check that variances are equal for all 
    samples when your data comes from a non normal distribution. This
    function is created to work only when target variable is binary 

    H0: variances_1 = variances_2 = variances.
    H2: variances_1 != variances_2.

    if p_values < 0.05 rejecct the null hypothesis

    Arguments:
        dataset: pandas dataframe
        target_variable: string
            Name of the target variable  
        input_varaible: string
            Name of the input variable
    """

    y_unique = dataset[target_variable].unique()
    
    x1 = dataset.loc[dataset[target_variable] == y_unique[0] ,input_variable].to_numpy()
    x2 = dataset.loc[dataset[target_variable] == y_unique[1] ,input_variable].to_numpy()

    levene = stats.levene(x1, x2)

    print('------------------------------------------------------------------------------')
    print(f'statistic={levene[0]}, p_value={levene[1]}\n')
    if levene[1] < 0.05:
        print(f'Since {levene[1]} < 0.05 you can reject the null hypothesis, \nso variances_1 != variances_2')
    else:
        print(f'Since {levene[1]} > 0.05 you cannot reject the null hypothesis, \nso variances_1 = variances_2')
    print('------------------------------------------------------------------------------\n')


def KruskalTest(dataset, target_variable, input_variable):

    """
    The Kruskal-Wallis H test is a rank-based nonparametric test 
    that can be used to determine if there are statistically significant 
    differences between two or more groups of an independent variable on 
    a continuous or ordinal dependent variable. This function is created 
    to work only when target variable is binary 

    Assumption:
        - Continuoues variable not need to follow a normal distribution 
        - The distributions in each group should have the same shape.

    H0: medians_1 = medians_2 = medians.
    H2: medians_1 != medians_2.

    If p_values < 0.05 rejecct the null hypothesis

    Arguments:
        dataset: pandas dataframe
        target_variable: string
            Name of the target variable  
        input_varaible: string
            Name of the input variable
    """

    y_unique = dataset[target_variable].unique()
    
    x1 = dataset.loc[dataset[target_variable] == y_unique[0] ,input_variable]
    x2 = dataset.loc[dataset[target_variable] == y_unique[1] ,input_variable]

    print('--------------------------------Skewness and Kurtosis-------------------------')
    print( f'Skweness and for kurtosis for y:{y_unique[0]}. Skweness={round(x1.skew(),3)}, Kurtosis={round(x1.kurtosis(),3)}')
    print( f'Skweness and for kurtosis for y:{y_unique[1]}. Skweness={round(x2.skew(),3)}, Kurtosis={round(x2.kurtosis(),3)}')
    print('------------------------------------------------------------------------------\n')

    kruskal = stats.kruskal(x1, x2)
    print('------------------------------------------------------------------------------')
    print(f'statistic={kruskal[0]}, p_value={kruskal[1]}\n')
    if kruskal[1] < 0.05:
        print(f'Since {kruskal[1]} < 0.05 you can reject the null hypothesis, \nso we have that medians_1 != medians_2')
    else:
        print(f'Since {kruskal[1]} > 0.05 you cannot reject the null hypothesis \nso we have that medians_1 = medians_2')
    print('------------------------------------------------------------------------------\n')
    
 

#Creamers V Correlation
def CramersV(dataset, input_feature, target_feature, plot_histogram=True ,histnorm='percent', color=None):
    """
    This function computes cramer's V correlation coefficient which is a measure of association between two nominal variables.
    
    H0: there is not a relationship between the variables.
    H1: there is a relationship between the variables.

    Arguments:
        dataset: pandas dataframe
        target_variable: string
            Name of the target variable  
        input_varaible: string
            Name of the input variable
        plot_histogram: bool
            If True plot a histogram
        histnorm: string (default='percentage')
            It can be either 'percent' or 'count'. If 'percent'
            show the percengate of each category, if 'count' show 
            the frequency of each category.
        color: string
            Name of column in dataset. Values from this column are used to
            assign color to marks.
            
    If p_value < 0.5 you can reject the null hypothesis
    """

    obs = pd.crosstab(dataset[input_feature], dataset[target_feature])
    chi2, p, dof, ex = stats.chi2_contingency(obs, correction=False)

    dimension = dataset[[input_feature, target_feature]].notnull().prod(axis=1).sum()
    cramer = np.sqrt( (chi2/dimension) /  (np.min(obs.shape) -1 ))
    print("------------------------------------------------------------------------------------------------------")
    print(f'CramersV: {cramer}, chi2:{chi2}, p_value:{p}\n')
    if p < 0.05:
        print(f'Since {p} < 0.05 you can reject the null hypothesis, \nso there is a relationship between the variables.')
    else:
        print(f'Since {p} > 0.05 you cannot reject the null hypothesis, \nso there is not a relationship between the variables.')
    print("------------------------------------------------------------------------------------------------------\n")

    if plot_histogram:
        fig= px.histogram(dataset, x=input_feature, histnorm=histnorm, color=color, barmode='group')
        fig.show()
