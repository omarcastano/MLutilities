from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, average_precision_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
import numpy as np
from ipywidgets import widgets
from sklearn.metrics import roc_auc_score


def threshold_example(threshold=0.5):

    """
    Illustrates how model metrics such as accuracy
    depends on threshold probability

    Argumetns: 
        threshold: float
            Threshold for the probability
    """
    np.random.seed(seed=42)

    #Defines labels
    y_pred = np.random.random(size=50)
    y_true = (y_pred + np.random.uniform(low=-0.3, high=0.3, size=len(y_pred)) > 0.5)*1
    constant = [1]*len(y_pred)

    #Plot labels
    plt.figure(figsize=(15, 2))
    plt.scatter(y_pred, constant, c=y_true, cmap='Set1', s=50)
    plt.yticks(color='w')
    plt.vlines(threshold, ymin=0.95, ymax=1.05, colors='k', lw=5)
    plt.xlabel('Probability')
    
    #plot prediction countoruns
    X1, X2 = np.meshgrid(np.arange(0,1,0.01), np.arange(0.95, 1.05, 0.01))
    Z = (X1.ravel() > threshold)*1
    Z = Z.reshape(X1.shape)
    plt.contourf(X1, X2, Z, alpha=0.3, cmap="Set1")
    plt.grid(False)

    print("-------------------------")
    print(f"Accuracy = {((y_pred > threshold)*1 == y_true).sum()}/{50} =" ,((y_pred > threshold)*1 == y_true).mean())
    print("-------------------------")


#threshold dependence
def threshold_metric_evaluation(y_true, y_score, metric='Accuracy', threshold=0.5):

    """
    Plot a value of a given metric for several probability threshold. This function
    only work for a binary classification problem

    Argumetns:
        y_true: array-like
            true labels
        y_score: array-like
            predicted scores 
        metric: string
            one of the metric from the following list
            Accuracy, Precision, Recall, f1_score, FPR
            FNR, NPV, TNR
        threshold: Threshold for the probability
    """


    metrics_dict = {'Accuracy':lambda tn, fp, fn, tp:(tp+tn)/(tp+fp+tn+fn+10e-8), 
                    'Precision':lambda tn, fp, fn, tp:(tp)/(tp+fp+10e-8), 
                    'Recall':lambda tn, fp, fn, tp:(tp)/(tp+fn+10e-8),
                    'f1_score': lambda tn, fp, fn, tp:(2*((tp)/(tp+fp+10e-8))*((tp)/(tp+fn+10e-8)))/(((tp)/(tp+fp+10e-8)) + ((tp)/(tp+fn+10e-8))),
                    'FPR': lambda tn, fp, fn, tp: fp/(fp+tn),
                    'FNR': lambda tn, fp, fn, tp: fn/(fn+tp),
                    'NPV': lambda tn, fp, fn, tp: tn/(tn+fn),
                    'TNR': lambda tn, fp, fn, tp: tn/(tn+fp)
                    }

    metrics = []
    thresholds = []

    for t in np.arange(0.01,0.99,0.01):
        y_pred = (y_score[:,1] >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.append(metrics_dict[metric](tn, fp, fn, tp))
        thresholds.append(t)


    thresholds = np.array(thresholds)
    metrics = np.array(metrics)

    idx = (np.abs(np.array(thresholds) - threshold)).argmin()

    fig, ax = plt.subplots(1,2 ,figsize=(25,8))
    ax[0].plot(thresholds, metrics, lw=5)
    
    ax[0].set_xlabel('Threshold', fontsize=15)
    ax[0].set_ylabel(f'{metric}', fontsize=15)

    ax[0].vlines(threshold, ymin=np.min(metrics), ymax=1.0, colors='r')
    ax[0].text(threshold, 0.85, f'{metric}:{metrics[idx].round(3)}', fontsize=20)

    ax[0].plot(thresholds[metrics.argmax()], metrics.max(), 'rD')


    #Confusion Matrix
    y_pred = (y_score[:,1] >= threshold).astype(int)
    conf_mt = confusion_matrix(y_true, y_pred)
    conf_mt_str = conf_mt.copy().astype(str)
    conf_mt_str[0,0] = "TN = " + str(conf_mt[0,0])
    conf_mt_str[1,1] = "TP = " + str(conf_mt[1,1])
    conf_mt_str[1,0] = "FN = " + str(conf_mt[1,0])
    conf_mt_str[0,1] = "FP = " + str(conf_mt[0,1])
    
    sns.heatmap(conf_mt, annot=conf_mt_str, cbar=False, cmap="Blues", fmt="" ,annot_kws={'size':20}, ax=ax[1])
    ax[1].set_title("Congusion Matrix", fontsize=20)
    ax[1].set_xlabel("Predicted Label", fontsize=20)
    ax[1].set_ylabel("True Label", fontsize=20)


def precision_recall_tradeoff(y_true, y_score, threshold=0.5):

    """
    This function allow you to evaluate the precision-recall tradeoff

    Arguments:
        y_true: numpy array
            true labels
        y_score: numpy array
            predicted scores such as probability
        threshold: float
            classification threshold for the probability
    """

    precision = []
    recall = []

    metrics_dict = {'Accuracy':lambda tn, fp, fn, tp:(tp+tn)/(tp+fp+tn+fn+10e-8), 
                'Precision':lambda tn, fp, fn, tp:(tp)/(tp+fp+10e-8), 
                'Recall':lambda tn, fp, fn, tp:(tp)/(tp+fn+10e-8),
                'f1_score': lambda tn, fp, fn, tp:(2*((tp)/(tp+fp+10e-8))*((tp)/(tp+fn+10e-8)))/(((tp)/(tp+fp+10e-8)) + ((tp)/(tp+fn+10e-8)))
                }

    metrics = []
    thresholds = []

    for t in np.arange(0.01,0.99,0.01):
        y_pred = (y_score[:,1] >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision.append(metrics_dict['Precision'](tn, fp, fn, tp))
        recall.append(metrics_dict['Recall'](tn, fp, fn, tp))
        thresholds.append(t)

   
    thresholds = np.array(thresholds)

    recall = np.array(recall)
    precision = np.array(precision)

    idx = (np.abs(np.array(thresholds) - threshold)).argmin()
    
    fig, ax = plt.subplots(1,2 ,figsize=(25,8))
    ax[0].plot(thresholds, precision, lw=5, label='precision')
    ax[0].plot(thresholds, recall, lw=5, label='recall')

    ax[0].set_xlabel('Threshold', fontsize=15)
    ax[0].set_ylabel('Score', fontsize=15)
    
    ax[0].vlines(threshold, ymin=0.0, ymax=1.0, colors='r')
    ax[0].text(threshold, 0.85, f'Precisin:{precision[idx].round(3)}\n recall:{recall[idx].round(3)}\n f1_score:{((2*precision[idx]*recall[idx])/(precision[idx]+recall[idx])).round(2)}', fontsize=20)
    ax[0].legend()


    #Confusion Matrix
    y_pred = (y_score[:,1] >= threshold).astype(int)
    conf_mt = confusion_matrix(y_true, y_pred)

    sns.heatmap(conf_mt, annot=True, cbar=False, cmap="Blues", fmt='.5g' ,annot_kws={'size':20}, ax=ax[1])
    ax[1].set_title("Congusion Matrix", fontsize=20)
    ax[1].set_xlabel("Predicted Label", fontsize=20)
    ax[1].set_ylabel("True Label", fontsize=20)


def precision_recall_curve(y_true, y_score, threshold=0.5):

    """
    Plots Preciison recall curve and PR AUC

    Arguments:
        y_true: numpy array
            true labels
        y_score: numpy array
            predicted scores such as probability
        threshold: float
            classification threshold for the probability
    """

    precision = []
    recall = []

    metrics_dict = {'Accuracy':lambda tn, fp, fn, tp:(tp+tn)/(tp+fp+tn+fn+10e-8), 
                'Precision':lambda tn, fp, fn, tp:(tp)/(tp+fp+10e-8), 
                'Recall':lambda tn, fp, fn, tp:(tp)/(tp+fn+10e-8),
                'f1_score': lambda tn, fp, fn, tp:(2*((tp)/(tp+fp+10e-8))*((tp)/(tp+fn+10e-8)))/(((tp)/(tp+fp+10e-8)) + ((tp)/(tp+fn+10e-8)))
                }

    metrics = []
    thresholds = []

    for t in np.arange(0.01,0.99,0.01):
        y_pred = (y_score[:,1] >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision.append(metrics_dict['Precision'](tn, fp, fn, tp))
        recall.append(metrics_dict['Recall'](tn, fp, fn, tp))
        thresholds.append(t)


   
    thresholds = np.array(thresholds)

    recall = np.array(recall)
    precision = np.array(precision)

    idx = (np.abs(np.array(thresholds) - threshold)).argmin()
    
    fig, ax = plt.subplots(1,2 ,figsize=(25,8))

    ax[0].plot(recall, precision, lw=5)

    ax[0].set_ylabel('Precision', fontsize=20)
    ax[0].set_xlabel('Recall', fontsize=20)
    ax[0].plot(recall[idx].round(3), precision[idx].round(3), 'rD', markersize=10)
    ax[0].text(recall[idx].round(3), precision[idx].round(3),
                f'Precisin:{precision[idx].round(3)}\n recall:{recall[idx].round(3)}\n f1_score:{((2*precision[idx]*recall[idx])/(precision[idx]+recall[idx])).round(2)}', fontsize=20)


    #Confusion Matrix
    y_pred = (y_score[:,1] >= threshold).astype(int)
    conf_mt = confusion_matrix(y_true, y_pred)
    ax[0].set_title(f'PR AUC:{average_precision_score(y_true, y_score[:,1]).round(3)}', fontsize=25)

    sns.heatmap(conf_mt, annot=True, cbar=False, cmap="Blues", fmt='.5g' ,annot_kws={'size':20}, ax=ax[1])
    ax[1].set_title("Congusion Matrix", fontsize=20)
    ax[1].set_xlabel("Predicted Label", fontsize=20)
    ax[1].set_ylabel("True Label", fontsize=20)

def ROC_curve(y_true, y_score, threshold=0.5):

    """
    Compute ROC curve and AUC

    Arguments:

    """

    TPR = []
    FPR = []
    thresholds = []

    for t in np.arange(0.01,0.99,0.01):
            y_pred = (y_score[:,1] >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            TPR.append(tp/(tp + fn))
            FPR.append(fp/(fp + tn))
            thresholds.append(t)


    thresholds = np.array(thresholds)

    TPR = np.array(TPR)
    FPR = np.array(FPR)

    idx = (np.abs(np.array(thresholds) - threshold)).argmin()

    fig, ax = plt.subplots(1,2 ,figsize=(25,8))

    

    ax[0].plot(FPR, TPR, lw=5)

    ax[0].set_ylabel('TPR', fontsize=20)
    ax[0].set_xlabel('FPR', fontsize=20)
    ax[0].plot(FPR[idx].round(3), TPR[idx].round(3), 'rD', markersize=10)
    ax[0].text(FPR[idx].round(3), TPR[idx].round(3),
                f'TPR:{TPR[idx].round(3)}\n FPR:{FPR[idx].round(3)}', fontsize=20)


    #Confusion Matrix
    y_pred = (y_score[:,1] >= threshold).astype(int)
    conf_mt = confusion_matrix(y_true, y_pred)
    conf_mt_str = conf_mt.copy().astype(str)
    conf_mt_str[0,0] = "TN = " + str(conf_mt[0,0])
    conf_mt_str[1,1] = "TP = " + str(conf_mt[1,1])
    conf_mt_str[1,0] = "FN = " + str(conf_mt[1,0])
    conf_mt_str[0,1] = "FP = " + str(conf_mt[0,1])

    ax[0].set_title(f'ROC AUC:{roc_auc_score(y_true, y_score[:,1]).round(3)}', fontsize=25)

    sns.heatmap(conf_mt, annot=conf_mt_str, cbar=False, cmap="Blues", fmt="" ,annot_kws={'size':20}, ax=ax[1])
    ax[1].set_title("Congusion Matrix", fontsize=20)
    ax[1].set_xlabel("Predicted Label", fontsize=20)
    ax[1].set_ylabel("True Label", fontsize=20)