import pickle

metrics_keys = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "TPR",
    "TNR",
    "FNR",
    "FPR",
    "NPV",
]

# functions to compute each metric in terms of the confusion matrix entries
def accuracy(tn, fp, fn, tp):
    return (tp + tn) / (tp + fp + tn + fn + 10e-8)


def precision(tn, fp, fn, tp):
    return tp / (tp + fp + 10e-8)


def recall(tn, fp, fn, tp):
    return tp / (tp + fn + 10e-8)


def f1_score(tn, fp, fn, tp):
    return (
        2
        * ((tp / (tp + fp + 10e-8)) * (tp / (tp + fn + 10e-8)))
        / ((tp / (tp + fp)) + (tp / (tp + fn)))
    )


def tpr(tn, fp, fn, tp):
    return tp / (tp + fn + 10e-8)


def tnr(tn, fp, fn, tp):
    return tn / (tn + fp + 10e-8)


def fnr(tn, fp, fn, tp):
    return fn / (fn + tp + 10e-8)


def fpr(tn, fp, fn, tp):
    return fp / (fp + tn + 10e-8)


def npv(tn, fp, fn, tp):
    return tn / (tn + fn + 10e-8)


metrics_functions = [
    accuracy,
    precision,
    recall,
    f1_score,
    tpr,
    tnr,
    fnr,
    fpr,
    npv,
]

# confusion matrix quadrants asociated with each metric
metric_quadrants = [
    [0, 1, 2, 3],
    [1, 3],
    [3, 2],
    [3, 1, 2],
    [3, 2],
    [0, 1],
    [3, 2],
    [0, 1],
    [0, 2],
]

# mathematical expression for each metric
metrics_formulas = [
    r"$\frac{TP + {TN}}{{TP}+{TN}+{FP}+{FN}}$",
    r"$\frac{TP}{TP + FP}$",
    r"$\frac{TP}{TP+FN}$",
    r"$2 \frac{P*R}{P+R}$",
    r"$\frac{TP}{TP+FN}$",
    r"$\frac{TN}{TN + FP}$",
    r"$\frac{FN}{FN + TP}$",
    r"$\frac{FP}{FP + TN}$",
    r"$\frac{TN}{TN + FN}$",
]

if __name__ == "__main__":
    metrics_data = {}
    for i, key in enumerate(metrics_keys):
        metrics_data[key] = {}
        metrics_data[key]["function"] = metrics_functions[i]
        metrics_data[key]["formula"] = metrics_formulas[i]
        metrics_data[key]["quadrant"] = metric_quadrants[i]

    with open("metrics_data.pkl", "wb") as f:
        pickle.dump(metrics_data, f, protocol=pickle.HIGHEST_PROTOCOL)
