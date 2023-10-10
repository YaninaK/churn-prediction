from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_initial_bias_and_class_weight(y_train):
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()

    initial_bias = np.log([pos / neg])[0]
    print(f"initial_bias: {initial_bias}\n")

    total = y_train.shape[0]
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"Weight for class 0: {round(weight_for_0[0], 2)}")
    print(f"Weight for class 1: {round(weight_for_1[0], 2)}")

    return initial_bias, class_weight


def plot_loss(history):
    plt.semilogy(history.epoch, history.history["loss"], label="Train")
    plt.semilogy(
        history.epoch, history.history["val_loss"], label="Valid", linestyle="--"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM model Loss")
    plt.legend()


def plot_metrics(history):
    metrics = ["loss", "prc", "precision", "recall"]
    plt.figure(figsize=(10, 8))
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], label="Train")
        plt.plot(
            history.epoch, history.history["val_" + metric], linestyle="--", label="Val"
        )
        plt.xlabel("Epoch")
        plt.ylabel(name)
        if metric == "loss":
            plt.ylim([0, plt.ylim()[1]])
        elif metric == "auc":
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()


def plot_cm(labels, predictions, threshold=0.5):
    cm = confusion_matrix(labels, predictions > threshold)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix @{:.2f}".format(threshold))
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    print("No churn Detected (True Negatives): ", cm[0][0])
    print("No churn Incorrectly Detected (False Positives): ", cm[0][1])
    print("Churn Missed (False Negatives): ", cm[1][0])
    print("Churn Detected (True Positives): ", cm[1][1])
    print("Total Fraudulent Transactions: ", np.sum(cm[1]))


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel("False positives [%]")
    plt.ylabel("True positives [%]")
    plt.title("ROC Curve")
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect("equal")


def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Precision-Recall curve")
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect("equal")
