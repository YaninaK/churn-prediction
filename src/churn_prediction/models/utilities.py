import numpy as np
import matplotlib.pyplot as plt


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
