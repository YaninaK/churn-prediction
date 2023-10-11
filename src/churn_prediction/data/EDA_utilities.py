import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_correlated_features(
    df: pd.DataFrame, numeric_variables: list, threshold = 0.95
):
    corr = df[numeric_variables].corr()
    for i in range(len(numeric_variables)):
        for j in range(i + 1, len(numeric_variables)):
            if abs(corr.iloc[i, j]) > threshold:
                corr_ = round(corr.iloc[i, j], 2)
                print(corr_, numeric_variables[i], numeric_variables[j])


def plot_ks_test(df: pd.DataFrame, feature: str, churn, no_churn): 
    df_ks = pd.DataFrame()
    df_ks[feature] = np.sort(df[feature].unique())
    df_ks["F_no_churn"] = df_ks[feature].apply(lambda x: np.mean(no_churn <= x))
    df_ks["F_churn"] = df_ks[feature].apply(lambda x: np.mean(churn <= x))

    k = np.argmax(np.abs(df_ks["F_no_churn"] - df_ks["F_churn"]))
    ks_stat = np.abs(df_ks["F_churn"][k] - df_ks["F_no_churn"][k])
    y = (df_ks["F_churn"][k] + df_ks["F_no_churn"][k]) / 2

    plt.figure(figsize=(5, 4))
    plt.plot(feature, "F_no_churn", data=df_ks, label="No_churn")
    plt.plot(feature, "F_churn", data=df_ks, label="Churn")
    plt.errorbar(
        x=df_ks[feature][k],
        y=y,
        yerr=ks_stat / 2,
        color="k",
        capsize=5,
        mew=3,
        label=f"Test statistic: {ks_stat:.4f}",
    )
    plt.legend(loc="center right")
    plt.title(f"Kolmogorov-Smirnov Test\n{feature}")
