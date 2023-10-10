import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sklean_train_test_split
from typing import Tuple

__all__ = ["train_test_split"]

TARGET_NAME = "churn"
ID = "customerid"


def train_test_data_split(
    data: pd.DataFrame, test_size: float = 0.2, seed: int = 25
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    np.random.seed(seed)
    id_1 = data[data[TARGET_NAME] == 1][ID].unique()
    id_0 = data[data[TARGET_NAME] == 0][ID].unique()

    n0, n1 = len(id_0), len(id_1)
    test_id_0 = np.random.choice(n0, int(n0 * test_size), replace=False)
    test_id_1 = np.random.choice(n1, int(n1 * test_size), replace=False)

    test_id = list(set(test_id_1) | set(test_id_0))

    data_test = data[data[ID].isin(test_id)]
    data_train = data[~data[ID].isin(test_id)]

    return data_train, data_test


def train_test_split(
    df: pd.DataFrame, test_size: float = 0.3, random_state: int = 24
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    X_train, X_valid, y_train, y_valid = sklean_train_test_split(
        df.drop(TARGET_NAME, axis=1).fillna(0),
        df[[TARGET_NAME]],
        test_size=test_size,
        random_state=random_state,
        stratify=df[[TARGET_NAME]],
    )

    return X_train, X_valid, y_train, y_valid
