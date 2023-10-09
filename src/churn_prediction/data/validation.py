import pandas as pd
from sklearn.model_selection import train_test_split as sklean_train_test_split
from typing import Tuple

__all__ = ["train_test_split"]

TARGET_NAME = "churn"


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
