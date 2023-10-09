import logging
import pandas as pd
import numpy as np
from typing import Optional

from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)

__all__ = ["generate_time_series_features"]


def preprocess_seq(df: pd.DataFrame):
    seq = np.array([df["seq_total"].tolist(), df["seq_avg"].tolist()]).reshape(2, -1).T

    return seq


def fit_transform_seq(df: pd.DataFrame):
    seq = preprocess_seq(df)

    scaler = StandardScaler()
    seq = scaler.fit_transform(seq)

    return scaler, seq.reshape(-1, 3, 2)


def transform_seq(df: pd.DataFrame, scaler):
    seq = preprocess_seq(df)
    seq = scaler.transform(seq)

    return seq.reshape(-1, 3, 2)
