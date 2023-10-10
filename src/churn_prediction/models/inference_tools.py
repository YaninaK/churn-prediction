import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "churn_prediction"))

import logging
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Tuple

from data.make_dataset import generate_dataset
from features.binary_features import map_categorical_features
from features.time_series_features import transform_seq


logger = logging.getLogger(__name__)

__all__ = ["inference_preprocessing_pipeline"]


PATH = ""

FOLDER = "data/04_feature/"
SCALER_LSTM_PATH = FOLDER + "scaler_lstm.joblib"
SCALER_NUM_PATH = FOLDER + "scaler_num.joblib"
SELECTED_NUMERIC_FEATURES_PATH = FOLDER + "selected_numeric_features_nn.joblib"
SELECTED_FEATURES_PATH = FOLDER + "selected_features_nn.joblib"

TARGET_NAME = "churn"


def preprocessing_pipeline(
    data: pd.DataFrame,
    path: Optional[str] = None,
    scaler_lstm_path: Optional[str] = None,
    scaler_num_path: Optional[str] = None,
    selected_numeric_features_path: Optional[str] = None,
    selected_features_path: Optional[str] = None,
) -> Tuple[list, pd.DataFrame]:

    if path is None:
        path = PATH
    if scaler_lstm_path is None:
        scaler_lstm_path = path + SCALER_LSTM_PATH
    if scaler_num_path is None:
        scaler_num_path = path + SCALER_NUM_PATH
    if selected_numeric_features_path is None:
        selected_numeric_features_path = path + SELECTED_NUMERIC_FEATURES_PATH
    if selected_features_path is None:
        selected_features_path = path + SELECTED_FEATURES_PATH

    df = generate_dataset(data)
    df = map_categorical_features(df)
    X_test = df.drop(TARGET_NAME, axis=1)
    test_labels = df[[TARGET_NAME]]

    logging.info("Loading scaler_lstm...")

    scaler_lstm = joblib.load(scaler_lstm_path)
    seq_test = transform_seq(X_test, scaler_lstm)

    logging.info("Loading selected_numeric_features...")

    selected_numeric_features = joblib.load(selected_numeric_features_path)

    logging.info("Loading scaler_num...")

    scaler_num = joblib.load(scaler_num_path)
    X_test[selected_numeric_features] = scaler_num.transform(
        X_test[selected_numeric_features]
    )
    logging.info("Loading selected_features...")

    selected_features = joblib.load(selected_features_path)

    test_features = [
        seq_test,
        X_test["state"],
        X_test["education"],
        X_test["occupation"],
        X_test[selected_features],
    ]

    return test_features, test_labels
