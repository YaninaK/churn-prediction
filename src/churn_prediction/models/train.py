import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "churn_prediction"))

import logging
import numpy as np
import pandas as pd
from pickle import dump
from typing import Optional

from sklearn.preprocessing import StandardScaler

from data.make_dataset import generate_dataset
from data.validation import train_test_split
from features.time_series_features import fit_transform_seq, transform_seq
from features.binary_features import map_categorical_features
from .save_artifacts import (
    save_lstm_artifacts,
    save_nn_model_dataset,
    save_scaler_num,
    save_selected_features,
)

logger = logging.getLogger(__name__)

__all__ = ["preprocess_data"]


PATH = ""
CATEGORICAL_VARIABLES = [
    "customersuspended",
    "gender",
    "homeowner",
    "maritalstatus",
    "usesinternetservice",
    "usesvoiceservice",
]
SELECTED_NUMERIC_FEATURES = [
    "callfailurerate",
    "numberofcomplaints",
    "age",
    "unpaidbalance",
]


def data_preprocessing_pipeline(
    data: pd.DataFrame,
    path: Optional[str] = None,
    categorical_variables: Optional[list] = None,
    selected_numeric_features: Optional[list] = None,
    save_artifacts=True,
):
    if path is None:
        path = PATH
    if categorical_variables is None:
        categorical_variables = CATEGORICAL_VARIABLES
    if selected_numeric_features is None:
        selected_numeric_features = SELECTED_NUMERIC_FEATURES

    logging.info(
        "Generating dataset from raw data and mapping binary categorical features..."
    )

    df = generate_dataset(data)
    df = map_categorical_features(df)

    logging.info("Splitting dataset into train and validation parts...")

    X_train, X_valid, y_train, y_valid = train_test_split(df)

    logging.info("Generating input for LSTM model...")

    scaler_lstm, seq_train = fit_transform_seq(X_train)
    seq_valid = transform_seq(X_valid, scaler_lstm)
    lstm_artifacts = [seq_train, seq_valid, scaler_lstm]

    logging.info("Normalizing selected numeric features...")

    scaler_num = StandardScaler()
    X_train[selected_numeric_features] = scaler_num.fit_transform(
        X_train[selected_numeric_features]
    )
    X_valid[selected_numeric_features] = scaler_num.transform(
        X_valid[selected_numeric_features]
    )
    nn_model_dataset = [X_train, X_valid, y_train, y_valid]

    logging.info("Finalizing the list of selected features...")

    selected_features = selected_numeric_features + categorical_variables

    if save_artifacts:
        logging.info("Saving artifacts...")

        save_lstm_artifacts(lstm_artifacts, path)
        save_nn_model_dataset(nn_model_dataset, path)
        save_scaler_num(scaler_num, path)
        save_selected_features(selected_features, path)

    return lstm_artifacts, nn_model_dataset, selected_features
