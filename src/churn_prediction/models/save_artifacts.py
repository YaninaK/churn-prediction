import os
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Optional


__all__ = ["save_artifacts"]

logger = logging.getLogger()

PATH = ""

FOLDER_2 = "data/02_intermediate/"
FOLDER_3 = "data/03_primary/"

FOLDER_4 = "data/04_feature/"
SCALER_LSTM_PATH = FOLDER_4 + "scaler_lstm.joblib"
SCALER_NUM_PATH = FOLDER_4 + "scaler_num.joblib"
SELECTED_NUMERIC_FEATURES_PATH = FOLDER_4 + "selected_numeric_features_nn.joblib"
SELECTED_FEATURES_PATH = FOLDER_4 + "selected_features_nn.joblib"

FOLDER_5 = "data/05_model_input/"
SEQ_TRAIN_PATH = FOLDER_5 + "seq_train.npy"
SEQ_VALID_PATH = FOLDER_5 + "seq_valid.npy"
TRAIN_PATH = FOLDER_5 + "train.parquet.gzip"
VALID_PATH = FOLDER_5 + "valid.parquet.gzip"


def save_lstm_artifacts(
    lstm_artifacts: list,
    path: Optional[str] = None,
    seq_train_path: Optional[str] = None,
    seq_valid_path: Optional[str] = None,
    scaler_lstm_path: Optional[str] = None,
):
    logging.info("Saving lstm_artifacts...")

    [seq_train, seq_valid, scaler_lstm] = lstm_artifacts

    if path is None:
        path = PATH
    if seq_train_path is None:
        seq_train_path = SEQ_TRAIN_PATH
    if seq_valid_path is None:
        seq_valid_path = SEQ_VALID_PATH
    if scaler_lstm_path is None:
        scaler_lstm_path = SCALER_LSTM_PATH

    np.save(seq_train_path, seq_train)
    np.save(seq_valid_path, seq_valid)
    joblib.dump(scaler_lstm, scaler_lstm_path, 3)


def save_nn_model_dataset(
    nn_model_dataset: list,
    path: Optional[str] = None,
    train_path: Optional[str] = None,
    valid_path: Optional[str] = None,
):
    logging.info("Saving nn model dataset...")

    if path is None:
        path = PATH
    if train_path is None:
        train_path = TRAIN_PATH
    if valid_path is None:
        valid_path = VALID_PATH

    [X_train, X_valid, y_train, y_valid] = nn_model_dataset

    train = pd.concat([X_train, y_train], axis=1)
    valid = pd.concat([X_valid, y_valid], axis=1)
    train.to_parquet(train_path, compression="gzip")
    valid.to_parquet(valid_path, compression="gzip")


def save_scaler_num(
    scaler_num,
    path: Optional[str] = None,
    scaler_num_path: Optional[str] = None,
):
    logging.info("Saving scaler_num...")

    if path is None:
        path = PATH
    if scaler_num_path is None:
        scaler_num_path = SCALER_NUM_PATH

    joblib.dump(scaler_num, scaler_num_path, 3)


def save_selected_features(
    selected_numeric_features: list,
    selected_features: list,
    path: Optional[str] = None,
    selected_numeric_features_path: Optional[str] = None,
    selected_features_path: Optional[str] = None,
):
    logging.info("Saving selected_features...")

    if path is None:
        path = PATH
    if selected_numeric_features_path is None:
        selected_numeric_features_path = SELECTED_NUMERIC_FEATURES_PATH
    if selected_features_path is None:
        selected_features_path = SELECTED_FEATURES_PATH

    joblib.dump(selected_numeric_features, selected_numeric_features_path, 3)
    joblib.dump(selected_features, selected_features_path, 3)
