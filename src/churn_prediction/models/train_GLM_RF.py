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
from models.embeddings_tf import (
    fit_transform_embeddings,
    transform_embeddings,
    fit_transform_one_hot_encoding,
    transform_one_hot_encoding,
)
from .save_artifacts import (
    save_lstm_artifacts,
    save_nn_model_dataset,
    save_scaler_num,
    save_selected_features,
)

logger = logging.getLogger(__name__)

__all__ = ["preprocess_data"]


PATH = ""
SELECTED_FEATURES_PATH = "data/04_feature/selected_features_glm.joblib"
FOLDER = "data/05_model_input/"
TRAIN_PATH = FOLDER + "train_GLM.parquet.gzip"
VALID_PATH = FOLDER + "valid_GLM.parquet.gzip"

TARGET_NAME = "churn"
ID = "customerid"

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
FEATURE_FOR_EMBEDDINGS = "state"
EMBEDDINGS_DIM = 4

ONE_HOT_ENCODING_LIST = ["education", "occupation"]

DROP_LIST = [    
    TARGET_NAME,
    "seq_total",
    "seq_avg",
    "education",
    "occupation",
    "state",
]


def data_preprocessing_pipeline_GLM(
    data: pd.DataFrame,
    path: Optional[str] = None,
    train_path: Optional[str] = None,
    valid_path: Optional[str] = None,
    selected_features_path: Optional[str] = None,
    categorical_variables: Optional[list] = None,
    selected_numeric_features: Optional[list] = None,
    feature_for_embeddings: Optional[str] = None,
    embeddings_dim: Optional[int] = None,
    one_hot_encoding_list: Optional[list] = None,
    drop_list: Optional[list] = None,
    save_artifacts=True,
):
    if path is None:
        path = PATH
    if train_path is None:
        train_path = TRAIN_PATH
    if valid_path is None:
        valid_path = VALID_PATH
    if selected_features_path is None:
        selected_features_path = SELECTED_FEATURES_PATH

    if categorical_variables is None:
        categorical_variables = CATEGORICAL_VARIABLES
    if selected_numeric_features is None:
        selected_numeric_features = SELECTED_NUMERIC_FEATURES
    if feature_for_embeddings is None:
        feature_for_embeddings = FEATURE_FOR_EMBEDDINGS
    if embeddings_dim is None:
        embeddings_dim = EMBEDDINGS_DIM
    if one_hot_encoding_list is None:
        one_hot_encoding_list = ONE_HOT_ENCODING_LIST
    if drop_list is None:
        drop_list = DROP_LIST

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

    logging.info("Generating embeddings of categorical features...")

    train = pd.concat([X_train, y_train], axis=1)
    valid = pd.concat([X_valid, y_valid], axis=1)

    emb_train, lookup_and_embed = fit_transform_embeddings(
        train, feature_for_embeddings, embeddings_dim
    )
    emb_valid = transform_embeddings(
        valid, feature_for_embeddings, embeddings_dim, lookup_and_embed
    )
    new_features = emb_train.columns.tolist()

    train = pd.concat([train, emb_train], axis=1)
    valid = pd.concat([valid, emb_valid], axis=1)

    logging.info("One-hot encoding of categorical features...")

    for feature in one_hot_encoding_list:
        oh_train, str_lookup_layer = fit_transform_one_hot_encoding(train, feature)
        oh_valid = transform_one_hot_encoding(valid, feature, str_lookup_layer)
        new_features += oh_valid.columns.tolist()

        train = pd.concat([train, oh_train], axis=1)
        valid = pd.concat([valid, oh_valid], axis=1)

    logging.info("Constructing GLM dataset...")

    cols = [i for i in train.columns if not i in drop_list]
    X_train, y_train = train[cols], train[[TARGET_NAME]]
    X_valid, y_valid = valid[cols], valid[[TARGET_NAME]]
    GLM_dataset = [X_train, X_valid, y_train, y_valid]

    logging.info("Finalizing the lists of selected features GLM...")

    selected_features_GLM = (
        selected_numeric_features + categorical_variables + new_features
    )

    if save_artifacts:
        logging.info("Saving artifacts...")

        save_lstm_artifacts(lstm_artifacts, path)
        save_nn_model_dataset(GLM_dataset, path, train_path, valid_path)
        save_scaler_num(scaler_num, path)
        save_selected_features(
            selected_numeric_features,
            selected_features_GLM,
            path,
            selected_features_path,
        )

    return lstm_artifacts, GLM_dataset, selected_features_GLM
