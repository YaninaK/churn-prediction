#!/usr/bin/env python3
"""Train and save model for churn-prediction"""

import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), ".."))

import logging
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Optional

from src.churn_prediction.data.make_dataset import load_data
from src.churn_prediction.models import train
from src.churn_prediction.models.utilities import get_initial_bias_and_class_weight
from src.churn_prediction.models.LSTM_embeddings_model import get_LSTM_model
from src.churn_prediction.models.serialize import store


logger = logging.getLogger()


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-d",
        "--data_path",
        required=False,
        default="data/01_raw/telco-customer-churn.csv",
        help="dataset store path",
    )
    argparser.add_argument(
        "-o1",
        "--output1",
        required=True,
        help="filename to store model",
    )
    argparser.add_argument(
        "-o2",
        "--output2",
        required=True,
        help="filename to store model history",
    )
    argparser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    args = argparser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info("Reading data...")

    data = load_data(args.data_path)

    logging.info("Preprocessing data...")

    (
        lstm_artifacts,
        nn_model_dataset,
        selected_features,
    ) = train.data_preprocessing_pipeline(data)

    logging.info("Training the model...")

    train_store(
        lstm_artifacts, nn_model_dataset, selected_features, args.output1, args.output2
    )


def train_store(
    lstm_artifacts: list,
    nn_model_dataset: list,
    selected_features: list,
    filename_model: str,
    filename_history: str,
):
    """
    Trains and stores LSTM embedding model.
    """

    [seq_train, seq_valid, scaler_lstm] = lstm_artifacts
    [X_train, X_valid, y_train, y_valid] = nn_model_dataset

    logging.info(f"Training the model on {len(X_train)}  items...")

    vocab_s = X_train["state"].unique().tolist()
    n_labels_s = len(vocab_s)
    vocab_e = X_train["education"].unique().tolist()
    vocab_o = X_train["occupation"].unique().tolist()
    n_features_other = len(selected_features)

    params = {
        "input_sequence_length": 3,
        "n_features": 2,
        "n_units": 8,
        "vocab_s": vocab_s,
        "n_labels_s": n_labels_s,
        "embedding_size_s": 4,
        "vocab_e": vocab_e,
        "vocab_o": vocab_o,
        "n_features_other": n_features_other,
        "n_units_others": 8,
        "n_units_all": 16,
    }

    initial_bias, class_weight = get_initial_bias_and_class_weight(y_train)
    lstm_model = get_LSTM_model(**params, output_bias=initial_bias)

    n_epochs = 100
    batch_size = 64
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 3e-2 * 0.95**epoch
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_prc",
        patience=20,
        min_delta=1e-06,
        verbose=1,
        mode="max",
        restore_best_weights=True,
    )

    history = lstm_model.fit(
        [
            seq_train,
            X_train["state"],
            X_train["education"],
            X_train["occupation"],
            X_train[selected_features],
        ],
        y_train,
        epochs=n_epochs,
        validation_data=(
            [
                seq_valid,
                X_valid["state"],
                X_valid["education"],
                X_valid["occupation"],
                X_valid[selected_features],
            ],
            y_valid,
        ),
        class_weight=class_weight,
        batch_size=batch_size,
        verbose=0,
        callbacks=[reduce_lr, early_stopping],
        shuffle=True,
        workers=-1,
        use_multiprocessing=True,
    )

    store(lstm_model, history, filename_model, filename_history)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(e)
        sys.exit(1)
