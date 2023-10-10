#!/usr/bin/env python3
"""Inference for churn-prediction"""

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

from src.churn_prediction.models.inference_tools import preprocessing_pipeline
from src.churn_prediction.models.serialize import load


logger = logging.getLogger()


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-d",
        "--data_path",
        required=False,
        default="data/01_raw/data_test.csv",
        help="dataset store path",
    )
    argparser.add_argument(
        "-m",
        "--model_path",
        required=False,
        default="models/",
        help="model store path",
    )
    argparser.add_argument(
        "-n",
        "--model_name",
        required=False,
        default="LSTM_emb_model_v1",
        help="Model name without .keras extention",
    )
    argparser.add_argument(
        "-o",
        "--output",
        required=False,
        default="data/06_model_output/scores.csv",
        help="filename to store output",
    )
    argparser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    args = argparser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info("Reading data...")

    data = pd.read_csv(args.data_path)

    logging.info("Preprocessing data...")

    test_features, test_labels = preprocessing_pipeline(data)

    logging.info("Loading the model...")

    lstm_model = load(args.model_name, args.model_path)

    logging.info("Performing inference...")

    predictions = lstm_model.predict(test_features)

    logging.info("Calculating clients scores...")

    result = calculate_scores(predictions, test_features)

    logging.info("Saving scores...")

    result.to_csv(args.output_path)


def calculate_scores(predictions, test_features):
    result = predictions * 1000 // 10
    result = pd.DataFrame(result, index=test_features[-1].index, columns=["scores"])
    result["scores"] = result["scores"].astype(int)

    return result


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(e)
        sys.exit(1)
