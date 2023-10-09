import logging
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["load_dataset"]

PATH = ""
FOLDER = "data/01_raw/"
DATA_PATH = "telco-customer-churn.csv"

ID = "customerid"
VARS_TO_DROP = [
    "totalcallduration",
    "avgcallduration",
    "noadditionallines",
    "year",
    "month",
]


def load_data(
    path: Optional[str] = None,
    folder: Optional[str] = None,
    data_path: Optional[str] = None,
) -> pd.DataFrame:

    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER
    if data_path is None:
        data_path = path + folder + DATA_PATH

    logging.info(f"Reading data from {data_path}...")

    data = pd.read_csv(data_path)

    return data


def seq_padding(x, T=3):
    x = x.tolist()
    if len(x) < T:
        n = T - len(x)
        return x + n * [0]

    return x


def generate_dataset(df, vars_to_drop: Optional[list] = None) -> pd.DataFrame:
    if vars_to_drop is None:
        vars_to_drop = VARS_TO_DROP

    logging.info(f"Generating dataset...")

    features = [i for i in df.columns if not i in vars_to_drop]
    agg_const = df[features].groupby(ID)[features].first()
    agg_add = df.groupby(ID).agg(
        no_info_1=("month", lambda x: 1 if x.min() > 1 else 0),
        no_info_3=("month", lambda x: 1 if x.max() < 3 else 0),
        seq_total_max=("totalcallduration", "max"),
        seq_total_min=("totalcallduration", "min"),
        seq_avg_max=("avgcallduration", "max"),
        seq_avg_min=("avgcallduration", "min"),
        seq_total_range=("totalcallduration", lambda x: np.log(x.max() - x.min() + 1)),
        seq_avg_range=("avgcallduration", lambda x: np.log(x.max() - x.min() + 1)),
        seq_total=("totalcallduration", seq_padding),
        seq_avg=("avgcallduration", seq_padding),
    )

    return pd.concat([agg_const, agg_add], axis=1)
