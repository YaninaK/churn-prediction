import logging
import pandas as pd


def map_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X["customersuspended"] = X["customersuspended"].map({"Yes": 1, "No": 0})
    X["gender"] = X["gender"].map({"Male": 1, "Female": 0})
    X["homeowner"] = X["homeowner"].map({"Yes": 1, "No": 0})
    X["maritalstatus"] = X["maritalstatus"].map({"Single": 1, "Married": 0})
    X["usesinternetservice"] = X["usesinternetservice"].map({"Yes": 1, "No": 0})
    X["usesvoiceservice"] = X["usesvoiceservice"].map({"Yes": 1, "No": 0})

    return X
