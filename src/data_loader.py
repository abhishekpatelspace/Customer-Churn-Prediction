"""
data_loader.py — Data loading and cleaning for the Telco Churn dataset.
"""

import os
import pandas as pd
import numpy as np


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn CSV.

    Parameters
    ----------
    path : str, optional
        Absolute or relative path to the CSV file.
        Defaults to ``data/Telco-Customer-Churn.csv`` relative to project root.

    Returns
    -------
    pd.DataFrame
    """
    if path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(project_root, "data", "Telco-Customer-Churn.csv")

    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:
      - Convert TotalCharges from object → numeric (coerce errors to NaN)
      - Fill NaN in TotalCharges with the column median
      - Drop the customerID column (no predictive value)
      - Remove duplicate rows

    Returns a cleaned copy — the original is not mutated.
    """
    df = df.copy()

    # TotalCharges: some rows contain blank strings → convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Drop customerID
    df.drop("customerID", axis=1, inplace=True, errors="ignore")

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def get_feature_target_split(df: pd.DataFrame):
    """
    Split a cleaned dataframe into features (X) and target (y).

    The target column ``Churn`` is mapped: 'No' → 0, 'Yes' → 1.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    df = df.copy()
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return X, y
