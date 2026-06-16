"""
feature_engineering.py — Preprocessing and feature engineering utilities.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def identify_column_types(X: pd.DataFrame):
    """
    Auto-detect categorical vs numerical columns in the feature matrix.

    Returns
    -------
    cat_cols : list[str]
    num_cols : list[str]
    """
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    return cat_cols, num_cols


def build_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that:
      - One-hot-encodes categorical columns (handles unknown categories)
      - Passes through numerical columns as-is

    Returns
    -------
    ColumnTransformer
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def create_tenure_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``TenureGroup`` column based on tenure bins.

    Bins: 0-12, 12-24, 24-48, 48-72 months.

    Returns a copy with the new column.
    """
    df = df.copy()
    bins = [0, 12, 24, 48, 72]
    labels = ["0-12", "12-24", "24-48", "48-72"]
    df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    return df
