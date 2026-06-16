"""
predict.py — Load a saved model and make predictions.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np


# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


# ── Loader ───────────────────────────────────────────────────────────────────

def load_model(model_dir: str | None = None):
    """
    Load the saved model pipeline and metadata.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
    metadata : dict
    """
    if model_dir is None:
        model_dir = MODELS_DIR

    model_path = os.path.join(model_dir, "churn_model.joblib")
    meta_path = os.path.join(model_dir, "model_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            "Run `python -m src.train` first."
        )

    pipeline = joblib.load(model_path)

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    return pipeline, metadata


# ── Single prediction ────────────────────────────────────────────────────────

def predict_single(customer: dict, pipeline=None, metadata=None):
    """
    Predict churn for a single customer.

    Parameters
    ----------
    customer : dict
        Dictionary of feature values, e.g.
        ``{'gender': 'Male', 'tenure': 12, 'MonthlyCharges': 65.0, ...}``
    pipeline : Pipeline, optional
        A pre-loaded pipeline. If None, loads from disk.
    metadata : dict, optional
        Pre-loaded metadata. If None, loads from disk.

    Returns
    -------
    dict with keys:
        prediction  : int (0 = No churn, 1 = Churn)
        probability : float (churn probability 0-1)
        risk_level  : str ("Low", "Medium", "High")
        risk_factors: list[str]
    """
    if pipeline is None or metadata is None:
        pipeline, metadata = load_model()

    # Build input DataFrame
    input_df = pd.DataFrame([customer])

    # Align columns to match training data
    expected_cols = metadata.get("feature_columns", [])
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_cols]

    # Predict
    prediction = int(pipeline.predict(input_df)[0])
    probabilities = pipeline.predict_proba(input_df)[0]
    churn_prob = float(probabilities[1])

    # Determine risk level
    if churn_prob < 0.3:
        risk_level = "Low"
    elif churn_prob < 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Identify risk factors based on known patterns
    risk_factors = _analyze_risk_factors(customer)

    return {
        "prediction": prediction,
        "probability": churn_prob,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "retain_probability": float(probabilities[0]),
    }


# ── Batch prediction ─────────────────────────────────────────────────────────

def predict_batch(df: pd.DataFrame, pipeline=None, metadata=None) -> pd.DataFrame:
    """
    Predict churn for a batch of customers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with one row per customer.
    pipeline : Pipeline, optional
    metadata : dict, optional

    Returns
    -------
    pd.DataFrame with added columns: Prediction, Churn_Probability, Risk_Level
    """
    if pipeline is None or metadata is None:
        pipeline, metadata = load_model()

    result = df.copy()

    # Align columns
    expected_cols = metadata.get("feature_columns", [])
    for col in expected_cols:
        if col not in result.columns:
            result[col] = 0
    aligned = result[expected_cols]

    predictions = pipeline.predict(aligned)
    probabilities = pipeline.predict_proba(aligned)[:, 1]

    result["Prediction"] = predictions
    result["Churn_Probability"] = probabilities
    result["Risk_Level"] = pd.cut(
        probabilities,
        bins=[-0.01, 0.3, 0.6, 1.01],
        labels=["Low", "Medium", "High"],
    )

    return result


# ── Risk factor analysis ─────────────────────────────────────────────────────

def _analyze_risk_factors(customer: dict) -> list[str]:
    """Identify human-readable risk factors from customer features."""
    factors = []

    contract = customer.get("Contract", "")
    if contract == "Month-to-month":
        factors.append("Month-to-month contract — highest churn risk")

    internet = customer.get("InternetService", "")
    if internet == "Fiber optic":
        factors.append("Fiber optic internet — correlated with higher churn")

    payment = customer.get("PaymentMethod", "")
    if payment == "Electronic check":
        factors.append("Electronic check payment — associated with higher churn")

    tenure = customer.get("tenure", 999)
    if isinstance(tenure, (int, float)) and tenure < 12:
        factors.append("Low tenure (<12 months) — new customers churn more")

    security = customer.get("OnlineSecurity", "")
    if security == "No":
        factors.append("No online security — lack of add-on services increases risk")

    techsupport = customer.get("TechSupport", "")
    if techsupport == "No":
        factors.append("No tech support — customers without support churn more")

    monthly = customer.get("MonthlyCharges", 0)
    if isinstance(monthly, (int, float)) and monthly > 70:
        factors.append("High monthly charges (>$70) — cost may be a factor")

    backup = customer.get("OnlineBackup", "")
    if backup == "No":
        factors.append("No online backup — fewer services linked to higher churn")

    return factors
