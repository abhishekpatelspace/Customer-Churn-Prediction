"""
train.py — Full training pipeline for customer churn prediction.

Usage:
    python -m src.train
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_raw_data, clean_data, get_feature_target_split
from src.feature_engineering import identify_column_types, build_preprocessor


# ── Configuration ────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Telco-Customer-Churn.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ── Helpers ──────────────────────────────────────────────────────────────────

def _calc_metrics(y_true, y_pred, y_prob):
    """Compute a dict of standard classification metrics."""
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1 Score": round(f1_score(y_true, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_true, y_prob), 4),
    }


def _print_section(title: str):
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


# ── Main training function ───────────────────────────────────────────────────

def train():
    """Run the full training pipeline and save the best model."""
    start = time.time()

    _print_section("LOADING DATA")
    df_raw = load_raw_data(DATA_PATH)
    print(f"  Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")

    _print_section("CLEANING DATA")
    df = clean_data(df_raw)
    print(f"  After cleaning: {len(df)} rows, {len(df.columns)} columns")

    _print_section("PREPARING FEATURES & TARGET")
    X, y = get_feature_target_split(df)
    churn_rate = y.mean() * 100
    print(f"  Features: {X.shape[1]} | Target distribution: {churn_rate:.1f}% churn")

    cat_cols, num_cols = identify_column_types(X)
    print(f"  Categorical: {len(cat_cols)} cols -> {cat_cols}")
    print(f"  Numerical:   {len(num_cols)} cols -> {num_cols}")

    _print_section("TRAIN-TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Build the shared preprocessor
    preprocessor = build_preprocessor(cat_cols, num_cols)

    # ── Train Random Forest ──────────────────────────────────────────────
    _print_section("TRAINING: RANDOM FOREST")
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]
    rf_metrics = _calc_metrics(y_test, y_pred_rf, y_prob_rf)
    for k, v in rf_metrics.items():
        print(f"  {k:>12}: {v}")

    # ── Train XGBoost ────────────────────────────────────────────────────
    xgb_available = False
    xgb_pipeline = None
    xgb_metrics = None

    try:
        from xgboost import XGBClassifier

        _print_section("TRAINING: XGBOOST")

        neg = int(y_train.value_counts()[0])
        pos = int(y_train.value_counts()[1])
        scale_pos = neg / pos
        print(f"  scale_pos_weight = {scale_pos:.2f}")

        xgb_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
            )),
        ])
        xgb_pipeline.fit(X_train, y_train)
        y_pred_xgb = xgb_pipeline.predict(X_test)
        y_prob_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]
        xgb_metrics = _calc_metrics(y_test, y_pred_xgb, y_prob_xgb)
        for k, v in xgb_metrics.items():
            print(f"  {k:>12}: {v}")
        xgb_available = True

    except ImportError:
        print("\n  [!] XGBoost not installed -- using Random Forest only.")

    # ── Select best model ────────────────────────────────────────────────
    _print_section("MODEL SELECTION")
    if xgb_available and xgb_metrics["ROC-AUC"] >= rf_metrics["ROC-AUC"]:
        best_pipeline = xgb_pipeline
        best_name = "XGBoost"
        best_metrics = xgb_metrics
    else:
        best_pipeline = rf_pipeline
        best_name = "Random Forest"
        best_metrics = rf_metrics

    print(f"  [BEST] Best model: {best_name} (ROC-AUC = {best_metrics['ROC-AUC']})")

    # ── Feature importance ───────────────────────────────────────────────
    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    importances = best_pipeline.named_steps["model"].feature_importances_.tolist()
    importance_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print(f"\n  Top 10 features:")
    for fname, imp in importance_pairs[:10]:
        print(f"    {fname:>40}: {imp:.4f}")

    # ── Save model & metadata ────────────────────────────────────────────
    _print_section("SAVING MODEL")
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "churn_model.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"  Pipeline saved -> {model_path}")

    metadata = {
        "model_name": best_name,
        "metrics": best_metrics,
        "rf_metrics": rf_metrics,
        "xgb_metrics": xgb_metrics,
        "feature_columns": X_train.columns.tolist(),
        "categorical_columns": cat_cols,
        "numerical_columns": num_cols,
        "feature_importance": {fname: imp for fname, imp in importance_pairs[:30]},
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "churn_rate_pct": round(churn_rate, 2),
        "trained_at": datetime.now().isoformat(),
    }

    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved -> {meta_path}")

    elapsed = time.time() - start
    _print_section("DONE")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Run the dashboard with:  streamlit run app.py\n")

    return best_pipeline, metadata


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
