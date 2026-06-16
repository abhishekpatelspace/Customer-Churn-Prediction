# 📉 Customer Churn Prediction Pipeline

A production-grade machine learning pipeline for predicting Telco customer churn in **real-time**, with a premium interactive Streamlit dashboard.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python -m src.train
```
This trains Random Forest & XGBoost models, auto-selects the best one by ROC-AUC, and saves it to `models/`.

### 3. Launch the Dashboard
```bash
streamlit run app.py
```
Navigate to the **🔮 Predict Churn** page to make real-time predictions.

## 📁 Project Structure

```
├── data/                          # Dataset
│   └── Telco-Customer-Churn.csv
├── models/                        # Saved trained models
│   ├── churn_model.joblib
│   └── model_metadata.json
├── src/                           # Source modules
│   ├── data_loader.py             # Data loading & cleaning
│   ├── feature_engineering.py     # Preprocessing & feature engineering
│   ├── train.py                   # Training pipeline
│   └── predict.py                 # Prediction API
├── notebooks/                     # Original Colab notebook (reference)
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Dependencies
└── README.md
```

## 🧠 Models

| Model | Description |
|-------|-------------|
| **Random Forest** | 300 trees, max_depth=10, balanced class weights |
| **XGBoost** | 300 rounds, lr=0.05, scale_pos_weight for imbalance |

The best model (by ROC-AUC) is automatically selected and saved.

## 📊 Dashboard Pages

- **🏠 Overview** — KPIs, churn distribution, revenue impact
- **📊 Exploratory Analysis** — Distributions, correlations, statistical tests
- **🤖 Model Performance** — Metrics comparison, confusion matrices, ROC curves
- **🔮 Predict Churn** — Real-time single-customer prediction with risk analysis

## 📦 Dataset

[Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) — 7,043 customers with 21 features.
