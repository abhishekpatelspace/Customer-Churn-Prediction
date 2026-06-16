"""
Customer Churn Prediction — Streamlit Dashboard
=================================================
A premium, interactive ML dashboard for Telco customer churn analysis.
Features: EDA visualizations, model training, evaluation, and REAL-TIME prediction.

Usage:
    streamlit run app.py
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)

warnings.filterwarnings("ignore")

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_raw_data, clean_data, get_feature_target_split
from src.feature_engineering import identify_column_types, build_preprocessor
from src.predict import load_model, predict_single


# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Dynamic Theme & Styling ──────────────────────────────────────────────────

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

is_light = (st.session_state.theme == "Light")

if is_light:
    css_vars = """
    :root {
        --accent: #4F46E5;
        --accent-light: #6C63FF;
        --accent-dark: #3730A3;
        --success: #059669;
        --danger: #DC2626;
        --warning: #D97706;
        --bg-dark: #F8FAFC;
        --bg-card: #FFFFFF;
        --bg-card-hover: #F1F5F9;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --border: #E2E8F0;
    }
    .stApp {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 50%, #E2E8F0 100%) !important;
        color: var(--text-primary) !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8FAFC 0%, #E2E8F0 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    """
    TEXT_COLOR = "#0F172A"
    TEXT_SECONDARY = "#475569"
    PLOT_BG = "rgba(241, 245, 249, 0.6)"
    GRID_COLOR = "rgba(226, 232, 240, 0.8)"
    GAUGE_BG = "rgba(241, 245, 249, 0.8)"
else:
    css_vars = """
    :root {
        --accent: #6C63FF;
        --accent-light: #8B83FF;
        --accent-dark: #4F46E5;
        --success: #10B981;
        --danger: #EF4444;
        --warning: #F59E0B;
        --bg-dark: #0F0F1A;
        --bg-card: #1A1A2E;
        --bg-card-hover: #22223A;
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --border: #2D2D44;
    }
    .stApp {
        background: linear-gradient(135deg, #0F0F1A 0%, #1A1A2E 50%, #16213E 100%) !important;
        color: var(--text-primary) !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1A2E 0%, #0F0F1A 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    """
    TEXT_COLOR = "#F1F5F9"
    TEXT_SECONDARY = "#94A3B8"
    PLOT_BG = "rgba(26,26,46,0.6)"
    GRID_COLOR = "rgba(45,45,68,0.5)"
    GAUGE_BG = "rgba(26, 26, 46, 0.8)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

{css_vars}

.stApp {{
    font-family: 'Inter', sans-serif;
}}

section[data-testid="stSidebar"] .stRadio label p {{
    color: var(--text-primary) !important;
    font-weight: 500;
}}

h1 {{
    background: linear-gradient(135deg, var(--accent), var(--accent-light), #EC4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}}

h2, h3, h4, h5, h6, p, span, li {{
    color: var(--text-primary) !important;
}}

label[data-testid="stWidgetLabel"] p {{
    color: var(--text-primary) !important;
    font-weight: 500;
}}

div[data-testid="stMetric"] {{
    background: linear-gradient(135deg, var(--bg-card), var(--bg-card-hover)) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 4px 24px rgba(108, 99, 255, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}

div[data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.12);
}}

div[data-testid="stMetric"] label {{
    color: var(--text-secondary) !important;
    font-weight: 500; font-size: 0.85rem;
    text-transform: uppercase; letter-spacing: 0.5px;
}}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: var(--text-primary) !important;
    font-weight: 700; font-size: 1.8rem !important;
}}

.stTabs [data-baseweb="tab-list"] {{ gap: 8px; background: transparent; }}

.stTabs [data-baseweb="tab"] {{
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 12px; color: var(--text-secondary) !important;
    padding: 10px 24px; font-weight: 500; transition: all 0.2s ease;
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, var(--accent), var(--accent-light)) !important;
    color: white !important; border-color: var(--accent) !important;
}}

.stButton > button {{
    background: linear-gradient(135deg, var(--accent), var(--accent-dark)) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 12px 32px !important; font-weight: 600 !important; font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(108, 99, 255, 0.2) !important;
}}

.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(108, 99, 255, 0.3) !important;
    color: white !important;
}}

/* Selectbox, Number Input, Text Input and Buttons Styling */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stTextInput"] input {{
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}}

/* Custom Number Input wrapper, input field, and buttons (+ / -) */
div[data-testid="stNumberInput"] {{
    background-color: transparent !important;
}}
div[data-testid="stNumberInput"] > div {{
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}}
div[data-testid="stNumberInput"] > div div,
div[data-testid="stNumberInput"] input {{
    background-color: transparent !important;
    color: var(--text-primary) !important;
    border: none !important;
}}
div[data-testid="stNumberInput"] button {{
    background-color: var(--bg-card-hover) !important;
    color: var(--text-primary) !important;
    border: none !important;
    border-radius: 6px !important;
    margin: 2px !important;
}}
div[data-testid="stNumberInput"] button:hover {{
    background-color: var(--accent) !important;
    color: white !important;
}}

/* Ensure selectbox dropdown arrow is visible */
div[data-testid="stSelectbox"] svg {{
    fill: var(--text-primary) !important;
}}

/* Ensure selected option text is visible inside input */
div[data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] p {{
    color: var(--text-primary) !important;
}}

/* Selectbox Dropdown Menu Popovers styling */
div[data-baseweb="popover"],
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] [data-baseweb="option"],
div[data-testid="stVirtualDropdown"],
ul[data-testid="stSelectboxVirtualDropdown"] {{
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
}}

ul[data-testid="stSelectboxVirtualDropdown"] li[role="option"],
ul[data-testid="stSelectboxVirtualDropdown"] [data-baseweb="option"] {{
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
}}

ul[data-testid="stSelectboxVirtualDropdown"] li[role="option"]:hover,
ul[data-testid="stSelectboxVirtualDropdown"] [data-baseweb="option"]:hover,
ul[data-testid="stSelectboxVirtualDropdown"] li[role="option"][aria-selected="true"],
ul[data-testid="stSelectboxVirtualDropdown"] [data-baseweb="option"][aria-selected="true"] {{
    background-color: var(--bg-card-hover) !important;
    color: var(--text-primary) !important;
}}

.streamlit-expanderHeader {{
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}}

.streamlit-expanderHeader p {{
    color: var(--text-primary) !important; font-weight: 600;
}}

div[data-testid="stAlert"] {{ border-radius: 12px; border: none; }}
.stDataFrame {{ border-radius: 12px; overflow: hidden; }}

.hero-divider {{
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    margin: 16px 0 24px; border-radius: 2px;
}}

.prediction-card {{
    padding: 32px; border-radius: 20px; text-align: center;
    font-size: 1.4rem; font-weight: 700; margin: 16px 0;
    animation: fadeIn 0.5s ease;
}}

.churn-yes {{
    background: linear-gradient(135deg, #7F1D1D, #991B1B) !important;
    border: 1px solid #DC2626 !important; color: #FCA5A5 !important;
    box-shadow: 0 8px 32px rgba(239, 68, 68, 0.2) !important;
}}

.churn-no {{
    background: linear-gradient(135deg, #064E3B, #065F46) !important;
    border: 1px solid #10B981 !important; color: #6EE7B7 !important;
    box-shadow: 0 8px 32px rgba(16, 185, 129, 0.2) !important;
}}

.churn-medium {{
    background: linear-gradient(135deg, #78350F, #92400E) !important;
    border: 1px solid #F59E0B !important; color: #FDE68A !important;
    box-shadow: 0 8px 32px rgba(245, 158, 11, 0.2) !important;
}}

.status-badge {{
    display: inline-block; padding: 6px 16px; border-radius: 20px;
    font-size: 0.85rem; font-weight: 600; letter-spacing: 0.5px;
}}

.badge-ready {{
    background: rgba(16, 185, 129, 0.15) !important; color: #6EE7B7 !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
}}

.badge-missing {{
    background: rgba(239, 68, 68, 0.15) !important; color: #FCA5A5 !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
}}

@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.kpi-container {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin: 16px 0 24px;
}}
.kpi-card {{
    flex: 1 1 calc(20% - 16px);
    min-width: 150px;
    background: linear-gradient(135deg, var(--bg-card), var(--bg-card-hover)) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px;
    padding: 16px 20px;
    box-shadow: 0 4px 24px rgba(108, 99, 255, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
.kpi-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.12);
}}
.kpi-label {{
    color: var(--text-secondary) !important;
    font-weight: 500;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.kpi-value {{
    color: var(--text-primary) !important;
    font-weight: 700;
    font-size: 1.8rem;
    margin-top: 4px;
}}
.kpi-delta {{
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 2px;
}}
.kpi-delta.up {{
    color: var(--danger) !important;
}}
.kpi-delta.down {{
    color: var(--success) !important;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


# ── Plotly Theme ─────────────────────────────────────────────────────────────

_PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=PLOT_BG,
    font=dict(family="Inter", color=TEXT_COLOR),
    margin=dict(l=40, r=40, t=50, b=40),
)

_DEFAULT_AXIS = dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)


def themed_layout(**overrides):
    """Return a Plotly layout dict with themed defaults, safely merged with overrides."""
    layout = dict(_PLOTLY_BASE)
    layout["xaxis"] = {**_DEFAULT_AXIS, **overrides.pop("xaxis", {})}
    layout["yaxis"] = {**_DEFAULT_AXIS, **overrides.pop("yaxis", {})}
    layout.update(overrides)
    return layout


COLORS = [
    "#6C63FF", "#EC4899", "#10B981", "#F59E0B", "#3B82F6",
    "#8B5CF6", "#EF4444", "#14B8A6", "#F97316", "#06B6D4",
]


# ── Data Loading (cached) ───────────────────────────────────────────────────

@st.cache_data
def load_dashboard_data():
    """Load and clean the dataset for EDA pages."""
    csv_path = os.path.join(PROJECT_ROOT, "data", "Telco-Customer-Churn.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, "Telco-Customer-Churn.csv")
    df = load_raw_data(csv_path)
    df = clean_data(df)
    return df


@st.cache_resource
def _load_model_cached():
    """Load model from disk. If missing, raises FileNotFoundError so Streamlit doesn't cache the failure."""
    return load_model()

def load_trained_model():
    """Load the pre-trained model pipeline and metadata."""
    try:
        pipeline, metadata = _load_model_cached()
        return pipeline, metadata, True
    except Exception as e:
        if not isinstance(e, FileNotFoundError):
            import traceback
            st.sidebar.error(f"Error loading model: {e}")
            st.sidebar.code(traceback.format_exc())
        return None, None, False


@st.cache_resource
def train_models_live(df_input):
    """Fallback: train models on-the-fly if no saved model exists."""
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier

    df = df_input.copy()
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    cat_cols, num_cols = identify_column_types(X)
    preprocessor = build_preprocessor(cat_cols, num_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=42, class_weight="balanced"
        )),
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

    def calc_metrics(y_true, y_pred, y_prob):
        return {
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred), 4),
            "Recall": round(recall_score(y_true, y_pred), 4),
            "F1 Score": round(f1_score(y_true, y_pred), 4),
            "ROC-AUC": round(roc_auc_score(y_true, y_prob), 4),
        }

    rf_metrics = calc_metrics(y_test, y_pred_rf, y_prob_rf)

    xgb_pipeline = None
    xgb_metrics = None
    y_pred_xgb = y_prob_xgb = None
    xgb_available = False

    try:
        from xgboost import XGBClassifier
        neg = int(y_train.value_counts()[0])
        pos = int(y_train.value_counts()[1])

        xgb_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=neg / pos,
                eval_metric="logloss", random_state=42,
            )),
        ])
        xgb_pipeline.fit(X_train, y_train)
        y_pred_xgb = xgb_pipeline.predict(X_test)
        y_prob_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]
        xgb_metrics = calc_metrics(y_test, y_pred_xgb, y_prob_xgb)
        xgb_available = True
    except ImportError:
        pass

    best_pipeline = xgb_pipeline if xgb_available else rf_pipeline
    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = best_pipeline.named_steps["model"].feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names, "Importance": importances,
    }).sort_values("Importance", ascending=False)

    return {
        "rf_pipeline": rf_pipeline, "xgb_pipeline": xgb_pipeline,
        "xgb_available": xgb_available,
        "rf_metrics": rf_metrics, "xgb_metrics": xgb_metrics,
        "y_test": y_test,
        "y_pred_rf": y_pred_rf, "y_prob_rf": y_prob_rf,
        "y_pred_xgb": y_pred_xgb, "y_prob_xgb": y_prob_xgb,
        "importance_df": importance_df,
        "cat_cols": cat_cols, "num_cols": num_cols, "X_train": X_train,
    }


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📉 Churn Predictor")
    st.markdown('<div class="hero-divider"></div>', unsafe_allow_html=True)

    # Theme toggle
    theme_toggle = st.toggle("☀️ Light Mode", value=(st.session_state.theme == "Light"))
    new_theme = "Light" if theme_toggle else "Dark"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Exploratory Analysis", "🤖 Model Performance", "🔮 Predict Churn"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    _, _, model_ready = load_trained_model()
    if model_ready:
        st.markdown('<span class="status-badge badge-ready">Model Loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-missing">No Saved Model</span>', unsafe_allow_html=True)
        st.caption("Run `python -m src.train` to train.")

    st.markdown("---")
    st.caption("Built with Streamlit | scikit-learn | XGBoost")
    st.caption("Dataset: Telco Customer Churn")


# ── Load Data ────────────────────────────────────────────────────────────────

df = load_dashboard_data()

if page == "🤖 Model Performance":
    with st.spinner("Preparing model metrics..."):
        results = train_models_live(df)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.markdown("# Customer Churn Prediction")
    st.markdown('<div class="hero-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        "Analyze customer behaviour, identify churn drivers, and predict which "
        "customers are likely to leave -- powered by **Random Forest** and **XGBoost**."
    )

    total = len(df)
    churned = int((df["Churn"] == "Yes").sum())
    retained = total - churned
    churn_rate = churned / total * 100
    avg_tenure = df["tenure"].mean()
    avg_monthly = df["MonthlyCharges"].mean()

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Total Customers</div>
            <div class="kpi-value">{total:,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Churned Customers</div>
            <div class="kpi-value">{churned:,}</div>
            <div class="kpi-delta up">▲ {churn_rate:.1f}% Churn Rate</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Retained Customers</div>
            <div class="kpi-value">{retained:,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Avg Tenure</div>
            <div class="kpi-value">{avg_tenure:.0f} mo</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Avg Monthly Charges</div>
            <div class="kpi-value">${avg_monthly:.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:
        churn_counts = df["Churn"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=churn_counts.index, values=churn_counts.values,
            hole=0.55, marker=dict(colors=["#10B981", "#EF4444"]),
            textinfo="label+percent",
            textfont=dict(size=14, family="Inter"),
        )])
        fig.update_layout(
            title="Churn Distribution", showlegend=False,
            **themed_layout(height=380),
            annotations=[dict(
                text=f"<b>{churn_rate:.1f}%</b><br>churn",
                x=0.5, y=0.5, font_size=18, showarrow=False,
                font=dict(color=TEXT_COLOR, family="Inter")
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rev_by_churn = df.groupby("Churn")["MonthlyCharges"].sum().reset_index()
        fig = go.Figure(data=[go.Bar(
            x=rev_by_churn["Churn"], y=rev_by_churn["MonthlyCharges"],
            marker=dict(color=["#10B981", "#EF4444"], cornerradius=8),
            text=[f"${v:,.0f}" for v in rev_by_churn["MonthlyCharges"]],
            textposition="outside",
            textfont=dict(size=13, family="Inter", color=TEXT_COLOR),
        )])
        fig.update_layout(
            title="Monthly Revenue by Churn Status",
            xaxis_title="Churn Status", yaxis_title="Total Monthly Charges ($)",
            **themed_layout(height=380),
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        contract_data = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(x=contract_data.index.astype(str), y=contract_data["No"], name="Retained", marker_color="#10B981"))
        fig.add_trace(go.Bar(x=contract_data.index.astype(str), y=contract_data["Yes"], name="Churned", marker_color="#EF4444"))
        fig.update_layout(
            barmode="stack", title="Contract Type -- Churn Rate (%)", yaxis_title="Percentage (%)",
            **themed_layout(height=380),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        internet_data = pd.crosstab(df["InternetService"], df["Churn"], normalize="index") * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(x=internet_data.index.astype(str), y=internet_data["No"], name="Retained", marker_color="#10B981"))
        fig.add_trace(go.Bar(x=internet_data.index.astype(str), y=internet_data["Yes"], name="Churned", marker_color="#EF4444"))
        fig.update_layout(
            barmode="stack", title="Internet Service -- Churn Rate (%)", yaxis_title="Percentage (%)",
            **themed_layout(height=380),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Preview Raw Data"):
        st.dataframe(df.head(100), use_container_width=True, height=350)
        st.caption(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Exploratory Analysis":
    st.markdown("# Exploratory Data Analysis")
    st.markdown('<div class="hero-divider"></div>', unsafe_allow_html=True)

    tabs = st.tabs(["Distributions", "Churn vs Features", "Correlation", "Statistical Tests"])

    # Tab 1: Distributions
    with tabs[0]:
        st.subheader("Numerical Feature Distributions")
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Tenure", "Monthly Charges", "Total Charges"))
        for i, col in enumerate(["tenure", "MonthlyCharges", "TotalCharges"]):
            fig.add_trace(go.Histogram(x=df[col], nbinsx=35, name=col, marker_color=COLORS[i], opacity=0.85), row=1, col=i + 1)
        fig.update_layout(**themed_layout(height=380, showlegend=False))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Categorical Feature Breakdown")
        cat_feature = st.selectbox("Select feature", [
            "gender", "SeniorCitizen", "Partner", "Dependents",
            "PhoneService", "InternetService", "Contract", "PaperlessBilling", "PaymentMethod"
        ])
        counts = df[cat_feature].value_counts()
        fig = go.Figure(data=[go.Bar(
            x=counts.index.astype(str), y=counts.values,
            marker=dict(color=COLORS[:len(counts)], cornerradius=8),
            text=counts.values, textposition="outside",
            textfont=dict(color=TEXT_COLOR, family="Inter"),
        )])
        fig.update_layout(
            title=f"Distribution of {cat_feature}",
            xaxis_title=cat_feature, yaxis_title="Count",
            **themed_layout(height=380),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Churn vs Features
    with tabs[1]:
        st.subheader("Numerical Features vs Churn")
        col_a, col_b, col_c = st.columns(3)
        for col_widget, feature in zip([col_a, col_b, col_c], ["tenure", "MonthlyCharges", "TotalCharges"]):
            with col_widget:
                fig = go.Figure()
                for label, clr in [("No", "#10B981"), ("Yes", "#EF4444")]:
                    subset = df[df["Churn"] == label][feature]
                    fig.add_trace(go.Box(y=subset, name=label, marker_color=clr, boxmean="sd"))
                fig.update_layout(title=f"{feature} vs Churn", **themed_layout(height=350, showlegend=False))
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Categorical Features vs Churn")
        cat_churn_feature = st.selectbox("Select categorical feature", [
            "InternetService", "Contract", "PaymentMethod", "Partner", "Dependents",
            "TechSupport", "OnlineSecurity", "PaperlessBilling", "StreamingTV", "StreamingMovies",
        ], key="cat_churn")

        churn_data = pd.crosstab(df[cat_churn_feature], df["Churn"], normalize="index") * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(x=churn_data.index.astype(str), y=churn_data["No"], name="Retained", marker_color="#10B981"))
        fig.add_trace(go.Bar(x=churn_data.index.astype(str), y=churn_data["Yes"], name="Churned", marker_color="#EF4444"))
        fig.update_layout(
            barmode="stack", title=f"{cat_churn_feature} -- Churn Rate (%)", yaxis_title="Percentage (%)",
            **themed_layout(height=400),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Correlation
    with tabs[2]:
        st.subheader("Correlation Heatmap")
        df_corr = df.copy()
        df_corr["Churn_num"] = df_corr["Churn"].map({"Yes": 1, "No": 0})
        corr_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn_num"]
        corr = df_corr[corr_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu_r", zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont=dict(size=14),
        ))
        fig.update_layout(title="Correlation Matrix", **themed_layout(height=450))
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Key Insights**: Tenure has a negative correlation with Churn "
            "(longer customers stay more). Monthly Charges has a positive correlation "
            "(higher cost -> more churn risk)."
        )

    # Tab 4: Statistical Tests
    with tabs[3]:
        from scipy.stats import chi2_contingency, ttest_ind

        st.subheader("Chi-Square Test (Categorical vs Churn)")
        chi2_results = []
        cat_cols_test = [
            "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod",
        ]
        for col in cat_cols_test:
            contingency = pd.crosstab(df[col], df["Churn"])
            chi2, p, dof, exp = chi2_contingency(contingency)
            chi2_results.append({
                "Feature": col, "Chi2 Statistic": round(chi2, 2),
                "p-value": f"{p:.2e}",
                "Significant (p<0.05)": "Yes" if p < 0.05 else "No",
            })
        st.dataframe(pd.DataFrame(chi2_results), use_container_width=True, hide_index=True)

        st.subheader("T-Test (Numerical vs Churn)")
        churn_yes = df[df["Churn"] == "Yes"]
        churn_no = df[df["Churn"] == "No"]
        ttest_results = []
        for feat in ["tenure", "MonthlyCharges", "TotalCharges"]:
            stat, p = ttest_ind(churn_yes[feat], churn_no[feat])
            ttest_results.append({
                "Feature": feat, "T-Statistic": round(stat, 4),
                "p-value": f"{p:.2e}",
                "Significant (p<0.05)": "Yes" if p < 0.05 else "No",
            })
        st.dataframe(pd.DataFrame(ttest_results), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Model Performance":
    st.markdown("# Model Performance")
    st.markdown('<div class="hero-divider"></div>', unsafe_allow_html=True)

    st.subheader("Metrics Comparison")

    models = {"Random Forest": results["rf_metrics"]}
    if results["xgb_available"]:
        models["XGBoost"] = results["xgb_metrics"]

    cols = st.columns(len(models))
    for i, (name, metrics) in enumerate(models.items()):
        with cols[i]:
            st.markdown(f"#### {name}")
            for metric_name, value in metrics.items():
                st.metric(metric_name, f"{value:.4f}")

    if results["xgb_available"]:
        metric_names = list(results["rf_metrics"].keys())
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=metric_names, y=list(results["rf_metrics"].values()),
            name="Random Forest", marker_color="#6C63FF",
            text=[f"{v:.3f}" for v in results["rf_metrics"].values()],
            textposition="outside", textfont=dict(color=TEXT_COLOR),
        ))
        fig.add_trace(go.Bar(
            x=metric_names, y=list(results["xgb_metrics"].values()),
            name="XGBoost", marker_color="#EC4899",
            text=[f"{v:.3f}" for v in results["xgb_metrics"].values()],
            textposition="outside", textfont=dict(color=TEXT_COLOR),
        ))
        fig.update_layout(
            barmode="group", title="Model Metrics Comparison",
            **themed_layout(height=420, yaxis=dict(range=[0, 1.1])),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrices
    st.subheader("Confusion Matrices")
    cm_cols = st.columns(2 if results["xgb_available"] else 1)

    def plot_confusion(y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=["Predicted No", "Predicted Yes"], y=["Actual No", "Actual Yes"],
            colorscale=[[0, "#F1F5F9" if is_light else "#1A1A2E"], [1, "#4F46E5" if is_light else "#6C63FF"]],
            text=cm, texttemplate="%{text}",
            textfont=dict(size=20, color=TEXT_COLOR), showscale=False,
        ))
        fig.update_layout(title=title, **themed_layout(height=370))
        return fig

    with cm_cols[0]:
        st.plotly_chart(plot_confusion(results["y_test"], results["y_pred_rf"], "Random Forest"), use_container_width=True)

    if results["xgb_available"]:
        with cm_cols[1]:
            st.plotly_chart(plot_confusion(results["y_test"], results["y_pred_xgb"], "XGBoost"), use_container_width=True)

    # ROC Curve
    st.subheader("ROC Curve")
    fig = go.Figure()
    fpr_rf, tpr_rf, _ = roc_curve(results["y_test"], results["y_prob_rf"])
    fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode="lines", name=f"RF (AUC={results['rf_metrics']['ROC-AUC']:.3f})", line=dict(color="#6C63FF", width=3)))

    if results["xgb_available"]:
        fpr_xgb, tpr_xgb, _ = roc_curve(results["y_test"], results["y_prob_xgb"])
        fig.add_trace(go.Scatter(x=fpr_xgb, y=tpr_xgb, mode="lines", name=f"XGB (AUC={results['xgb_metrics']['ROC-AUC']:.3f})", line=dict(color="#EC4899", width=3)))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color=TEXT_SECONDARY, width=1, dash="dash")))
    fig.update_layout(
        title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        **themed_layout(height=450),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    st.subheader("Top 15 Feature Importances")
    top15 = results["importance_df"].head(15)
    fig = go.Figure(data=[go.Bar(
        x=top15["Importance"], y=top15["Feature"], orientation="h",
        marker=dict(
            color=top15["Importance"],
            colorscale=[[0, "#6C63FF"], [0.5, "#A78BFA"], [1, "#EC4899"]],
            cornerradius=6,
        ),
        text=[f"{v:.4f}" for v in top15["Importance"]],
        textposition="outside", textfont=dict(color=TEXT_COLOR, size=11),
    )])
    fig.update_layout(
        title="Feature Importance (Best Model)", xaxis_title="Importance",
        **themed_layout(height=520, yaxis=dict(autorange="reversed")),
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT CHURN (Real-Time)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Predict Churn":
    st.markdown("# Real-Time Churn Prediction")
    st.markdown('<div class="hero-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        "Enter customer details below and get an **instant prediction** on whether "
        "they are likely to churn -- powered by a pre-trained ML model."
    )

    pipeline, metadata, model_ready = load_trained_model()

    if not model_ready:
        st.warning(
            "**No pre-trained model found.** Training a model on-the-fly for this session.\n\n"
            "For faster startup, run `python -m src.train` in your terminal first."
        )
        with st.spinner("Training model..."):
            fallback_results = train_models_live(df)
            pipeline = (
                fallback_results["xgb_pipeline"]
                if fallback_results["xgb_available"]
                else fallback_results["rf_pipeline"]
            )
            metadata = {
                "feature_columns": fallback_results["X_train"].columns.tolist(),
                "model_name": "XGBoost" if fallback_results["xgb_available"] else "Random Forest",
            }
            model_ready = True

    model_name = metadata.get("model_name", "ML Model")
    st.info(f"Using **{model_name}** for predictions")

    # Main container layout (live updates on change)
    st.markdown("### Customer Information")
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with r1c2:
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with r1c3:
        partner = st.selectbox("Partner", ["Yes", "No"])
    with r1c4:
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    st.markdown("### Service Details")
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        tenure = st.number_input("Tenure (months)", 0, 72, 12)
    with r2c2:
        monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    with r2c3:
        total = st.number_input("Total Charges ($)", 0.0, 9000.0, 800.0, step=10.0)

    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
    with r3c2:
        multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    with r3c3:
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

    st.markdown("### Additional Services")
    r4c1, r4c2, r4c3 = st.columns(3)
    with r4c1:
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    with r4c2:
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    with r4c3:
        protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

    r5c1, r5c2, r5c3 = st.columns(3)
    with r5c1:
        techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    with r5c2:
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    with r5c3:
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.markdown("### Billing")
    r6c1, r6c2, r6c3 = st.columns(3)
    with r6c1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with r6c2:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with r6c3:
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    st.markdown("---")

    if True:
        customer_data = {
            "gender": gender, "SeniorCitizen": senior,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": techsupport,
            "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }

        result = predict_single(customer_data, pipeline=pipeline, metadata=metadata)
        prediction = result["prediction"]
        churn_prob = result["probability"]
        risk_level = result["risk_level"]
        risk_factors = result["risk_factors"]

        st.markdown("")

        if prediction == 1 and risk_level == "High":
            st.markdown(
                '<div class="prediction-card churn-yes">'
                'HIGH CHURN RISK<br>'
                f'<span style="font-size:2.2rem">{churn_prob * 100:.1f}%</span> '
                'probability of leaving</div>',
                unsafe_allow_html=True
            )
        elif prediction == 1:
            st.markdown(
                '<div class="prediction-card churn-medium">'
                'MODERATE CHURN RISK<br>'
                f'<span style="font-size:2.2rem">{churn_prob * 100:.1f}%</span> '
                'probability of leaving</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="prediction-card churn-no">'
                'LOW CHURN RISK<br>'
                f'<span style="font-size:2.2rem">{result["retain_probability"] * 100:.1f}%</span> '
                'probability of staying</div>',
                unsafe_allow_html=True
            )

        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob * 100,
            domain=dict(x=[0, 1], y=[0, 1]),
            title=dict(text="Churn Probability", font=dict(size=18, color=TEXT_COLOR)),
            number=dict(suffix="%", font=dict(size=36, color=TEXT_COLOR)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=TEXT_SECONDARY),
                bar=dict(color="#6C63FF"),
                bgcolor=GAUGE_BG,
                steps=[
                    dict(range=[0, 30], color="rgba(16,185,129,0.3)"),
                    dict(range=[30, 60], color="rgba(245,158,11,0.3)"),
                    dict(range=[60, 100], color="rgba(239,68,68,0.3)"),
                ],
                threshold=dict(line=dict(color="#EF4444", width=3), thickness=0.8, value=50),
            ),
        ))
        fig.update_layout(**themed_layout(height=350))
        st.plotly_chart(fig, use_container_width=True)

        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("Stay Probability", f"{result['retain_probability'] * 100:.1f}%")
        with prob_col2:
            st.metric("Churn Probability", f"{churn_prob * 100:.1f}%")

        with st.expander("Risk Factor Analysis", expanded=True):
            if not risk_factors:
                st.success("No major risk factors identified. Customer profile looks stable!")
            else:
                for factor in risk_factors:
                    st.markdown(f"- **{factor}**")

                st.markdown("---")
                st.markdown("##### Retention Suggestions")
                if any("Month-to-month" in f for f in risk_factors):
                    st.markdown("-> Offer a discounted annual contract to lock in the customer")
                if any("Electronic check" in f for f in risk_factors):
                    st.markdown("-> Incentivize switching to automatic payment methods")
                if any("tech support" in f.lower() for f in risk_factors):
                    st.markdown("-> Offer a free tech support trial period")
                if any("security" in f.lower() for f in risk_factors):
                    st.markdown("-> Bundle online security as a free add-on for 3 months")
                if any("monthly charges" in f.lower() for f in risk_factors):
                    st.markdown("-> Review pricing or offer a loyalty discount")
