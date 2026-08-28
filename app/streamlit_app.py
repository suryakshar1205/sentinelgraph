"""
Main Streamlit Application Entrypoint for SentinelGraph.
Razorpay Buildathon 2026 — Track 02: AI Risk Manager.
"""

import sys
import os

# Ensure project root is in sys.path regardless of execution directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import json

from cases.store import RiskCaseStore
from models.baseline import TransactionBaselineModel

# Page configuration
st.set_page_config(
    page_title="SentinelGraph | AI Risk Manager",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Merchant-Risk Dark Theme CSS
st.markdown("""
<style>
    /* Dark Theme Core Tokens */
    .stApp {
        background-color: #0b0f19;
        color: #f0f6fc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }
    
    /* Metrics Card Styling */
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #f0f6fc !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    
    /* Card Container */
    .risk-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    /* Button Styling */
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    
    /* Dataframe Header */
    [data-testid="stDataFrame"] {
        border: 1px solid #30363d;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    """Load case store and baseline model once."""
    case_store = RiskCaseStore()
    
    baseline_model = None
    model_path = "models/artifacts/baseline_model.joblib"
    if os.path.exists(model_path):
        try:
            baseline_model = TransactionBaselineModel.load(model_path)
        except Exception as e:
            print(f"Warning loading model: {e}")
            
    return case_store, baseline_model


@st.cache_data
def load_datasets():
    """Load sample datasets for rendering."""
    test_df = None
    tx_df = None
    
    if os.path.exists("data/generated/test_frozen.parquet"):
        try:
            test_df = pd.read_parquet("data/generated/test_frozen.parquet")
        except Exception:
            pass
            
    if os.path.exists("data/generated/transactions.parquet"):
        try:
            # Read first 50,000 for fast memory footprint in UI
            tx_df = pd.read_parquet("data/generated/transactions.parquet").head(50000)
        except Exception:
            pass
            
    return test_df, tx_df


def main():
    case_store, baseline_model = load_resources()
    test_df, tx_df = load_datasets()

    # Sidebar Navigation
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <span style="font-size: 32px;">🛡️</span>
            <div>
                <h3 style="margin: 0; color: #f0f6fc; font-weight: 700; font-size: 20px;">SentinelGraph</h3>
                <span style="font-size: 11px; color: #00d2ff; letter-spacing: 0.5px; font-weight: 600;">TRACK 02: AI RISK MANAGER</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "1. Executive Dashboard",
            "2. Live Replay Studio",
            "3. Risk Case Detail",
            "4. Network Graph Explorer",
            "5. Evaluation & Ablation",
            "6. Methodology & Architecture"
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Razorpay Buildathon 2026\nCoordinated Abuse Early-Warning System")
    st.sidebar.caption("🔒 **Safety:** Strictly Defense-Only")

    # Routing
    if page == "1. Executive Dashboard":
        from app.dashboard import render_dashboard
        render_dashboard(case_store, tx_df)
    elif page == "2. Live Replay Studio":
        from app.replay import render_replay
        render_replay(test_df, baseline_model)
    elif page == "3. Risk Case Detail":
        from app.risk_case import render_risk_case
        render_risk_case(case_store)
    elif page == "4. Network Graph Explorer":
        from app.graph_view import render_graph_explorer
        render_graph_explorer(tx_df)
    elif page == "5. Evaluation & Ablation":
        from app.evaluation_view import render_evaluation_view
        render_evaluation_view()
    elif page == "6. Methodology & Architecture":
        from app.methodology import render_methodology
        render_methodology()


if __name__ == "__main__":
    main()
