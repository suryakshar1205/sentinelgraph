"""
FastAPI REST API for SentinelGraph Risk Management Engine.
Provides real-time scoring, risk case retrieval, metrics, and investigation explanation endpoints.
"""

import sys
import os

# Ensure project root is in sys.path regardless of execution directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data.schemas import RiskCase, DefensiveAction, SeverityLevel
from cases.store import RiskCaseStore
from llm.explainer import RiskExplainer
from features.pipeline import FeaturePipeline
from models.baseline import TransactionBaselineModel
from graph.signals import GraphSignalScorer
from temporal.escalation import TemporalEscalationScorer
from fusion.risk import RiskFusionEngine

app = FastAPI(
    title="SentinelGraph API",
    description="AI-Powered Early-Warning Risk Management API for Coordinated Merchant Abuse",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

case_store = RiskCaseStore()
explainer = RiskExplainer()

# Lazy-loaded pipeline components
_baseline_model: Optional[TransactionBaselineModel] = None
_feature_pipeline = FeaturePipeline()
_graph_scorer = GraphSignalScorer()
_temporal_scorer = TemporalEscalationScorer()
_fusion_engine = RiskFusionEngine()


def get_model():
    global _baseline_model
    if _baseline_model is None:
        model_path = "models/artifacts/baseline_model.joblib"
        if os.path.exists(model_path):
            _baseline_model = TransactionBaselineModel.load(model_path)
    return _baseline_model


class ScoreRequest(BaseModel):
    transactions: List[Dict[str, Any]]


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "SentinelGraph Risk Engine",
        "active_cases": len(case_store.list_cases())
    }


@app.get("/cases", response_model=List[RiskCase])
def list_risk_cases():
    """Retrieve all detected risk cases ordered by severity."""
    return case_store.list_cases()


@app.get("/cases/{case_id}", response_model=RiskCase)
def get_risk_case(case_id: str):
    """Retrieve details and verifiable evidence for a single risk case."""
    case = case_store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
    return case


@app.get("/clusters/{cluster_id}")
def get_cluster_details(cluster_id: str):
    """Retrieve subgraph details for a cluster by matching case or ring ID."""
    cases = case_store.list_cases()
    matched = [c for c in cases if c.case_id == cluster_id or c.ring_id == cluster_id]
    if not matched:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")
    return matched[0]


@app.get("/metrics")
def get_evaluation_metrics():
    """Retrieve held-out evaluation and ablation results."""
    metrics_path = "evaluation/artifacts/ablation_results.json"
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Evaluation results not yet generated. Run evaluation pipeline first.")
    with open(metrics_path, "r") as f:
        return json.load(f)


@app.post("/explain/{case_id}")
def explain_risk_case(case_id: str):
    """Generate structured AI investigation summary strictly from case evidence."""
    case = case_store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
    return explainer.explain_case(case)


@app.post("/score")
def score_transactions(req: ScoreRequest):
    """Score a batch of incoming transactions using the complete behavioral + graph + temporal fusion pipeline."""
    if not req.transactions:
        return {"scores": []}
    
    bm = get_model()
    df = pd.DataFrame(req.transactions)
    
    # Fill defaults if missing
    import time
    if "timestamp_unix" not in df.columns:
        if "timestamp" in df.columns:
            df["timestamp_unix"] = pd.to_datetime(df["timestamp"]).astype(int) / 1e9
        else:
            df["timestamp_unix"] = time.time()
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp_unix"], unit="s")
    if "account_created_unix" not in df.columns:
        df["account_created_unix"] = df["timestamp_unix"] - 86400.0
    if "customer_id" not in df.columns:
        df["customer_id"] = [f"cust_{i}" for i in range(len(df))]
    if "merchant_id" not in df.columns:
        df["merchant_id"] = "merch_default"
    if "device_id" not in df.columns:
        df["device_id"] = [f"dev_{i}" for i in range(len(df))]
    if "card_id" not in df.columns:
        df["card_id"] = [f"card_{i}" for i in range(len(df))]
    if "ip_address" not in df.columns:
        df["ip_address"] = [f"192.168.1.{i%250}" for i in range(len(df))]
    if "amount" not in df.columns:
        df["amount"] = 1000.0

    beh_df, ent_df, tem_df = _feature_pipeline.extract_full_features(df)
    
    if bm is not None:
        beh_probs = bm.predict_proba(df)
    else:
        beh_probs = np.zeros(len(df))
        
    g_scores = _graph_scorer.compute_graph_scores_for_dataframe(ent_df)
    t_scores = _temporal_scorer.compute_escalation_scores_for_dataframe(tem_df)
    
    fused_df = _fusion_engine.fuse_dataframe(beh_probs, g_scores, t_scores)
    
    results = []
    for i in range(len(df)):
        results.append({
            "transaction_id": df.iloc[i].get("transaction_id", f"tx_{i}"),
            "behavioral_score": float(fused_df["behavioral_score"].iloc[i]),
            "graph_score": float(fused_df["graph_score"].iloc[i]),
            "temporal_score": float(fused_df["temporal_score"].iloc[i]),
            "final_risk_score": float(fused_df["final_risk_score"].iloc[i]),
            "confidence": float(fused_df["confidence"].iloc[i]),
            "severity": str(fused_df["severity"].iloc[i]),
            "recommended_action": str(fused_df["recommended_action"].iloc[i])
        })
        
    return {"scores": results}
