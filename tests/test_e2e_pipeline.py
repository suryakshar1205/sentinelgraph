"""
End-to-End Pipeline Integration Test for SentinelGraph.
Verifies complete flow from data -> features -> baseline -> graph -> temporal -> fusion -> cases -> evaluation.
"""

import pytest
import numpy as np
import pandas as pd
import datetime

from data.patterns import PatternGenerator
from features.pipeline import FeaturePipeline
from models.baseline import TransactionBaselineModel
from graph.builder import EntityGraphBuilder
from graph.communities import GraphCommunityDetector
from graph.signals import GraphSignalScorer
from temporal.escalation import TemporalEscalationScorer
from fusion.risk import RiskFusionEngine
from cases.builder import RiskCaseBuilder
from evaluation.metrics import EvaluationEngine
from evaluation.early_warning import EarlyWarningEvaluator


def test_complete_e2e_pipeline_mini():
    """Run full mini pipeline end-to-end."""
    rng = np.random.default_rng(20260827)
    start_t = datetime.datetime(2026, 8, 1, 0, 0, 0)
    pgen = PatternGenerator(rng, start_t)
    
    # 1. Generate mini stream (200 transactions: 150 legit, 50 ring)
    records = []
    for i in range(20):
        prof = pgen.generate_legitimate_user_profile(f"cust_legit_{i}")
        for j in range(7):
            t = start_t + datetime.timedelta(hours=j*2)
            records.append({
                "transaction_id": f"tx_legit_{i}_{j}",
                "timestamp": t.isoformat(),
                "timestamp_unix": t.timestamp(),
                "customer_id": prof["customer_id"],
                "account_created_at": prof["account_created_at"],
                "account_created_unix": prof["account_created_unix"],
                "device_id": prof["primary_device"],
                "ip_id": prof["primary_ip"],
                "payment_instrument_id": prof["primary_card"],
                "merchant_id": "merch_001",
                "amount": 250.0,
                "location": "Mumbai",
                "transaction_type": "payment",
                "label": "legitimate",
                "abuse_type": "none",
                "ring_id": None,
                "is_fraud": 0
            })
            
    # Add coordinated ring
    ring_records = pgen.create_coordinated_ring("RING_MINI_01", "pattern_a_shared_device", start_t + datetime.timedelta(days=1), ["merch_001"])
    for idx, r in enumerate(ring_records):
        r["transaction_id"] = f"tx_ring_{idx}"
    
    df = pd.DataFrame(records + ring_records).sort_values(by="timestamp_unix").reset_index(drop=True)
    
    # Split
    n_train = int(len(df) * 0.7)
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()
    
    # Feature extraction
    pipe = FeaturePipeline()
    beh_df, ent_df, tem_df = pipe.extract_full_features(test_df)
    
    # Train baseline
    bm = TransactionBaselineModel()
    bm.fit(train_df, test_df)
    beh_probs = bm.predict_proba(test_df)
    
    # Graph & temporal scoring
    g_scorer = GraphSignalScorer()
    g_scores = g_scorer.compute_graph_scores_for_dataframe(ent_df)
    
    t_scorer = TemporalEscalationScorer()
    t_scores = t_scorer.compute_escalation_scores_for_dataframe(tem_df)
    
    # Fusion
    fusion = RiskFusionEngine()
    fused_df = fusion.fuse_dataframe(beh_probs, g_scores, t_scores)
    
    # Cases
    case_builder = RiskCaseBuilder()
    case = case_builder.build_case_from_cluster(
        case_id="CASE_E2E_01",
        cluster_data={"num_accounts": 15, "num_devices": 2, "num_ips": 3, "num_cards": 2},
        cluster_txs_df=test_df[test_df["ring_id"].notna()],
        fused_row={
            "behavioral_score": float(fused_df["behavioral_score"].mean()),
            "graph_score": float(fused_df["graph_score"].mean()),
            "temporal_score": float(fused_df["temporal_score"].mean()),
            "final_risk_score": float(fused_df["final_risk_score"].max()),
            "confidence": 92.0,
            "severity": "HIGH",
            "recommended_action": "HOLD"
        }
    )
    assert case.case_id == "CASE_E2E_01"
    assert len(case.evidence_items) >= 2
    
    # Evaluation
    y_true = test_df["is_fraud"].values
    stage_c_preds = (fused_df["final_risk_score"].values >= bm.optimal_threshold * 100.0).astype(int)
    metrics = EvaluationEngine.compute_classification_metrics(y_true, stage_c_preds, fused_df["final_risk_score"].values / 100.0)
    
    assert metrics["precision"] > 0.0
    assert metrics["recall"] > 0.0
    assert metrics["f1"] > 0.0
