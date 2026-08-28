"""
Hardening & Audit Unit Tests for SentinelGraph.
Tests point-in-time safety, deterministic ring activation, early warning lead times, evidence traceability, and guardrails.
"""

import pytest
import numpy as np
import pandas as pd
from features.temporal import TemporalFeatureExtractor
from temporal.escalation import TemporalEscalationScorer
from fusion.risk import RiskFusionEngine
from evaluation.early_warning import EarlyWarningEvaluator
from cases.builder import RiskCaseBuilder
from llm.explainer import RiskExplainer
from data.schemas import SeverityLevel, DefensiveAction


def test_temporal_point_in_time_safety():
    """Verify temporal arrival velocity only counts past transactions <= t."""
    extractor = TemporalFeatureExtractor()
    data = [
        {
            "timestamp_unix": 1000.0,
            "account_created_unix": 900.0,
            "device_id": "dev_A",
            "ip_id": "ip_A",
            "customer_id": "cust_1"
        },
        {
            "timestamp_unix": 1500.0,
            "account_created_unix": 900.0,
            "device_id": "dev_A",
            "ip_id": "ip_A",
            "customer_id": "cust_2"
        },
        {
            "timestamp_unix": 5000.0,
            "account_created_unix": 900.0,
            "device_id": "dev_A",
            "ip_id": "ip_A",
            "customer_id": "cust_3"
        }
    ]
    df = pd.DataFrame(data)
    t_df = extractor.extract_features(df)
    
    # Event 0: 0 prior accounts on dev_A
    assert t_df["dev_recent_acc_1h"].iloc[0] == 0
    # Event 1 (+500s): 1 prior account on dev_A
    assert t_df["dev_recent_acc_1h"].iloc[1] == 1
    # Event 2 (+3500s from event 1, total 4000s from event 0): event 0 is >3600s ago, so 1 account in last 1h
    assert t_df["dev_recent_acc_1h"].iloc[2] == 1


def test_temporal_score_range():
    scorer = TemporalEscalationScorer()
    # Null event
    null_score = scorer.calculate_transaction_escalation_score({
        "dev_recent_acc_1h": 0.0,
        "ip_recent_acc_1h": 0.0,
        "is_acc_created_within_15m": 0.0,
        "is_acc_created_within_1h": 0.0
    })
    assert 0.0 <= null_score <= 0.05

    # Extreme burst event
    burst_score = scorer.calculate_transaction_escalation_score({
        "dev_recent_acc_1h": 10.0,
        "ip_recent_acc_1h": 10.0,
        "is_acc_created_within_15m": 1.0
    })
    assert burst_score >= 0.95


def test_early_warning_deterministic_timestamps():
    """Verify deterministic ring activation (2nd customer linkage) and material escalation (50% of final ring financial volume)."""
    # 4 transactions:
    # Tx 0: t=100, cust_1, amt=1000
    # Tx 1: t=200, cust_2 (2nd distinct customer -> activation at t=200), amt=1000
    # Tx 2: t=300, cust_2 (cum amt = 3000 >= 50% of 4000 -> material escalation at t=300), amt=1000
    # Tx 3: t=400, cust_3, amt=1000
    df = pd.DataFrame([
        {"timestamp_unix": 100.0, "customer_id": "cust_1", "amount": 1000.0, "ring_id": "R1", "abuse_type": "pattern_a", "is_fraud": 1},
        {"timestamp_unix": 200.0, "customer_id": "cust_2", "amount": 1000.0, "ring_id": "R1", "abuse_type": "pattern_a", "is_fraud": 1},
        {"timestamp_unix": 300.0, "customer_id": "cust_2", "amount": 1000.0, "ring_id": "R1", "abuse_type": "pattern_a", "is_fraud": 1},
        {"timestamp_unix": 400.0, "customer_id": "cust_3", "amount": 1000.0, "ring_id": "R1", "abuse_type": "pattern_a", "is_fraud": 1}
    ])
    
    # Baseline detects at tx 2 (t=300), Sentinel detects at tx 0 (t=100)
    pred_base = np.array([0, 0, 1, 1])
    pred_sent = np.array([1, 1, 1, 1])
    
    results = EarlyWarningEvaluator.evaluate_rings(df, pred_base, pred_sent, pred_sent)
    agg = results["aggregate_by_stage"]
    
    # Sentinel post-coordination latency (alert at t=200 vs t_coord=200): (200-200)/60 = 0.0 mins
    assert agg["stage_c_sentinelgraph"]["median_time_to_detection_min"] == 0.0
    # Baseline post-coordination latency (alert at t=300 vs t_coord=200): (300-200)/60 = 1.67 mins
    assert agg["stage_a_baseline"]["median_time_to_detection_min"] == pytest.approx(1.67, 0.1)
    
    # Sentinel predictive lead time (alert at t=100 vs t_coord=200): (200-100)/60 = 1.67 mins
    assert agg["stage_c_sentinelgraph"]["median_predictive_lead_time_min"] == pytest.approx(1.67, 0.1)
    # Baseline predictive lead time (alert at t=300 vs t_coord=200): clamped to 0.0 mins
    assert agg["stage_a_baseline"]["median_predictive_lead_time_min"] == 0.0

    # Sentinel alert lead time before 50% material volume (t_mat=200 vs t_first_alert=100): (200-100)/60 = 1.67 mins
    assert agg["stage_c_sentinelgraph"]["median_alert_lead_time_min"] == pytest.approx(1.67, 0.1)
    
    # Sentinel exposure at first alert: 1000 / 4000 = 25%
    assert agg["stage_c_sentinelgraph"]["median_exposure_at_first_alert_pct"] == 25.0


def test_evidence_traceability_and_guardrails():
    case_builder = RiskCaseBuilder()
    tx_df = pd.DataFrame([
        {
            "timestamp": "2026-08-01T12:00:00",
            "timestamp_unix": 1000.0,
            "customer_id": f"c_{i}",
            "device_id": "d_shared",
            "ip_id": "ip_shared",
            "payment_instrument_id": "card_shared",
            "merchant_id": "m_1",
            "amount": 2500.0,
            "account_created_unix": 900.0
        } for i in range(10)
    ])
    
    case = case_builder.build_case_from_cluster(
        case_id="1099",
        cluster_data={"num_accounts": 10, "num_devices": 1, "num_ips": 1, "num_cards": 1},
        cluster_txs_df=tx_df,
        fused_row={
            "behavioral_score": 70.0,
            "graph_score": 90.0,
            "temporal_score": 85.0,
            "final_risk_score": 88.0,
            "confidence": 94.0,
            "severity": "CRITICAL",
            "recommended_action": "HOLD"
        }
    )
    
    assert len(case.evidence_items) >= 3
    assert case.evidence_items[0].evidence_id == "E01"
    assert case.why_alert_fired != ""
    assert case.recommended_action == DefensiveAction.HOLD

    explainer = RiskExplainer()
    explanation = explainer.explain_case(case)
    
    # Check that explanation contains traceable evidence IDs
    assert "[E01]" in explanation["formatted_markdown"]
    assert "evidence_sources" in explanation
    assert len(explanation["evidence_sources"]) >= 3
