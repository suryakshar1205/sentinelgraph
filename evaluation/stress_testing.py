"""
Defensive Generalization & Stress Testing Suite for SentinelGraph.
Evaluates model robustness on non-standard, perturbed synthetic distributions.
Note: This suite is strictly separate from the official frozen held-out benchmark.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from features.pipeline import FeaturePipeline
from models.baseline import TransactionBaselineModel
from graph.signals import GraphSignalScorer
from temporal.escalation import TemporalEscalationScorer
from fusion.risk import RiskFusionEngine
from data.schemas import DefensiveAction


class GeneralizationStressTester:
    """
    Evaluates SentinelGraph performance across 6 distinct stress distributions
    without altering the frozen benchmark test set.
    """

    def __init__(self):
        self.feature_pipeline = FeaturePipeline()
        self.graph_scorer = GraphSignalScorer()
        self.temporal_scorer = TemporalEscalationScorer()
        self.fusion_engine = RiskFusionEngine()

    def run_all_stress_tests(self, baseline_model: TransactionBaselineModel) -> List[Dict[str, Any]]:
        scenarios = [
            self._test_sparse_ring_density(baseline_model),
            self._test_sub_threshold_amounts(baseline_model),
            self._test_rapid_activation_burst(baseline_model),
            self._test_mixed_cluster_noisy_ip(baseline_model),
            self._test_weak_relationship_device_only(baseline_model),
            self._test_slow_drip_coordination(baseline_model),
        ]
        return scenarios

    def _test_sparse_ring_density(self, model: TransactionBaselineModel) -> Dict[str, Any]:
        """Stress Test 1: Low-density ring (only 3 accounts sharing 1 device instead of 20)."""
        txs = [
            {"amount": 4200.0, "customer_age_days": 2.0, "device_id": "D_SPARSE_1", "card_id": "C_SPARSE_1", "ip_address": "198.51.100.1", "customer_id": "CUST_S1", "timestamp_unix": 1700000000.0, "timestamp": "2023-11-14T22:13:20Z", "account_created_unix": 1699800000.0},
            {"amount": 4500.0, "customer_age_days": 1.0, "device_id": "D_SPARSE_1", "card_id": "C_SPARSE_2", "ip_address": "198.51.100.2", "customer_id": "CUST_S2", "timestamp_unix": 1700000300.0, "timestamp": "2023-11-14T22:18:20Z", "account_created_unix": 1699900000.0},
            {"amount": 4100.0, "customer_age_days": 1.0, "device_id": "D_SPARSE_1", "card_id": "C_SPARSE_3", "ip_address": "198.51.100.3", "customer_id": "CUST_S3", "timestamp_unix": 1700000600.0, "timestamp": "2023-11-14T22:23:20Z", "account_created_unix": 1699900000.0},
        ]
        df = pd.DataFrame(txs)
        beh_score = float(model.predict_proba(df).mean() * 100.0) if hasattr(model, "predict_proba") else 60.0
        
        # Entity sharing counts: 3 accounts on 1 device
        entity_df = pd.DataFrame([{"dev_acc_count": 3, "card_acc_count": 1, "ip_acc_count": 1}])
        graph_score = float(self.graph_scorer.compute_graph_scores_for_dataframe(entity_df)[0] * 100.0)
        
        temporal_df = pd.DataFrame([{
            "dev_recent_acc_1h": 3, "ip_recent_acc_1h": 1,
            "is_acc_created_within_15m": 1, "is_acc_created_within_1h": 1, "is_acc_created_within_24h": 1
        }])
        temp_score = float(self.temporal_scorer.compute_escalation_scores_for_dataframe(temporal_df)[0] * 100.0)
        
        risk, conf, sev, action = self.fusion_engine.fuse(beh_score / 100.0, graph_score / 100.0, temp_score / 100.0)
        
        return {
            "test_id": "ST_01",
            "name": "Sparse Ring Density (3 accounts)",
            "description": "Small 3-account ring sharing 1 device; tests sensitivity when ring size is small.",
            "behavioral_score": beh_score,
            "graph_score": graph_score,
            "temporal_score": temp_score,
            "final_risk_score": risk,
            "confidence": conf,
            "action": action.value,
            "expected_action": "REVIEW",
            "passed": action in [DefensiveAction.REVIEW, DefensiveAction.HOLD]
        }

    def _test_sub_threshold_amounts(self, model: TransactionBaselineModel) -> Dict[str, Any]:
        """Stress Test 2: Sub-threshold micro-amounts (INR 150-300 to evade amount filters)."""
        txs = [
            {"amount": 180.0, "customer_age_days": 1.0, "device_id": "D_MICRO_1", "card_id": "C_MICRO_1", "ip_address": "203.0.113.1", "customer_id": "CUST_M1", "timestamp_unix": 1700000000.0, "timestamp": "2023-11-14T22:13:20Z", "account_created_unix": 1699900000.0},
            {"amount": 220.0, "customer_age_days": 1.0, "device_id": "D_MICRO_1", "card_id": "C_MICRO_1", "ip_address": "203.0.113.1", "customer_id": "CUST_M2", "timestamp_unix": 1700000300.0, "timestamp": "2023-11-14T22:18:20Z", "account_created_unix": 1699900000.0},
        ]
        df = pd.DataFrame(txs)
        beh_score = float(model.predict_proba(df).mean() * 100.0) if hasattr(model, "predict_proba") else 40.0
        
        entity_df = pd.DataFrame([{"dev_acc_count": 8, "card_acc_count": 8, "ip_acc_count": 8}])
        graph_score = float(self.graph_scorer.compute_graph_scores_for_dataframe(entity_df)[0] * 100.0)
        
        temporal_df = pd.DataFrame([{
            "dev_recent_acc_1h": 8, "ip_recent_acc_1h": 8,
            "is_acc_created_within_15m": 1, "is_acc_created_within_1h": 1, "is_acc_created_within_24h": 1
        }])
        temp_score = float(self.temporal_scorer.compute_escalation_scores_for_dataframe(temporal_df)[0] * 100.0)
        
        risk, conf, sev, action = self.fusion_engine.fuse(beh_score / 100.0, graph_score / 100.0, temp_score / 100.0)
        
        return {
            "test_id": "ST_02",
            "name": "Sub-Threshold Micro Amounts (INR 180-220)",
            "description": "Individually tiny amounts designed to bypass transaction-level value anomaly filters.",
            "behavioral_score": beh_score,
            "graph_score": graph_score,
            "temporal_score": temp_score,
            "final_risk_score": risk,
            "confidence": conf,
            "action": action.value,
            "expected_action": "HOLD",
            "passed": action == DefensiveAction.HOLD
        }

    def _test_rapid_activation_burst(self, model: TransactionBaselineModel) -> Dict[str, Any]:
        """Stress Test 3: Sudden synchronized activation across 15 accounts in 2 minutes."""
        entity_df = pd.DataFrame([{"dev_acc_count": 15, "card_acc_count": 5, "ip_acc_count": 2}])
        graph_score = float(self.graph_scorer.compute_graph_scores_for_dataframe(entity_df)[0] * 100.0)
        
        temporal_df = pd.DataFrame([{
            "dev_recent_acc_1h": 15, "ip_recent_acc_1h": 15,
            "is_acc_created_within_15m": 1, "is_acc_created_within_1h": 1, "is_acc_created_within_24h": 1
        }])
        temp_score = float(self.temporal_scorer.compute_escalation_scores_for_dataframe(temporal_df)[0] * 100.0)
        
        beh_score = 65.0
        risk, conf, sev, action = self.fusion_engine.fuse(beh_score / 100.0, graph_score / 100.0, temp_score / 100.0)
        
        return {
            "test_id": "ST_03",
            "name": "Rapid Synchronized Activation (15 accounts / 2m)",
            "description": "Flash botnet activation; verifies temporal escalation amplification.",
            "behavioral_score": beh_score,
            "graph_score": graph_score,
            "temporal_score": temp_score,
            "final_risk_score": risk,
            "confidence": conf,
            "action": action.value,
            "expected_action": "HOLD",
            "passed": action == DefensiveAction.HOLD
        }

    def _test_mixed_cluster_noisy_ip(self, model: TransactionBaselineModel) -> Dict[str, Any]:
        """Stress Test 4: Public corporate gateway IP shared across 50 normal users + 1 fraud attempt."""
        # 50 accounts on 1 IP, but only 1 device per account (no device sharing)
        entity_df = pd.DataFrame([{"dev_acc_count": 1, "card_acc_count": 1, "ip_acc_count": 50}])
        graph_score = float(self.graph_scorer.compute_graph_scores_for_dataframe(entity_df)[0] * 100.0)
        
        temporal_df = pd.DataFrame([{
            "dev_recent_acc_1h": 0, "ip_recent_acc_1h": 2,
            "is_acc_created_within_15m": 0, "is_acc_created_within_1h": 0, "is_acc_created_within_24h": 0
        }])
        temp_score = float(self.temporal_scorer.compute_escalation_scores_for_dataframe(temporal_df)[0] * 100.0)
        
        beh_score = 15.0  # Normal individual transaction
        risk, conf, sev, action = self.fusion_engine.fuse(beh_score / 100.0, graph_score / 100.0, temp_score / 100.0)
        
        return {
            "test_id": "ST_04",
            "name": "Noisy Public Gateway IP (50 accounts / 1 IP)",
            "description": "High IP concentration without device or card reuse; verifies false-positive resistance.",
            "behavioral_score": beh_score,
            "graph_score": graph_score,
            "temporal_score": temp_score,
            "final_risk_score": risk,
            "confidence": conf,
            "action": action.value,
            "expected_action": "ALLOW",
            "passed": action == DefensiveAction.ALLOW
        }

    def _test_weak_relationship_device_only(self, model: TransactionBaselineModel) -> Dict[str, Any]:
        """Stress Test 5: Device sharing without payment instrument sharing."""
        entity_df = pd.DataFrame([{"dev_acc_count": 4, "card_acc_count": 1, "ip_acc_count": 2}])
        graph_score = float(self.graph_scorer.compute_graph_scores_for_dataframe(entity_df)[0] * 100.0)
        
        temporal_df = pd.DataFrame([{
            "dev_recent_acc_1h": 2, "ip_recent_acc_1h": 1,
            "is_acc_created_within_15m": 1, "is_acc_created_within_1h": 1, "is_acc_created_within_24h": 1
        }])
        temp_score = float(self.temporal_scorer.compute_escalation_scores_for_dataframe(temporal_df)[0] * 100.0)
        
        beh_score = 45.0
        risk, conf, sev, action = self.fusion_engine.fuse(beh_score / 100.0, graph_score / 100.0, temp_score / 100.0)
        
        return {
            "test_id": "ST_05",
            "name": "Device Reuse Without Card Sharing",
            "description": "Shared device across 4 accounts with distinct payment cards.",
            "behavioral_score": beh_score,
            "graph_score": graph_score,
            "temporal_score": temp_score,
            "final_risk_score": risk,
            "confidence": conf,
            "action": action.value,
            "expected_action": "REVIEW",
            "passed": action in [DefensiveAction.REVIEW, DefensiveAction.HOLD]
        }

    def _test_slow_drip_coordination(self, model: TransactionBaselineModel) -> Dict[str, Any]:
        """Stress Test 6: Low-frequency slow drip coordination over days (zero hourly burst)."""
        entity_df = pd.DataFrame([{"dev_acc_count": 8, "card_acc_count": 3, "ip_acc_count": 2}])
        graph_score = float(self.graph_scorer.compute_graph_scores_for_dataframe(entity_df)[0] * 100.0)
        
        # Temporal score is 0 because velocity burst is intentionally suppressed by attacker
        temporal_df = pd.DataFrame([{
            "dev_recent_acc_1h": 0, "ip_recent_acc_1h": 0,
            "is_acc_created_within_15m": 0, "is_acc_created_within_1h": 0, "is_acc_created_within_24h": 0
        }])
        temp_score = float(self.temporal_scorer.compute_escalation_scores_for_dataframe(temporal_df)[0] * 100.0)
        
        beh_score = 50.0
        risk, conf, sev, action = self.fusion_engine.fuse(beh_score / 100.0, graph_score / 100.0, temp_score / 100.0)
        
        return {
            "test_id": "ST_06",
            "name": "Slow-Drip Coordinated Abuse (Zero Burst)",
            "description": "Attacker spaces transactions over days; graph relationship detects ring even with zero velocity burst.",
            "behavioral_score": beh_score,
            "graph_score": graph_score,
            "temporal_score": temp_score,
            "final_risk_score": risk,
            "confidence": conf,
            "action": action.value,
            "expected_action": "REVIEW",
            "passed": action in [DefensiveAction.REVIEW, DefensiveAction.HOLD]
        }
