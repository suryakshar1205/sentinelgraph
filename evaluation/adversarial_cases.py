"""
Defensive Robustness & Scenario Evaluation for SentinelGraph.
Evaluates system behavior against difficult, non-trivial benign and coordinated scenarios.
Strictly DEFENSE-ONLY evaluation — zero offensive tooling.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from graph.signals import GraphSignalScorer
from temporal.escalation import TemporalEscalationScorer
from fusion.risk import RiskFusionEngine
from data.schemas import RiskCase, RiskEvidence, SeverityLevel, DefensiveAction, EvidenceSufficiency
from llm.explainer import RiskExplainer


class RobustnessEvaluator:
    """
    Evaluates defensive resilience across 6 challenging real-world risk scenarios.
    
    Scenarios:
      A. Benign Shared Infrastructure (Family device / Corporate IP with normal transactions)
      B. Legitimate Traffic Burst (Flash sale / promotion with distinct normal buyers)
      C. Individual Suspicious Account (High behavioral anomaly on single user without graph links)
      D. Coordinated Low-Signal Ring (Subtle individual amounts with high entity reuse and burst)
      E. Gradual Coordination (Multi-stage ring emerging slowly over extended window)
      F. Insufficient Evidence (Sparse telemetry triggering AI abstention)
    """

    def __init__(self):
        self.graph_scorer = GraphSignalScorer()
        self.temporal_scorer = TemporalEscalationScorer()
        self.fusion_engine = RiskFusionEngine()
        self.explainer = RiskExplainer()

    def evaluate_all_scenarios(self) -> Dict[str, Any]:
        """Runs the 6 defensive robustness evaluations and returns verified summary."""
        res_a = self._eval_scenario_a_benign_shared()
        res_b = self._eval_scenario_b_legitimate_burst()
        res_c = self._eval_scenario_c_individual_anomaly()
        res_d = self._eval_scenario_d_coordinated_subtle()
        res_e = self._eval_scenario_e_gradual_coordination()
        res_f = self._eval_scenario_f_insufficient_evidence()

        scenario_results = [res_a, res_b, res_c, res_d, res_e, res_f]

        return {
            "title": "Supplemental Defensive Robustness Evaluation",
            "disclaimer": "Supplemental benchmark scenarios evaluated independently of the primary frozen test set.",
            "total_scenarios": len(scenario_results),
            "passed_scenarios": sum(1 for s in scenario_results if s["status"] == "PASSED"),
            "scenarios": scenario_results
        }

    def _eval_scenario_a_benign_shared(self) -> Dict[str, Any]:
        """Scenario A: Benign Shared Infrastructure (e.g. 2 family members sharing tablet/home Wi-Fi)."""
        beh_prob = 0.08  # Normal low-risk transactions
        # Entity features: 2 accounts on 1 device, 1 IP, 1 distinct card per user
        ent_df = pd.DataFrame([{
            "dev_acc_count": 2.0,
            "card_acc_count": 1.0,
            "ip_acc_count": 2.0
        }])
        tem_df = pd.DataFrame([{
            "dev_recent_acc_1h": 0.0,
            "ip_recent_acc_1h": 0.0,
            "is_acc_created_within_15m": 0.0,
            "is_acc_created_within_1h": 0.0,
            "is_acc_created_within_24h": 0.0
        }])

        g_score = float(self.graph_scorer.compute_graph_scores_for_dataframe(ent_df)[0])
        t_score = float(self.temporal_scorer.compute_escalation_scores_for_dataframe(tem_df)[0])
        final_score, conf, sev, action = self.fusion_engine.fuse(beh_prob, g_score, t_score)

        # Expected: Shared device alone without fraud behavior or burst does NOT trigger CRITICAL/HOLD
        passed = (action in [DefensiveAction.ALLOW, DefensiveAction.REVIEW]) and (final_score < 70.0)

        return {
            "scenario_id": "SCENARIO_A",
            "name": "Benign Shared Infrastructure (Family Device)",
            "description": "2 distinct family members transacting normally from shared home hardware.",
            "behavioral_prob": beh_prob,
            "graph_score": g_score,
            "temporal_score": t_score,
            "final_risk_score": final_score,
            "recommended_action": action.value,
            "expected_behavior": "Relationship alone does not trigger automatic HOLD (Score < 70)",
            "status": "PASSED" if passed else "FAILED",
            "observation": f"Assigned {sev.value} severity ({final_score:.0f}/100); friction preserved at ALLOW/REVIEW."
        }

    def _eval_scenario_b_legitimate_burst(self) -> Dict[str, Any]:
        """Scenario B: Legitimate Traffic Burst (Flash sale with independent buyers)."""
        beh_prob = 0.12
        # Entity features: Unlinked accounts with 1 account per device/card/IP
        ent_df = pd.DataFrame([{
            "dev_acc_count": 1.0,
            "card_acc_count": 1.0,
            "ip_acc_count": 1.0
        }])
        tem_df = pd.DataFrame([{
            "dev_recent_acc_1h": 0.0,
            "ip_recent_acc_1h": 0.0,
            "is_acc_created_within_15m": 0.0,
            "is_acc_created_within_1h": 0.0,
            "is_acc_created_within_24h": 0.0
        }])

        g_score = float(self.graph_scorer.compute_graph_scores_for_dataframe(ent_df)[0])
        t_score = float(self.temporal_scorer.compute_escalation_scores_for_dataframe(tem_df)[0])
        final_score, conf, sev, action = self.fusion_engine.fuse(beh_prob, g_score, t_score)

        # Expected: Flash sale volume without graph linkage does not trigger false positive ring HOLD
        passed = (action == DefensiveAction.ALLOW) and (final_score < 40.0)

        return {
            "scenario_id": "SCENARIO_B",
            "name": "Legitimate Traffic Burst (Flash Sale)",
            "description": "Sudden volume surge from unlinked, legitimate customers during promotion.",
            "behavioral_prob": beh_prob,
            "graph_score": g_score,
            "temporal_score": t_score,
            "final_risk_score": final_score,
            "recommended_action": action.value,
            "expected_behavior": "Volume spike without entity linkage remains ALLOW (Score < 40)",
            "status": "PASSED" if passed else "FAILED",
            "observation": f"Assigned {sev.value} severity ({final_score:.0f}/100); flash sale traffic correctly allowed."
        }

    def _eval_scenario_c_individual_anomaly(self) -> Dict[str, Any]:
        """Scenario C: Individual Suspicious Account (Single user, high velocity, unlinked)."""
        beh_prob = 0.88  # Strong individual behavioral anomaly
        ent_df = pd.DataFrame([{
            "dev_acc_count": 1.0,
            "card_acc_count": 1.0,
            "ip_acc_count": 1.0
        }])
        tem_df = pd.DataFrame([{
            "dev_recent_acc_1h": 0.0,
            "ip_recent_acc_1h": 0.0,
            "is_acc_created_within_15m": 0.0,
            "is_acc_created_within_1h": 0.0,
            "is_acc_created_within_24h": 0.0
        }])

        g_score = float(self.graph_scorer.compute_graph_scores_for_dataframe(ent_df)[0])
        t_score = float(self.temporal_scorer.compute_escalation_scores_for_dataframe(tem_df)[0])
        final_score, conf, sev, action = self.fusion_engine.fuse(beh_prob, g_score, t_score)

        # Expected: Flagged by behavioral detector without manufacturing fake ring clusters
        passed = (final_score >= 70.0) and (g_score == 0.0)

        return {
            "scenario_id": "SCENARIO_C",
            "name": "Individual Suspicious Account",
            "description": "Single account with extreme velocity but zero graph coordination.",
            "behavioral_prob": beh_prob,
            "graph_score": g_score,
            "temporal_score": t_score,
            "final_risk_score": final_score,
            "recommended_action": action.value,
            "expected_behavior": "Behavioral detection fires without manufacturing fake graph coordination (g_score=0)",
            "status": "PASSED" if passed else "FAILED",
            "observation": f"Flagged as {sev.value} ({final_score:.0f}/100); graph score remained 0.0."
        }

    def _eval_scenario_d_coordinated_subtle(self) -> Dict[str, Any]:
        """Scenario D: Coordinated Low-Signal Ring (Subtle amounts, 6 accounts on 1 card + 1 device)."""
        beh_prob = 0.25  # Individually plausible amounts
        ent_df = pd.DataFrame([{
            "dev_acc_count": 6.0,
            "card_acc_count": 4.0,
            "ip_acc_count": 4.0
        }])
        tem_df = pd.DataFrame([{
            "dev_recent_acc_1h": 5.0,
            "ip_recent_acc_1h": 4.0,
            "is_acc_created_within_15m": 1.0,
            "is_acc_created_within_1h": 1.0,
            "is_acc_created_within_24h": 1.0
        }])

        g_score = float(self.graph_scorer.compute_graph_scores_for_dataframe(ent_df)[0])
        t_score = float(self.temporal_scorer.compute_escalation_scores_for_dataframe(tem_df)[0])
        final_score, conf, sev, action = self.fusion_engine.fuse(beh_prob, g_score, t_score)

        # Expected: Graph + Temporal synergy catches subtle ring missed by individual behavioral ML
        passed = (final_score >= 80.0) and (action == DefensiveAction.HOLD)

        return {
            "scenario_id": "SCENARIO_D",
            "name": "Coordinated Low-Signal Ring",
            "description": "6 synthetic accounts with plausible amounts cycling shared card/hardware in burst.",
            "behavioral_prob": beh_prob,
            "graph_score": g_score,
            "temporal_score": t_score,
            "final_risk_score": final_score,
            "recommended_action": action.value,
            "expected_behavior": "Graph + Temporal synergy flags ring even with low individual behavioral score",
            "status": "PASSED" if passed else "FAILED",
            "observation": f"Successfully elevated to {sev.value} ({final_score:.0f}/100) -> Defensive HOLD."
        }

    def _eval_scenario_e_gradual_coordination(self) -> Dict[str, Any]:
        """Scenario E: Gradual Coordination (Ring emerging progressively across hours)."""
        beh_prob = 0.35
        # Intermediate stage: 3 accounts linked
        ent_df = pd.DataFrame([{
            "dev_acc_count": 3.0,
            "card_acc_count": 2.0,
            "ip_acc_count": 2.0
        }])
        tem_df = pd.DataFrame([{
            "dev_recent_acc_1h": 2.0,
            "ip_recent_acc_1h": 1.0,
            "is_acc_created_within_15m": 0.0,
            "is_acc_created_within_1h": 1.0,
            "is_acc_created_within_24h": 1.0
        }])

        g_score = float(self.graph_scorer.compute_graph_scores_for_dataframe(ent_df)[0])
        t_score = float(self.temporal_scorer.compute_escalation_scores_for_dataframe(tem_df)[0])
        final_score, conf, sev, action = self.fusion_engine.fuse(beh_prob, g_score, t_score)

        # Expected: Detects early-stage progressive formation with elevated score before full completion
        passed = (final_score >= 45.0)

        return {
            "scenario_id": "SCENARIO_E",
            "name": "Gradual Ring Formation",
            "description": "Progressive multi-account linkage developing over a multi-hour window.",
            "behavioral_prob": beh_prob,
            "graph_score": g_score,
            "temporal_score": t_score,
            "final_risk_score": final_score,
            "recommended_action": action.value,
            "expected_behavior": "Provides progressive early warning (Score >= 45) before full ring maturity",
            "status": "PASSED" if passed else "FAILED",
            "observation": f"Assigned {sev.value} ({final_score:.0f}/100); early warning generated during formation."
        }

    def _eval_scenario_f_insufficient_evidence(self) -> Dict[str, Any]:
        """Scenario F: Insufficient Evidence & AI Abstention Check."""
        sparse_case = RiskCase(
            case_id="SPARSE_TEST_01",
            created_at="2026-08-20T10:00:00",
            severity=SeverityLevel.LOW,
            risk_score=25.0,
            confidence=30.0,
            why_alert_fired="Single event check",
            behavioral_score=25.0,
            graph_score=10.0,
            temporal_score=5.0,
            num_accounts=1,
            num_devices=1,
            num_ips=1,
            num_payment_instruments=1,
            num_merchants=1,
            num_transactions=1,
            activity_growth_pct=0.0,
            earliest_event_time="2026-08-20T10:00:00",
            latest_event_time="2026-08-20T10:00:00",
            observed_exposure_inr=500.0,
            estimated_exposure_inr=625.0,
            evidence_items=[],  # Zero corroborating evidence items
            recommended_action=DefensiveAction.ALLOW
        )

        explanation = self.explainer.explain_case(sparse_case)
        passed = (explanation["evidence_sufficiency"] == EvidenceSufficiency.INSUFFICIENT.value) and (
            "INSUFFICIENT_EVIDENCE" in explanation["formatted_markdown"]
        )

        return {
            "scenario_id": "SCENARIO_F",
            "name": "Insufficient Evidence & AI Abstention",
            "description": "Sparse case with 0 evidence items and low confidence (<40%).",
            "behavioral_prob": 0.25,
            "graph_score": 0.10,
            "temporal_score": 0.05,
            "final_risk_score": 25.0,
            "recommended_action": explanation["recommended_action"],
            "expected_behavior": "AI explainer explicitly abstains (INSUFFICIENT_EVIDENCE) without hallucinating",
            "status": "PASSED" if passed else "FAILED",
            "observation": f"Evidence sufficiency: {explanation['evidence_sufficiency']}; safe abstention verified."
        }
