"""
Deterministic risk fusion and multi-signal confidence estimation for SentinelGraph.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from fusion.thresholds import RiskThresholdManager
from data.schemas import SeverityLevel, DefensiveAction


class RiskFusionEngine:
    """Fuses behavioral, graph, and temporal scores into unified calibrated risk cases."""

    def __init__(
        self,
        w_behavioral: float = 0.30,
        w_graph: float = 0.45,
        w_temporal: float = 0.25
    ):
        self.w_beh = w_behavioral
        self.w_graph = w_graph
        self.w_temp = w_temporal
        self.threshold_mgr = RiskThresholdManager()

    def fuse(
        self,
        beh_prob: float,
        graph_score: float,
        temp_score: float
    ) -> Tuple[float, float, SeverityLevel, DefensiveAction]:
        """
        Fuse individual 0-1 scores into [0, 100] final score, confidence, severity, and action.
        Catches both isolated anomalies and subtle coordinated abuse rings.
        """
        weighted_blend = (
            self.w_beh * beh_prob +
            self.w_graph * graph_score +
            self.w_temp * temp_score
        )
        
        # Max-blend ensures baseline behavioral anomalies are not suppressed while ring signals elevate risk
        composite = max(beh_prob, weighted_blend, 0.50 * graph_score + 0.50 * temp_score)
        final_score = float(np.clip(composite * 100.0, 0.0, 100.0))
        
        # Calculate Confidence Score (0 - 100) based on signal consistency and magnitude
        active_signals = sum([
            1 if beh_prob >= 0.40 else 0,
            1 if graph_score >= 0.35 else 0,
            1 if temp_score >= 0.30 else 0
        ])
        
        max_sig = max(beh_prob, graph_score, temp_score)
        agreement_boost = 0.20 if active_signals >= 2 else (0.35 if active_signals == 3 else 0.0)
        confidence = float(np.clip((max_sig + agreement_boost) * 100.0, 15.0, 99.0))
        
        severity = self.threshold_mgr.map_score_to_severity(final_score)
        action = self.threshold_mgr.recommend_action(final_score, confidence)
        
        return final_score, confidence, severity, action

    def fuse_dataframe(
        self,
        beh_probs: np.ndarray,
        graph_scores: np.ndarray,
        temp_scores: np.ndarray
    ) -> pd.DataFrame:
        """Vectorized fusion over arrays."""
        weighted_blend = (
            self.w_beh * beh_probs +
            self.w_graph * graph_scores +
            self.w_temp * temp_scores
        )
        graph_temp_blend = 0.50 * graph_scores + 0.50 * temp_scores
        composite = np.maximum.reduce([beh_probs, weighted_blend, graph_temp_blend])
        final_scores = np.clip(composite * 100.0, 0.0, 100.0)
        
        active_signals = (
            (beh_probs >= 0.40).astype(int) +
            (graph_scores >= 0.35).astype(int) +
            (temp_scores >= 0.30).astype(int)
        )
        
        max_sig = np.maximum.reduce([beh_probs, graph_scores, temp_scores])
        agreement_boost = np.where(active_signals == 3, 0.35, np.where(active_signals == 2, 0.20, 0.0))
        confidences = np.clip((max_sig + agreement_boost) * 100.0, 15.0, 99.0)
        
        severities = [self.threshold_mgr.map_score_to_severity(s).value for s in final_scores]
        actions = [
            self.threshold_mgr.recommend_action(s, c).value
            for s, c in zip(final_scores, confidences)
        ]
        
        return pd.DataFrame({
            "behavioral_score": beh_probs * 100.0,
            "graph_score": graph_scores * 100.0,
            "temporal_score": temp_scores * 100.0,
            "final_risk_score": final_scores,
            "confidence": confidences,
            "severity": severities,
            "recommended_action": actions
        })
