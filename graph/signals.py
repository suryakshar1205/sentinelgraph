"""
Graph coordination and entity reuse scoring for SentinelGraph.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


class GraphSignalScorer:
    """Calculates deterministic network coordination scores from graph topology."""

    def __init__(
        self,
        w_device: float = 0.35,
        w_card: float = 0.35,
        w_ip: float = 0.15,
        w_creation: float = 0.15
    ):
        self.w_device = w_device
        self.w_card = w_card
        self.w_ip = w_ip
        self.w_creation = w_creation

    def calculate_transaction_graph_score(self, entity_feat_row: Dict[str, float]) -> float:
        """
        Calculate single transaction point-in-time graph coordination score in [0, 1].
        """
        dev_count = entity_feat_row.get("dev_acc_count", 0.0)
        card_count = entity_feat_row.get("card_acc_count", 0.0)
        ip_count = entity_feat_row.get("ip_acc_count", 0.0)
        
        # Sublinear scaling of entity sharing degrees
        s_device = min(1.0, (max(0.0, dev_count - 1.0) / 4.0))
        s_card = min(1.0, (max(0.0, card_count - 1.0) / 2.0))
        s_ip = min(1.0, (max(0.0, ip_count - 2.0) / 10.0))
        
        # High sharing multiplier
        high_mult = 1.0
        if dev_count >= 5 or card_count >= 3:
            high_mult = 1.35
            
        raw_score = (
            self.w_device * s_device +
            self.w_card * s_card +
            self.w_ip * s_ip
        ) * high_mult
        
        return float(np.clip(raw_score, 0.0, 1.0))

    def compute_graph_scores_for_dataframe(self, entity_df: pd.DataFrame) -> np.ndarray:
        """Vectorized graph score calculation for dataframe."""
        dev_count = entity_df["dev_acc_count"].values
        card_count = entity_df["card_acc_count"].values
        ip_count = entity_df["ip_acc_count"].values
        
        s_device = np.clip((np.maximum(0.0, dev_count - 1.0) / 4.0), 0.0, 1.0)
        s_card = np.clip((np.maximum(0.0, card_count - 1.0) / 2.0), 0.0, 1.0)
        s_ip = np.clip((np.maximum(0.0, ip_count - 2.0) / 10.0), 0.0, 1.0)
        
        high_mult = np.where((dev_count >= 5) | (card_count >= 3), 1.35, 1.0)
        
        raw_scores = (
            self.w_device * s_device +
            self.w_card * s_card +
            self.w_ip * s_ip
        ) * high_mult
        
        return np.clip(raw_scores, 0.0, 1.0)
