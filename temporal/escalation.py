"""
Temporal escalation and early-warning scoring for SentinelGraph.
Measures entity-level arrival bursts and rapid account creation waves.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


class TemporalEscalationScorer:
    """Combines entity-level arrival velocity, account creation bursts, and surge acceleration into an escalation score in [0, 1]."""

    def __init__(
        self,
        w_dev_arrival: float = 0.40,
        w_ip_arrival: float = 0.20,
        w_acc_creation: float = 0.40
    ):
        self.w_dev_arrival = w_dev_arrival
        self.w_ip_arrival = w_ip_arrival
        self.w_acc_creation = w_acc_creation

    def calculate_transaction_escalation_score(self, temporal_feat_row: Dict[str, float]) -> float:
        """Calculate transaction-level point-in-time temporal escalation score in [0, 1]."""
        dev_1h = temporal_feat_row.get("dev_recent_acc_1h", 0.0)
        ip_1h = temporal_feat_row.get("ip_recent_acc_1h", 0.0)
        is_new_15m = temporal_feat_row.get("is_acc_created_within_15m", 0.0)
        is_new_1h = temporal_feat_row.get("is_acc_created_within_1h", 0.0)
        is_new_24h = temporal_feat_row.get("is_acc_created_within_24h", 0.0)
        
        # Normalized signals
        # Dev arrival: 0 for 1 account, scales to 1.0 for >= 4 accounts in 1h
        s_dev = min(1.0, max(0.0, dev_1h / 3.0))
        # IP arrival: 0 for <= 2 accounts, scales to 1.0 for >= 8 accounts in 1h
        s_ip = min(1.0, max(0.0, (ip_1h - 1.0) / 6.0))
        # Account creation burst
        s_creation = 1.0 if is_new_15m > 0 else (0.75 if is_new_1h > 0 else (0.35 if is_new_24h > 0 else 0.0))
        
        raw_score = (
            self.w_dev_arrival * s_dev +
            self.w_ip_arrival * s_ip +
            self.w_acc_creation * s_creation
        )
        return float(np.clip(raw_score, 0.0, 1.0))

    def compute_escalation_scores_for_dataframe(self, temporal_df: pd.DataFrame) -> np.ndarray:
        """Vectorized computation for dataframe."""
        dev_1h = temporal_df["dev_recent_acc_1h"].values
        ip_1h = temporal_df["ip_recent_acc_1h"].values
        is_new_15m = temporal_df["is_acc_created_within_15m"].values
        is_new_1h = temporal_df["is_acc_created_within_1h"].values
        is_new_24h = temporal_df["is_acc_created_within_24h"].values
        
        s_dev = np.clip(dev_1h / 3.0, 0.0, 1.0)
        s_ip = np.clip(np.maximum(0.0, ip_1h - 1.0) / 6.0, 0.0, 1.0)
        s_creation = np.where(
            is_new_15m > 0, 1.0,
            np.where(is_new_1h > 0, 0.75, np.where(is_new_24h > 0, 0.35, 0.0))
        )
        
        raw_scores = (
            self.w_dev_arrival * s_dev +
            self.w_ip_arrival * s_ip +
            self.w_acc_creation * s_creation
        )
        return np.clip(raw_scores, 0.0, 1.0)
