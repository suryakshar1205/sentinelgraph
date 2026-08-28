"""
Cluster expansion velocity and entity arrival rate calculations for SentinelGraph.
"""

import numpy as np
from typing import Dict, List, Any


class ClusterGrowthTracker:
    """Measures the arrival velocity of new accounts and infrastructure within clusters."""

    def compute_growth_rate(self, current_active_accounts: int, prior_active_accounts: int) -> float:
        """
        Compute percentage growth rate of active ring accounts.
        """
        if prior_active_accounts <= 0:
            return float(current_active_accounts * 100.0) if current_active_accounts > 0 else 0.0
            
        rate = ((current_active_accounts - prior_active_accounts) / float(prior_active_accounts)) * 100.0
        return float(max(0.0, rate))
