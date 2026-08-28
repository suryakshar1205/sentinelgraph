"""
False-Positive and Business Cost Modeling for SentinelGraph.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


class BusinessCostModel:
    """Calculates expected operational friction and fraud loss costs."""

    def __init__(self, cost_fp: float = 150.0, cost_fn: float = 2500.0):
        self.cost_fp = cost_fp  # Operational cost per false positive (review/friction)
        self.cost_fn = cost_fn  # Direct loss per missed fraud transaction

    def calculate_total_cost(self, fp: int, fn: int) -> float:
        """Total Expected Business Cost = FP * cost_fp + FN * cost_fn"""
        return float(fp * self.cost_fp + fn * self.cost_fn)

    def calculate_financial_exposure(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        amounts: np.ndarray
    ) -> Dict[str, float]:
        """
        Computes detected vs missed monetary fraud exposure.
        """
        fraud_mask = (y_true == 1)
        total_fraud_exposure = float(np.sum(amounts[fraud_mask]))
        
        detected_mask = fraud_mask & (y_pred == 1)
        detected_exposure = float(np.sum(amounts[detected_mask]))
        
        missed_mask = fraud_mask & (y_pred == 0)
        missed_exposure = float(np.sum(amounts[missed_mask]))
        
        exposure_coverage_pct = (detected_exposure / max(1.0, total_fraud_exposure)) * 100.0

        return {
            "total_fraud_exposure_inr": total_fraud_exposure,
            "detected_exposure_inr": detected_exposure,
            "missed_exposure_inr": missed_exposure,
            "exposure_coverage_pct": float(exposure_coverage_pct)
        }

    def sensitivity_analysis(
        self,
        fp: int,
        fn: int,
        fp_cost_range: List[float] = [50.0, 100.0, 150.0, 250.0, 500.0],
        fn_cost_range: List[float] = [1000.0, 2000.0, 2500.0, 3500.0, 5000.0]
    ) -> List[Dict[str, Any]]:
        """Run cost sensitivity grid."""
        results = []
        for c_fp in fp_cost_range:
            for c_fn in fn_cost_range:
                total_cost = fp * c_fp + fn * c_fn
                results.append({
                    "cost_fp": c_fp,
                    "cost_fn": c_fn,
                    "fp_count": fp,
                    "fn_count": fn,
                    "fp_cost_total": fp * c_fp,
                    "fn_cost_total": fn * c_fn,
                    "total_cost": total_cost
                })
        return results
