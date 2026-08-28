"""
Bootstrap confidence interval estimation for SentinelGraph metrics.
"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score


class BootstrapEvaluator:
    """Calculates 95% empirical bootstrap confidence intervals."""

    def __init__(self, n_iterations: int = 500, random_state: int = 20260827):
        self.n_iterations = n_iterations
        self.rng = np.random.default_rng(random_state)

    def compute_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Compute mean and (2.5%, 97.5%) percentiles for Precision, Recall, and F1.
        """
        n = len(y_true)
        p_list = []
        r_list = []
        f1_list = []

        for _ in range(self.n_iterations):
            indices = self.rng.integers(0, n, size=n)
            y_t_samp = y_true[indices]
            y_p_samp = y_pred[indices]

            p_list.append(precision_score(y_t_samp, y_p_samp, zero_division=0))
            r_list.append(recall_score(y_t_samp, y_p_samp, zero_division=0))
            f1_list.append(f1_score(y_t_samp, y_p_samp, zero_division=0))

        def get_stats(arr):
            return {
                "mean": float(np.mean(arr)),
                "ci_lower": float(np.percentile(arr, 2.5)),
                "ci_upper": float(np.percentile(arr, 97.5))
            }

        return {
            "precision": get_stats(p_list),
            "recall": get_stats(r_list),
            "f1": get_stats(f1_list)
        }
