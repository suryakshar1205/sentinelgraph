"""
Evaluation metrics computation for SentinelGraph.
Evaluates detectors on the frozen held-out test set with honest reporting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


class EvaluationEngine:
    """Computes standard ML and business fraud metrics."""

    @staticmethod
    def compute_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, Any]:
        """Compute standard evaluation metrics."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = 0.0
            
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.0
            
        fpr = float(fp / max(1, fp + tn))

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "fpr": fpr,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }

    @staticmethod
    def compute_ring_detection_latency(
        test_df: pd.DataFrame,
        preds: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculates time-to-detection for each coordinated ring:
        Time from ring's first transaction to the first transaction detected by the model.
        """
        df_eval = test_df.copy()
        df_eval["pred"] = preds
        
        ring_groups = df_eval[df_eval["ring_id"].notna()].groupby("ring_id")
        latencies_minutes = []
        detected_rings = 0
        total_rings = len(ring_groups)
        
        for ring_id, group in ring_groups:
            group_sorted = group.sort_values(by="timestamp_unix")
            first_tx_time = group_sorted["timestamp_unix"].iloc[0]
            
            detected_txs = group_sorted[group_sorted["pred"] == 1]
            if not detected_txs.empty:
                first_det_time = detected_txs["timestamp_unix"].iloc[0]
                latency_s = max(0.0, first_det_time - first_tx_time)
                latencies_minutes.append(latency_s / 60.0)
                detected_rings += 1
            else:
                # Ring missed completely
                pass

        median_latency = float(np.median(latencies_minutes)) if latencies_minutes else 0.0
        mean_latency = float(np.mean(latencies_minutes)) if latencies_minutes else 0.0
        ring_recall = float(detected_rings / max(1, total_rings))

        return {
            "total_rings_in_test": total_rings,
            "detected_rings": detected_rings,
            "coordinated_ring_recall": ring_recall,
            "median_detection_latency_min": median_latency,
            "mean_detection_latency_min": mean_latency
        }
