"""
Early-Warning & Detection Latency Evaluation Engine for SentinelGraph.
Measures multi-account coordination timestamps, post-coordination detection latency,
predictive lead times, material financial volume milestones, and exposure at first alert.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


class EarlyWarningEvaluator:
    """
    Computes rigorous point-in-time early warning metrics across coordinated abuse rings on the frozen test set.
    
    Rigorous Timestamp & Metric Definitions:
      - Coordination Observable Timestamp (t_coord): First timestamp where multi-account coordination becomes
        observable (i.e. the moment a 2nd distinct customer account in the ring transacts on shared infrastructure).
      - First Alert Timestamp (t_alert_first): First timestamp where the detector crosses its alert threshold.
      - Post-Coordination Detection Timestamp (t_alert_post_coord): First alert timestamp occurring AT OR AFTER t_coord.
      - 50% Material Financial Volume Timestamp (t_mat): Timestamp where cumulative ring transaction value reaches 50%
        of the final total ring transaction value.
      - Post-Coordination Detection Latency: (t_alert_post_coord - t_coord) in minutes (always >= 0.0).
      - Predictive Lead Time to Coordination: (t_coord - t_alert_first) in minutes (positive if behavioral alert fires before 2nd account joins).
      - Alert Lead Time to 50% of Final Ring Financial Volume: (t_mat - t_alert_first) in minutes before 50% of final ring financial volume is reached.
      - Exposure at First Alert (%): Cumulative transaction value up to and including t_alert_first as a % of total ring financial volume.
      - Eventual Exposure Not Yet Occurred (%): (100% - Exposure at First Alert %). Early-detection indicator, not realized loss prevention.
    """

    @staticmethod
    def evaluate_rings(
        test_df: pd.DataFrame,
        stage_a_preds: np.ndarray,
        stage_b_preds: np.ndarray,
        stage_c_preds: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate early warning metrics across all coordinated rings in the frozen test set.
        Compares Baseline (Stage A) vs Baseline + Graph (Stage B) vs SentinelGraph (Stage C).
        """
        df = test_df.copy().reset_index(drop=True)
        df["pred_a"] = stage_a_preds
        df["pred_b"] = stage_b_preds
        df["pred_c"] = stage_c_preds

        # Filter only coordinated ring events
        ring_df = df[df["ring_id"].notna()]
        ring_groups = ring_df.groupby("ring_id")
        
        rings_summary = []
        
        for ring_id, group in ring_groups:
            group_sorted = group.sort_values(by="timestamp_unix").reset_index(drop=True)
            t_first_tx = float(group_sorted["timestamp_unix"].iloc[0])
            t_last_tx = float(group_sorted["timestamp_unix"].iloc[-1])
            total_ring_exposure = float(group_sorted["amount"].sum())
            total_ring_txs = len(group_sorted)

            # 1. Coordination Observable Timestamp (t_coord): 2nd distinct customer transacts
            seen_custs = set()
            coord_idx = 0
            for idx, row in group_sorted.iterrows():
                seen_custs.add(row["customer_id"])
                if len(seen_custs) >= 2 or (len(group_sorted) == 1):
                    coord_idx = idx
                    break
            t_coord = float(group_sorted["timestamp_unix"].iloc[coord_idx])

            # 2. 50% Material Financial Volume Timestamp (t_mat): Cumulative value >= 50% of total
            cum_amounts = group_sorted["amount"].cumsum().values
            mat_idx = int(np.argmax(cum_amounts >= (0.50 * total_ring_exposure)))
            t_material = float(group_sorted["timestamp_unix"].iloc[mat_idx])

            def get_detector_ring_stats(pred_col: str):
                det_txs = group_sorted[group_sorted[pred_col] == 1]
                if det_txs.empty:
                    return {
                        "detected": False,
                        "first_alert_time_unix": None,
                        "time_to_detection_min": None,
                        "predictive_lead_time_min": 0.0,
                        "alert_lead_time_min": 0.0,
                        "exposure_at_first_alert_inr": total_ring_exposure,
                        "exposure_at_first_alert_pct": 100.0,
                        "exposure_not_yet_occurred_pct": 0.0
                    }
                
                # First alert anywhere in ring
                first_det_idx = det_txs.index[0]
                t_alert_first = float(det_txs["timestamp_unix"].iloc[0])
                
                # Post-coordination alert (at or after t_coord)
                post_coord_det_txs = group_sorted.iloc[coord_idx:][group_sorted.iloc[coord_idx:][pred_col] == 1]
                if not post_coord_det_txs.empty:
                    t_alert_post_coord = float(post_coord_det_txs["timestamp_unix"].iloc[0])
                    time_to_detection_min = max(0.0, (t_alert_post_coord - t_coord) / 60.0)
                else:
                    t_alert_post_coord = t_alert_first
                    time_to_detection_min = 0.0
                
                # Predictive lead time (how early first alert fired before multi-account coordination became observable)
                predictive_lead_time_min = max(0.0, (t_coord - t_alert_first) / 60.0)
                
                # Alert lead time before 50% of final ring financial volume
                alert_lead_time_min = max(0.0, (t_material - t_alert_first) / 60.0)
                
                # Cumulative exposure up to and including the first alerted transaction
                exp_at_first_alert = float(group_sorted.iloc[:first_det_idx + 1]["amount"].sum())
                exp_at_first_alert_pct = (exp_at_first_alert / max(1.0, total_ring_exposure)) * 100.0
                unoccurred_pct = max(0.0, 100.0 - exp_at_first_alert_pct)

                return {
                    "detected": True,
                    "first_alert_time_unix": t_alert_first,
                    "post_coord_alert_time_unix": t_alert_post_coord,
                    "time_to_detection_min": time_to_detection_min,
                    "predictive_lead_time_min": predictive_lead_time_min,
                    "alert_lead_time_min": alert_lead_time_min,
                    "exposure_at_first_alert_inr": exp_at_first_alert,
                    "exposure_at_first_alert_pct": exp_at_first_alert_pct,
                    "exposure_not_yet_occurred_pct": unoccurred_pct
                }

            stats_a = get_detector_ring_stats("pred_a")
            stats_b = get_detector_ring_stats("pred_b")
            stats_c = get_detector_ring_stats("pred_c")

            rings_summary.append({
                "ring_id": ring_id,
                "pattern_type": group_sorted["abuse_type"].iloc[0],
                "total_transactions": total_ring_txs,
                "total_exposure_inr": total_ring_exposure,
                "timestamps": {
                    "first_transaction_unix": t_first_tx,
                    "coordination_observable_unix": t_coord,
                    "material_50pct_volume_unix": t_material,
                    "last_transaction_unix": t_last_tx,
                    "total_duration_min": (t_last_tx - t_first_tx) / 60.0
                },
                "stage_a": stats_a,
                "stage_b": stats_b,
                "stage_c": stats_c
            })

        # Aggregate metrics across all rings
        def aggregate_stage(stage_key: str):
            all_stats = [r[stage_key] for r in rings_summary]
            detected = [s for s in all_stats if s["detected"]]
            n_total = len(all_stats)
            n_det = len(detected)
            
            latencies = [s["time_to_detection_min"] for s in detected if s["time_to_detection_min"] is not None]
            pred_lead_times = [s["predictive_lead_time_min"] for s in detected]
            mat_lead_times = [s["alert_lead_time_min"] for s in detected]
            exp_pcts = [s["exposure_at_first_alert_pct"] for s in detected]
            unoccurred_pcts = [s["exposure_not_yet_occurred_pct"] for s in detected]
            
            return {
                "total_rings": n_total,
                "detected_rings": n_det,
                "ring_detection_rate": float(n_det / max(1, n_total)),
                "median_time_to_detection_min": float(np.median(latencies)) if latencies else 0.0,
                "mean_time_to_detection_min": float(np.mean(latencies)) if latencies else 0.0,
                "median_predictive_lead_time_min": float(np.median(pred_lead_times)) if pred_lead_times else 0.0,
                "median_alert_lead_time_min": float(np.median(mat_lead_times)) if mat_lead_times else 0.0,
                "mean_alert_lead_time_min": float(np.mean(mat_lead_times)) if mat_lead_times else 0.0,
                "median_exposure_at_first_alert_pct": float(np.median(exp_pcts)) if exp_pcts else 100.0,
                "median_exposure_not_yet_occurred_pct": float(np.median(unoccurred_pcts)) if unoccurred_pcts else 0.0,
                "mean_exposure_not_yet_occurred_pct": float(np.mean(unoccurred_pcts)) if unoccurred_pcts else 0.0
            }

        agg_a = aggregate_stage("stage_a")
        agg_b = aggregate_stage("stage_b")
        agg_c = aggregate_stage("stage_c")

        return {
            "rings_detail": rings_summary,
            "aggregate_by_stage": {
                "stage_a_baseline": agg_a,
                "stage_b_graph": agg_b,
                "stage_c_sentinelgraph": agg_c
            }
        }
