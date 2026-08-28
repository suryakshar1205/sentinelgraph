"""
3-Stage Ablation and Early-Warning Study for SentinelGraph on Frozen Held-Out Test Set.
Compares:
  Stage A: Baseline (Behavioral ML only)
  Stage B: Baseline + Graph Intelligence
  Stage C: SentinelGraph (Baseline + Graph + Temporal Escalation)
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import precision_recall_curve, confusion_matrix

from features.pipeline import FeaturePipeline
from models.baseline import TransactionBaselineModel
from graph.signals import GraphSignalScorer
from temporal.escalation import TemporalEscalationScorer
from fusion.risk import RiskFusionEngine
from evaluation.metrics import EvaluationEngine
from evaluation.costs import BusinessCostModel
from evaluation.bootstrap import BootstrapEvaluator
from evaluation.early_warning import EarlyWarningEvaluator
from evaluation.adversarial_cases import RobustnessEvaluator
from evaluation.cost_sensitivity import CostSensitivityEvaluator


class AblationRunner:
    """Executes frozen test set ablation and generates comparative proof table."""

    def __init__(self, config_path: str = "config.yaml"):
        self.pipeline = FeaturePipeline()
        self.graph_scorer = GraphSignalScorer()
        self.temporal_scorer = TemporalEscalationScorer()
        self.fusion_engine = RiskFusionEngine()
        self.cost_model = BusinessCostModel()
        self.bootstrap = BootstrapEvaluator(n_iterations=200)

    def run_ablation(
        self,
        test_df: pd.DataFrame,
        baseline_model: TransactionBaselineModel,
        output_dir: str = "evaluation/artifacts"
    ) -> Dict[str, Any]:
        """Run 3 stages on frozen test set and record exact results."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"Running formal ablation on frozen test set ({len(test_df):,} transactions)...")

        y_true = test_df["is_fraud"].values
        amounts = test_df["amount"].values

        # 1. Extract feature sets (strictly point-in-time safe)
        beh_df, ent_df, tem_df = self.pipeline.extract_full_features(test_df)

        # Baseline probabilities
        beh_probs = baseline_model.predict_proba(test_df)
        
        # Graph scores
        graph_scores = self.graph_scorer.compute_graph_scores_for_dataframe(ent_df)
        
        # Temporal scores
        temp_scores = self.temporal_scorer.compute_escalation_scores_for_dataframe(tem_df)

        # -------------------------------------------------------------
        # STAGE A: Baseline (Behavioral ML Only)
        # -------------------------------------------------------------
        stage_a_probs = beh_probs
        stage_a_preds = (stage_a_probs >= baseline_model.optimal_threshold).astype(int)
        stage_a_metrics = EvaluationEngine.compute_classification_metrics(y_true, stage_a_preds, stage_a_probs)
        stage_a_exposure = self.cost_model.calculate_financial_exposure(y_true, stage_a_preds, amounts)
        stage_a_cost = self.cost_model.calculate_total_cost(stage_a_metrics["fp"], stage_a_metrics["fn"])

        # -------------------------------------------------------------
        # STAGE B: Baseline + Graph Intelligence
        # -------------------------------------------------------------
        stage_b_probs = np.maximum(beh_probs, 0.45 * beh_probs + 0.55 * graph_scores)
        stage_b_preds = (stage_b_probs >= baseline_model.optimal_threshold).astype(int)
        stage_b_metrics = EvaluationEngine.compute_classification_metrics(y_true, stage_b_preds, stage_b_probs)
        stage_b_exposure = self.cost_model.calculate_financial_exposure(y_true, stage_b_preds, amounts)
        stage_b_cost = self.cost_model.calculate_total_cost(stage_b_metrics["fp"], stage_b_metrics["fn"])

        # -------------------------------------------------------------
        # STAGE C: SentinelGraph (Baseline + Graph + Temporal Escalation)
        # -------------------------------------------------------------
        fused_df = self.fusion_engine.fuse_dataframe(beh_probs, graph_scores, temp_scores)
        stage_c_probs = fused_df["final_risk_score"].values / 100.0
        stage_c_preds = (stage_c_probs >= baseline_model.optimal_threshold).astype(int)
        stage_c_metrics = EvaluationEngine.compute_classification_metrics(y_true, stage_c_preds, stage_c_probs)
        stage_c_exposure = self.cost_model.calculate_financial_exposure(y_true, stage_c_preds, amounts)
        stage_c_cost = self.cost_model.calculate_total_cost(stage_c_metrics["fp"], stage_c_metrics["fn"])
        
        # Bootstrap intervals for Stage C
        stage_c_ci = self.bootstrap.compute_confidence_intervals(y_true, stage_c_preds)

        # -------------------------------------------------------------
        # EARLY-WARNING EVALUATION ACROSS ALL TEST RINGS
        # -------------------------------------------------------------
        early_warning_results = EarlyWarningEvaluator.evaluate_rings(
            test_df=test_df,
            stage_a_preds=stage_a_preds,
            stage_b_preds=stage_b_preds,
            stage_c_preds=stage_c_preds
        )
        ew_agg = early_warning_results["aggregate_by_stage"]

        # PR Curves (sample 40 points for fast UI rendering)
        def compute_pr_curve_sample(y_t, y_p):
            p, r, _ = precision_recall_curve(y_t, y_p)
            step = max(1, len(p) // 40)
            return {
                "precision": [float(v) for v in p[::step]],
                "recall": [float(v) for v in r[::step]]
            }

        pr_curve_a = compute_pr_curve_sample(y_true, stage_a_probs)
        pr_curve_b = compute_pr_curve_sample(y_true, stage_b_probs)
        pr_curve_c = compute_pr_curve_sample(y_true, stage_c_probs)

        # Cost Sensitivity Grid
        cost_sensitivity = self.cost_model.sensitivity_analysis(
            fp=stage_c_metrics["fp"],
            fn=stage_c_metrics["fn"]
        )

        # -------------------------------------------------------------
        # TEMPORAL DIAGNOSTICS
        # -------------------------------------------------------------
        test_df_copy = test_df.copy()
        test_df_copy["temp_score"] = temp_scores
        test_df_copy["graph_score"] = graph_scores
        test_df_copy["beh_prob"] = beh_probs
        
        temp_by_label = {}
        for lbl in ["legitimate", "isolated_anomaly", "coordinated_ring"]:
            sub = test_df_copy[test_df_copy["label"] == lbl]
            temp_by_label[lbl] = {
                "mean": float(sub["temp_score"].mean()) if not sub.empty else 0.0,
                "median": float(sub["temp_score"].median()) if not sub.empty else 0.0,
                "max": float(sub["temp_score"].max()) if not sub.empty else 0.0
            }
            
        corr_graph_temp = float(np.corrcoef(graph_scores, temp_scores)[0, 1]) if len(graph_scores) > 1 else 0.0
        
        # Compare Stage B vs Stage C severity impact
        b_scores = stage_b_probs * 100.0
        c_scores = stage_c_probs * 100.0
        b_high_or_crit = (b_scores >= 50.0)
        c_high_or_crit = (c_scores >= 50.0)
        severity_elevated_count = int(np.sum((~b_high_or_crit) & c_high_or_crit))

        # Supplemental Robustness Scenarios Evaluation
        robustness_evaluator = RobustnessEvaluator()
        robustness_results = robustness_evaluator.evaluate_all_scenarios()

        # Comprehensive Cost Sensitivity Grid
        cost_sens_evaluator = CostSensitivityEvaluator()
        cost_sensitivity_grid = cost_sens_evaluator.evaluate_sensitivity_grid()

        results = {
            "metadata": {
                "dataset_seed": 20260827,
                "test_sample_count": len(test_df),
                "test_fraud_count": int(y_true.sum()),
                "total_rings_in_test": ew_agg["stage_c_sentinelgraph"]["total_rings"],
                "freeze_status": "STRICTLY_FROZEN",
                "threshold_selection_source": "validation_set_optimal_f1"
            },
            "cost_assumptions": {
                "cost_fp_inr": self.cost_model.cost_fp,
                "cost_fn_inr": self.cost_model.cost_fn,
                "disclaimer": "Configured prototype assumptions — not Razorpay actual production economics."
            },
            "stages": {
                "stage_a_baseline": {
                    "name": "Stage A: Baseline (Behavioral Only)",
                    "classification": stage_a_metrics,
                    "early_warning": ew_agg["stage_a_baseline"],
                    "financial_exposure": stage_a_exposure,
                    "total_business_cost_inr": stage_a_cost,
                    "pr_curve": pr_curve_a
                },
                "stage_b_graph": {
                    "name": "Stage B: Baseline + Graph",
                    "classification": stage_b_metrics,
                    "early_warning": ew_agg["stage_b_graph"],
                    "financial_exposure": stage_b_exposure,
                    "total_business_cost_inr": stage_b_cost,
                    "pr_curve": pr_curve_b
                },
                "stage_c_sentinelgraph": {
                    "name": "Stage C: SentinelGraph (Baseline + Graph + Temporal)",
                    "classification": stage_c_metrics,
                    "early_warning": ew_agg["stage_c_sentinelgraph"],
                    "financial_exposure": stage_c_exposure,
                    "total_business_cost_inr": stage_c_cost,
                    "bootstrap_95_ci": stage_c_ci,
                    "pr_curve": pr_curve_c
                }
            },
            "temporal_diagnostics": {
                "temporal_scores_by_label": temp_by_label,
                "graph_temporal_correlation": corr_graph_temp,
                "transactions_severity_elevated_by_temporal": severity_elevated_count,
                "high_escalation_ring_events": int(np.sum(temp_scores >= 0.40))
            },
            "early_warning_rings_detail": early_warning_results["rings_detail"],
            "cost_sensitivity_analysis": cost_sensitivity,
            "cost_sensitivity_grid": cost_sensitivity_grid,
            "supplemental_robustness_scenarios": robustness_results,
            "comparison_summary": {
                "f1_improvement_pct": float(((stage_c_metrics["f1"] - stage_a_metrics["f1"]) / max(1e-4, stage_a_metrics["f1"])) * 100.0),
                "cost_reduction_inr": float(stage_a_cost - stage_c_cost),
                "cost_reduction_pct": float(((stage_a_cost - stage_c_cost) / max(1.0, stage_a_cost)) * 100.0),
                "detected_exposure_delta_inr": float(stage_c_exposure["detected_exposure_inr"] - stage_a_exposure["detected_exposure_inr"])
            }
        }

        # Save to disk
        out_path = os.path.join(output_dir, "ablation_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        print("\n=================== ABLATION RESULTS (FROZEN TEST SET) ===================")
        print(f"{'Metric':<34} | {'Stage A (Baseline)':<18} | {'Stage B (+Graph)':<18} | {'Stage C (SentinelGraph)':<22}")
        print("-" * 104)
        print(f"{'Precision':<34} | {stage_a_metrics['precision']:<18.4f} | {stage_b_metrics['precision']:<18.4f} | {stage_c_metrics['precision']:<22.4f}")
        print(f"{'Recall':<34} | {stage_a_metrics['recall']:<18.4f} | {stage_b_metrics['recall']:<18.4f} | {stage_c_metrics['recall']:<22.4f}")
        print(f"{'F1 Score':<34} | {stage_a_metrics['f1']:<18.4f} | {stage_b_metrics['f1']:<18.4f} | {stage_c_metrics['f1']:<22.4f}")
        print(f"{'PR-AUC':<34} | {stage_a_metrics['pr_auc']:<18.4f} | {stage_b_metrics['pr_auc']:<18.4f} | {stage_c_metrics['pr_auc']:<22.4f}")
        print(f"{'Ring Detection Rate':<34} | {ew_agg['stage_a_baseline']['ring_detection_rate']*100:<17.1f}% | {ew_agg['stage_b_graph']['ring_detection_rate']*100:<17.1f}% | {ew_agg['stage_c_sentinelgraph']['ring_detection_rate']*100:<21.1f}%")
        print(f"{'Post-Coordination Latency':<34} | {ew_agg['stage_a_baseline']['median_time_to_detection_min']:<15.1f} min | {ew_agg['stage_b_graph']['median_time_to_detection_min']:<15.1f} min | {ew_agg['stage_c_sentinelgraph']['median_time_to_detection_min']:<19.1f} min")
        print(f"{'Predictive Lead Time (to Coord)':<34} | {ew_agg['stage_a_baseline']['median_predictive_lead_time_min']:<15.1f} min | {ew_agg['stage_b_graph']['median_predictive_lead_time_min']:<15.1f} min | {ew_agg['stage_c_sentinelgraph']['median_predictive_lead_time_min']:<19.1f} min")
        print(f"{'Alert Lead Time to 50% of Final Ring Financial Volume':<52} | {ew_agg['stage_a_baseline']['median_alert_lead_time_min']:<15.1f} min | {ew_agg['stage_b_graph']['median_alert_lead_time_min']:<15.1f} min | {ew_agg['stage_c_sentinelgraph']['median_alert_lead_time_min']:<19.1f} min")
        print(f"{'Exposure at First Alert':<34} | {ew_agg['stage_a_baseline']['median_exposure_at_first_alert_pct']:<17.1f}% | {ew_agg['stage_b_graph']['median_exposure_at_first_alert_pct']:<17.1f}% | {ew_agg['stage_c_sentinelgraph']['median_exposure_at_first_alert_pct']:<21.1f}%")
        print(f"{'Prototype Expected Cost':<34} | INR {stage_a_cost:<14,.0f} | INR {stage_b_cost:<14,.0f} | INR {stage_c_cost:<18,.0f}")
        print(f"{'Detected Suspicious Exposure':<34} | INR {stage_a_exposure['detected_exposure_inr']:<14,.0f} | INR {stage_b_exposure['detected_exposure_inr']:<14,.0f} | INR {stage_c_exposure['detected_exposure_inr']:<18,.0f}")
        print("=========================================================================\n")

        return results
