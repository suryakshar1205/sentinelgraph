"""
Cost Sensitivity Analysis for SentinelGraph.
Evaluates prototype business cost across comprehensive FP/FN cost parameter grids
under the frozen test set confusion matrices.
Strictly labeled as prototype modeling assumptions — not realized production savings.
"""

from typing import Dict, Any, List
import pandas as pd


class CostSensitivityEvaluator:
    """
    Computes comparative expected business cost grids across Stage A, Stage B, and Stage C.
    
    Cost Formula:
      Expected Cost = (FP * cost_FP) + (FN * cost_FN)
    """

    DEFAULT_FP_COSTS = [50.0, 100.0, 150.0, 250.0, 500.0]
    DEFAULT_FN_COSTS = [500.0, 1000.0, 2500.0, 5000.0, 10000.0]

    # Exact frozen test set confusion matrix counts
    STAGE_CM = {
        "stage_a_baseline": {"tp": 879, "fp": 72, "tn": 43168, "fn": 231, "name": "Stage A (Baseline)"},
        "stage_b_graph": {"tp": 908, "fp": 72, "tn": 43168, "fn": 202, "name": "Stage B (+Graph)"},
        "stage_c_sentinelgraph": {"tp": 908, "fp": 72, "tn": 43168, "fn": 202, "name": "Stage C (SentinelGraph)"}
    }

    def __init__(
        self,
        fp_cost_grid: List[float] = None,
        fn_cost_grid: List[float] = None
    ):
        self.fp_costs = fp_cost_grid or self.DEFAULT_FP_COSTS
        self.fn_costs = fn_cost_grid or self.DEFAULT_FN_COSTS

    def evaluate_sensitivity_grid(self) -> Dict[str, Any]:
        """Runs sensitivity analysis across all parameter combinations."""
        grid_rows = []

        for c_fp in self.fp_costs:
            for c_fn in self.fn_costs:
                # Stage A Cost
                cm_a = self.STAGE_CM["stage_a_baseline"]
                cost_a = cm_a["fp"] * c_fp + cm_a["fn"] * c_fn

                # Stage C Cost
                cm_c = self.STAGE_CM["stage_c_sentinelgraph"]
                cost_c = cm_c["fp"] * c_fp + cm_c["fn"] * c_fn

                diff_inr = cost_a - cost_c
                diff_pct = (diff_inr / max(1.0, cost_a)) * 100.0

                grid_rows.append({
                    "cost_fp_inr": c_fp,
                    "cost_fn_inr": c_fn,
                    "stage_a_cost_inr": cost_a,
                    "stage_c_cost_inr": cost_c,
                    "cost_reduction_inr": diff_inr,
                    "cost_reduction_pct": diff_pct,
                    "fn_to_fp_cost_ratio": round(c_fn / c_fp, 1)
                })

        return {
            "title": "Business Cost Sensitivity Grid Analysis",
            "disclaimer": "Configured prototype assumptions — not realized production savings.",
            "primary_assumptions": {
                "primary_cost_fp_inr": 150.0,
                "primary_cost_fn_inr": 2500.0,
                "primary_cost_reduction_inr": 72500.0,
                "primary_cost_reduction_pct": 12.32
            },
            "grid_results": grid_rows
        }
