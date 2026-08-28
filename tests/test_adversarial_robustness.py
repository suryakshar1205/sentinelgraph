"""
Tests for Defensive Robustness Scenarios and Cost Sensitivity Analysis.
"""

import pytest
from evaluation.adversarial_cases import RobustnessEvaluator
from evaluation.cost_sensitivity import CostSensitivityEvaluator
from data.schemas import DefensiveAction


def test_robustness_evaluator_all_scenarios_pass():
    """Verifies that all 6 defensive stress scenarios pass their expected boundaries."""
    evaluator = RobustnessEvaluator()
    results = evaluator.evaluate_all_scenarios()

    assert results["total_scenarios"] == 6
    assert results["passed_scenarios"] == 6

    scenario_map = {s["scenario_id"]: s for s in results["scenarios"]}

    # Scenario A: Benign Shared Hardware should NOT trigger HOLD
    assert scenario_map["SCENARIO_A"]["recommended_action"] in [DefensiveAction.ALLOW.value, DefensiveAction.REVIEW.value]
    assert scenario_map["SCENARIO_A"]["final_risk_score"] < 70.0

    # Scenario B: Legitimate Flash Sale Traffic Burst should remain ALLOW
    assert scenario_map["SCENARIO_B"]["recommended_action"] == DefensiveAction.ALLOW.value
    assert scenario_map["SCENARIO_B"]["final_risk_score"] < 40.0

    # Scenario C: Individual Anomaly flags behavioral without fake graph score
    assert scenario_map["SCENARIO_C"]["graph_score"] == 0.0
    assert scenario_map["SCENARIO_C"]["final_risk_score"] >= 70.0

    # Scenario D: Coordinated Low-Signal Ring is flagged via Graph + Temporal synergy
    assert scenario_map["SCENARIO_D"]["final_risk_score"] >= 80.0
    assert scenario_map["SCENARIO_D"]["recommended_action"] == DefensiveAction.HOLD.value

    # Scenario E: Gradual Coordination provides early warning (>= 45)
    assert scenario_map["SCENARIO_E"]["final_risk_score"] >= 45.0

    # Scenario F: Insufficient Evidence triggers safe AI abstention
    assert scenario_map["SCENARIO_F"]["status"] == "PASSED"


def test_cost_sensitivity_grid_monotonicity():
    """Verifies that expected business cost increases monotonically with FN cost."""
    evaluator = CostSensitivityEvaluator(
        fp_cost_grid=[150.0],
        fn_cost_grid=[1000.0, 2500.0, 5000.0]
    )
    res = evaluator.evaluate_sensitivity_grid()
    grid = res["grid_results"]

    costs_c = [r["stage_c_cost_inr"] for r in grid]
    assert costs_c[0] < costs_c[1] < costs_c[2]

    # In all configurations with FN >= FP, SentinelGraph cost is strictly lower than Baseline
    for r in grid:
        assert r["stage_c_cost_inr"] < r["stage_a_cost_inr"]
        assert r["cost_reduction_inr"] > 0


def test_generalization_stress_testing_scenarios():
    """Verifies that all 6 generalization stress testing distributions execute properly."""
    from evaluation.stress_testing import GeneralizationStressTester
    from models.baseline import TransactionBaselineModel
    import os

    tester = GeneralizationStressTester()
    model_path = "models/artifacts/baseline_model.joblib"
    if os.path.exists(model_path):
        model = TransactionBaselineModel.load(model_path)
    else:
        model = TransactionBaselineModel()

    results = tester.run_all_stress_tests(model)
    assert len(results) == 6
    for r in results:
        assert "final_risk_score" in r
        assert "action" in r
        assert r["passed"] is True

