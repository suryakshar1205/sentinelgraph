"""
Unit tests for evaluation metrics and cost calculation.
"""

import pytest
import numpy as np
from evaluation.metrics import EvaluationEngine
from evaluation.costs import BusinessCostModel


def test_classification_metrics():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    y_prob = np.array([0.9, 0.4, 0.1, 0.2, 0.85])

    metrics = EvaluationEngine.compute_classification_metrics(y_true, y_pred, y_prob)
    assert metrics["precision"] == 1.0  # 2 TP / 2 predicted positive
    assert metrics["recall"] == pytest.approx(2.0 / 3.0, 0.01)
    assert metrics["fp"] == 0
    assert metrics["fn"] == 1


def test_business_cost_model():
    cost_model = BusinessCostModel(cost_fp=150.0, cost_fn=2500.0)
    # 2 FP and 3 FN -> 2*150 + 3*2500 = 300 + 7500 = 7800
    total = cost_model.calculate_total_cost(fp=2, fn=3)
    assert total == 7800.0
