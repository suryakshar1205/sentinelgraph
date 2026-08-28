"""
Unit tests for calibrated risk fusion and thresholds.
"""

import pytest
from fusion.risk import RiskFusionEngine
from fusion.thresholds import RiskThresholdManager
from data.schemas import SeverityLevel, DefensiveAction


def test_fusion_engine_high_risk():
    engine = RiskFusionEngine(w_behavioral=0.30, w_graph=0.45, w_temporal=0.25)
    final_score, confidence, severity, action = engine.fuse(
        beh_prob=0.80,
        graph_score=0.90,
        temp_score=0.85
    )

    # 100 * (0.3*0.8 + 0.45*0.9 + 0.25*0.85) = 100 * (0.24 + 0.405 + 0.2125) = 85.75
    assert final_score == pytest.approx(85.75, 0.1)
    assert severity == SeverityLevel.CRITICAL or severity == SeverityLevel.HIGH
    assert action == DefensiveAction.HOLD
    assert confidence >= 85.0


def test_fusion_engine_low_risk():
    engine = RiskFusionEngine()
    final_score, confidence, severity, action = engine.fuse(
        beh_prob=0.05,
        graph_score=0.0,
        temp_score=0.02
    )

    assert final_score < 10.0
    assert severity == SeverityLevel.LOW
    assert action == DefensiveAction.ALLOW
