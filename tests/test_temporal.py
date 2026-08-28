"""
Unit tests for temporal escalation scoring.
"""

import pytest
from temporal.velocity import TemporalVelocityTracker
from temporal.escalation import TemporalEscalationScorer


def test_velocity_acceleration():
    tracker = TemporalVelocityTracker(short_window_s=1800, baseline_window_s=21600)
    # Expected baseline rate in 30 mins for 24 txs in 6 hours = 24 * (1800/21600) = 2 txs
    # If 10 txs in 30 mins, acceleration ratio = (10 - 2) / 2 = 4.0
    acc = tracker.compute_acceleration_ratio(short_count=10, baseline_count=24)
    assert acc == pytest.approx(4.0, 0.01)


def test_temporal_escalation_scorer():
    scorer = TemporalEscalationScorer()
    score_high = scorer.calculate_transaction_escalation_score({
        "dev_recent_acc_1h": 4.0,
        "ip_recent_acc_1h": 3.0,
        "is_acc_created_within_15m": 1.0
    })
    score_low = scorer.calculate_transaction_escalation_score({
        "dev_recent_acc_1h": 0.0,
        "ip_recent_acc_1h": 0.0,
        "is_acc_created_within_15m": 0.0,
        "is_acc_created_within_1h": 0.0,
        "is_acc_created_within_24h": 0.0
    })
    assert score_high >= 0.70
    assert score_low <= 0.05
