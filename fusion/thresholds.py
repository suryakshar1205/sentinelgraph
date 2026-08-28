"""
Threshold definitions and severity mapping for SentinelGraph.
"""

from data.schemas import SeverityLevel, DefensiveAction


class RiskThresholdManager:
    """Manages calibrated score-to-severity mappings and defensive recommendations."""

    def __init__(
        self,
        low_thresh: float = 25.0,
        medium_thresh: float = 50.0,
        high_thresh: float = 75.0,
        critical_thresh: float = 90.0
    ):
        self.low_thresh = low_thresh
        self.medium_thresh = medium_thresh
        self.high_thresh = high_thresh
        self.critical_thresh = critical_thresh

    def map_score_to_severity(self, score: float) -> SeverityLevel:
        """Map 0-100 risk score to standard severity level."""
        if score >= self.high_thresh:
            return SeverityLevel.CRITICAL if score >= self.critical_thresh else SeverityLevel.HIGH
        elif score >= self.medium_thresh:
            return SeverityLevel.MEDIUM
        return SeverityLevel.LOW

    def recommend_action(self, score: float, confidence: float) -> DefensiveAction:
        """
        Generate defense-only recommendation based on calibrated score and confidence.
        High/Critical with sufficient confidence triggers HOLD; moderate triggers REVIEW.
        """
        if score >= self.high_thresh and confidence >= 60.0:
            return DefensiveAction.HOLD
        elif score >= self.medium_thresh or (score >= 40.0 and confidence >= 80.0):
            return DefensiveAction.REVIEW
        return DefensiveAction.ALLOW
