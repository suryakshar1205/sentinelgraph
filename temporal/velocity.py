"""
Temporal velocity and activity window calculations for SentinelGraph.
"""

import numpy as np
import pandas as pd


class TemporalVelocityTracker:
    """Calculates short-window to rolling-baseline velocity ratios."""

    def __init__(self, short_window_s: int = 1800, baseline_window_s: int = 21600):
        self.short_window_s = short_window_s      # 30 mins
        self.baseline_window_s = baseline_window_s # 6 hours

    def compute_acceleration_ratio(self, short_count: float, baseline_count: float) -> float:
        """
        Compute normalized acceleration ratio between short window and baseline window.
        Expected baseline rate scaled to short window duration.
        """
        window_scale = self.short_window_s / max(1.0, float(self.baseline_window_s))
        expected_short = baseline_count * window_scale
        
        if expected_short <= 0:
            return 1.0 if short_count > 0 else 0.0
            
        ratio = (short_count - expected_short) / max(1.0, expected_short)
        return float(np.clip(ratio, 0.0, 10.0))
