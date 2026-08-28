"""
Probability calibration for SentinelGraph models.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator


class ModelCalibrator:
    """Calibrates output probabilities using isotonic regression or sigmoid scaling."""

    def __init__(self, method: str = "isotonic"):
        self.method = method

    def calibrate(self, base_estimator: BaseEstimator, X_val: np.ndarray, y_val: np.ndarray) -> CalibratedClassifierCV:
        """Calibrate fitted estimator against held-out validation set."""
        calibrated = CalibratedClassifierCV(estimator=base_estimator, method=self.method, cv="prefit")
        calibrated.fit(X_val, y_val)
        return calibrated
