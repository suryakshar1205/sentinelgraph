"""
Transaction-Level Baseline Model for SentinelGraph.
Trained strictly on transaction and customer behavioral features (no network/graph intelligence).
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from features.pipeline import FeaturePipeline
from models.calibration import ModelCalibrator


class TransactionBaselineModel:
    """Transaction-level baseline classifier without graph or relationship signals."""

    def __init__(self, random_state: int = 20260827):
        self.random_state = random_state
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=150,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=self.random_state
        )
        self.calibrator = None
        self.optimal_threshold: float = 0.5
        self.feature_columns = [
            "amount", "log_amount", "account_age_days", "is_new_account", "is_very_new_account",
            "hour_sin", "hour_cos", "is_weekend", "cust_tx_count_prior",
            "cust_mean_amt_prior", "amt_deviation_ratio",
            "cust_velocity_10m", "cust_velocity_1h", "cust_velocity_24h"
        ]

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
        """Train on training set and calibrate/tune threshold on validation set."""
        pipeline = FeaturePipeline()
        
        # 1. Extract baseline features
        X_train_df = pipeline.extract_baseline_features(train_df)
        X_train = X_train_df[self.feature_columns].values
        y_train = train_df["is_fraud"].values
        
        X_val_df = pipeline.extract_baseline_features(val_df)
        X_val = X_val_df[self.feature_columns].values
        y_val = val_df["is_fraud"].values

        print(f"Training baseline model on {len(X_train):,} samples (fraud rate: {y_train.mean():.4f})...")
        self.model.fit(X_train, y_train)

        # 2. Calibrate on validation set
        cal = ModelCalibrator(method="isotonic")
        self.calibrator = cal.calibrate(self.model, X_val, y_val)

        # 3. Find optimal threshold on validation set to maximize F1
        val_probs = self.calibrator.predict_proba(X_val)[:, 1]
        
        best_thresh = 0.5
        best_f1 = -1.0
        for thresh in np.linspace(0.1, 0.9, 81):
            preds = (val_probs >= thresh).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)
                
        self.optimal_threshold = best_thresh
        val_preds = (val_probs >= self.optimal_threshold).astype(int)
        
        metrics = {
            "val_precision": float(precision_score(y_val, val_preds, zero_division=0)),
            "val_recall": float(recall_score(y_val, val_preds, zero_division=0)),
            "val_f1": float(best_f1),
            "val_pr_auc": float(average_precision_score(y_val, val_probs)),
            "val_roc_auc": float(roc_auc_score(y_val, val_probs)),
            "optimal_threshold": self.optimal_threshold
        }
        print(f"Baseline validation metrics: F1={metrics['val_f1']:.4f}, Precision={metrics['val_precision']:.4f}, Recall={metrics['val_recall']:.4f}, PR-AUC={metrics['val_pr_auc']:.4f}")
        return metrics

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return calibrated risk probabilities in [0, 1]."""
        pipeline = FeaturePipeline()
        X_df = pipeline.extract_baseline_features(df)
        X = X_df[self.feature_columns].values
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X)[:, 1]
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """Return binary fraud predictions using calibrated threshold."""
        thresh = threshold if threshold is not None else self.optimal_threshold
        probs = self.predict_proba(df)
        return (probs >= thresh).astype(int)

    def save(self, artifact_path: str = "models/artifacts/baseline_model.joblib"):
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "calibrator": self.calibrator,
            "optimal_threshold": self.optimal_threshold,
            "feature_columns": self.feature_columns
        }, artifact_path)
        print(f"Saved baseline model to {artifact_path}")

    @classmethod
    def load(cls, artifact_path: str = "models/artifacts/baseline_model.joblib") -> "TransactionBaselineModel":
        data = joblib.load(artifact_path)
        instance = cls()
        instance.model = data["model"]
        instance.calibrator = data["calibrator"]
        instance.optimal_threshold = data["optimal_threshold"]
        instance.feature_columns = data["feature_columns"]
        return instance


if __name__ == "__main__":
    train_df = pd.read_parquet("data/generated/train.parquet")
    val_df = pd.read_parquet("data/generated/val.parquet")
    bm = TransactionBaselineModel()
    bm.fit(train_df, val_df)
    bm.save()
