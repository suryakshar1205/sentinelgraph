"""
Behavioral feature extraction for SentinelGraph.
Extracts transaction-level and individual customer behavioral signals strictly at detection time t.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class BehavioralFeatureExtractor:
    """Computes behavioral and velocity features without future lookahead."""

    def __init__(self):
        # Rolling historical trackers
        self.customer_history: Dict[str, List[Dict]] = {}
        self.merchant_history: Dict[str, List[float]] = {}

    def reset(self):
        self.customer_history.clear()
        self.merchant_history.clear()

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute leakage-free behavioral features sequentially or vectorially
        using detection-time information only.
        """
        # Ensure chronological ordering
        df = df.sort_values(by="timestamp_unix").reset_index(drop=True)
        
        # Datetime features
        dt_series = pd.to_datetime(df["timestamp"], format="ISO8601")
        hour = dt_series.dt.hour
        day_of_week = dt_series.dt.dayofweek
        
        # Cyclic hour encoding
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        is_weekend = (day_of_week >= 5).astype(float)
        
        # Account age in days at detection time t
        acc_age_days = (df["timestamp_unix"] - df["account_created_unix"]) / 86400.0
        acc_age_days = np.maximum(0.001, acc_age_days)
        is_new_account = (acc_age_days <= 1.0).astype(float)
        is_very_new_account = (acc_age_days <= 0.125).astype(float) # <= 3 hours

        # Amount features
        amt = df["amount"].values
        log_amt = np.log1p(amt)

        # Efficient cumulative per-customer & per-merchant calculations
        # Customer transaction count & cumulative amount prior to current tx
        cust_tx_count = df.groupby("customer_id").cumcount().values
        cust_cum_amt = df.groupby("customer_id")["amount"].cumsum().values - amt
        cust_mean_amt_prior = np.where(cust_tx_count > 0, cust_cum_amt / np.maximum(1, cust_tx_count), amt)
        
        # Deviation from personal average
        amt_deviation_ratio = amt / np.maximum(10.0, cust_mean_amt_prior)
        
        # Velocity rolling counts (10m, 1h, 24h)
        # Using vectorized searchsorted on sorted timestamps per customer
        v_10m = np.zeros(len(df), dtype=float)
        v_1h = np.zeros(len(df), dtype=float)
        v_24h = np.zeros(len(df), dtype=float)
        
        # Group by customer to calculate rolling velocities
        cust_groups = df.groupby("customer_id")
        for _, indices in cust_groups.groups.items():
            idx_arr = indices.values
            ts_arr = df.loc[idx_arr, "timestamp_unix"].values
            
            # For each tx in group, find previous txs within window
            # 10 mins = 600s, 1 hr = 3600s, 24 hr = 86400s
            left_10m = np.searchsorted(ts_arr, ts_arr - 600, side="left")
            left_1h = np.searchsorted(ts_arr, ts_arr - 3600, side="left")
            left_24h = np.searchsorted(ts_arr, ts_arr - 86400, side="left")
            
            curr_pos = np.arange(len(ts_arr))
            v_10m[idx_arr] = curr_pos - left_10m
            v_1h[idx_arr] = curr_pos - left_1h
            v_24h[idx_arr] = curr_pos - left_24h

        feature_df = pd.DataFrame({
            "amount": amt,
            "log_amount": log_amt,
            "account_age_days": acc_age_days,
            "is_new_account": is_new_account,
            "is_very_new_account": is_very_new_account,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "is_weekend": is_weekend,
            "cust_tx_count_prior": cust_tx_count,
            "cust_mean_amt_prior": cust_mean_amt_prior,
            "amt_deviation_ratio": amt_deviation_ratio,
            "cust_velocity_10m": v_10m,
            "cust_velocity_1h": v_1h,
            "cust_velocity_24h": v_24h
        }, index=df.index)

        return feature_df
