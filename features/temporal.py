"""
Temporal velocity, entity-burst and escalation feature extraction for SentinelGraph.
Computes point-in-time temporal acceleration strictly at timestamp t without lookahead.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class TemporalFeatureExtractor:
    """
    Computes point-in-time entity-level arrival bursts and temporal escalation.
    """

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute temporal acceleration, device/IP account velocity, and account creation burst density.
        Strictly chronological at detection time t.
        """
        df = df.sort_values(by="timestamp_unix").reset_index(drop=True)
        n = len(df)
        ts_arr = df["timestamp_unix"].values
        acc_created_arr = df["account_created_unix"].values if "account_created_unix" in df.columns else ts_arr - 86400.0
        dev_arr = df["device_id"].values if "device_id" in df.columns else np.array([f"dev_{i}" for i in range(n)])
        ip_arr = df["ip_id"].values if "ip_id" in df.columns else (df["ip_address"].values if "ip_address" in df.columns else np.array([f"ip_{i}" for i in range(n)]))
        cust_arr = df["customer_id"].values if "customer_id" in df.columns else np.array([f"cust_{i}" for i in range(n)])
        
        # 1. Time since account creation
        time_since_creation_s = np.maximum(1.0, ts_arr - acc_created_arr)
        time_since_creation_hours = time_since_creation_s / 3600.0
        
        is_acc_created_within_15m = (time_since_creation_s <= 900.0).astype(float)
        is_acc_created_within_1h = (time_since_creation_s <= 3600.0).astype(float)
        is_acc_created_within_6h = (time_since_creation_s <= 21600.0).astype(float)
        is_acc_created_within_24h = (time_since_creation_s <= 86400.0).astype(float)

        # 2. Entity-level account arrival velocity (new accounts on this device/IP in recent window)
        # Track history of (timestamp, cust_id) per device and per IP
        dev_recent_acc_1h = np.zeros(n, dtype=float)
        dev_recent_acc_24h = np.zeros(n, dtype=float)
        ip_recent_acc_1h = np.zeros(n, dtype=float)
        
        device_cust_timestamps: Dict[str, List[tuple]] = {}
        ip_cust_timestamps: Dict[str, List[tuple]] = {}
        
        for i in range(n):
            t = ts_arr[i]
            d = dev_arr[i]
            ip = ip_arr[i]
            c = cust_arr[i]
            
            # Query device history
            d_hist = device_cust_timestamps.setdefault(d, [])
            # Filter distinct customers within last 1h (3600s) and 24h (86400s)
            d_1h_custs = set()
            d_24h_custs = set()
            for (prev_t, prev_c) in reversed(d_hist):
                if t - prev_t > 86400:
                    break
                d_24h_custs.add(prev_c)
                if t - prev_t <= 3600:
                    d_1h_custs.add(prev_c)
                    
            dev_recent_acc_1h[i] = len(d_1h_custs)
            dev_recent_acc_24h[i] = len(d_24h_custs)
            
            # Query IP history
            ip_hist = ip_cust_timestamps.setdefault(ip, [])
            ip_1h_custs = set()
            for (prev_t, prev_c) in reversed(ip_hist):
                if t - prev_t > 3600:
                    break
                ip_1h_custs.add(prev_c)
            ip_recent_acc_1h[i] = len(ip_1h_custs)
            
            # Record current observation
            d_hist.append((t, c))
            ip_hist.append((t, c))

        # 3. Burst Acceleration Index
        # High when multiple distinct accounts arrive on same hardware in short window
        burst_acceleration = (
            dev_recent_acc_1h * 2.0 +
            ip_recent_acc_1h * 0.8 +
            is_acc_created_within_1h * 1.5 +
            is_acc_created_within_15m * 2.5
        )

        return pd.DataFrame({
            "time_since_creation_hours": time_since_creation_hours,
            "is_acc_created_within_15m": is_acc_created_within_15m,
            "is_acc_created_within_1h": is_acc_created_within_1h,
            "is_acc_created_within_6h": is_acc_created_within_6h,
            "is_acc_created_within_24h": is_acc_created_within_24h,
            "dev_recent_acc_1h": dev_recent_acc_1h,
            "dev_recent_acc_24h": dev_recent_acc_24h,
            "ip_recent_acc_1h": ip_recent_acc_1h,
            "burst_acceleration": burst_acceleration
        }, index=df.index)
