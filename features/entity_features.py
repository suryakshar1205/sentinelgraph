"""
Entity interaction features for SentinelGraph.
Computes point-in-time entity degrees and sharing ratios strictly at timestamp t.
"""

import numpy as np
import pandas as pd
from typing import Dict, Set


class EntityFeatureExtractor:
    """Calculates point-in-time entity reuse and graph degree statistics."""

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute point-in-time entity reuse features chronologically.
        Guarantees no future lookahead.
        """
        df = df.sort_values(by="timestamp_unix").reset_index(drop=True)
        n = len(df)
        
        dev_acc_count = np.zeros(n, dtype=float)
        ip_acc_count = np.zeros(n, dtype=float)
        card_acc_count = np.zeros(n, dtype=float)
        dev_tx_count = np.zeros(n, dtype=float)
        ip_tx_count = np.zeros(n, dtype=float)
        card_tx_count = np.zeros(n, dtype=float)
        
        # State dictionaries mapping entity -> set of observed customer_ids
        device_to_custs: Dict[str, Set[str]] = {}
        ip_to_custs: Dict[str, Set[str]] = {}
        card_to_custs: Dict[str, Set[str]] = {}
        
        # State dictionaries mapping entity -> tx count
        device_txs: Dict[str, int] = {}
        ip_txs: Dict[str, int] = {}
        card_txs: Dict[str, int] = {}
        
        cust_ids = df["customer_id"].values if "customer_id" in df.columns else np.array([f"cust_{i}" for i in range(n)])
        dev_ids = df["device_id"].values if "device_id" in df.columns else np.array([f"dev_{i}" for i in range(n)])
        ip_ids = df["ip_id"].values if "ip_id" in df.columns else (df["ip_address"].values if "ip_address" in df.columns else np.array([f"ip_{i}" for i in range(n)]))
        card_ids = df["payment_instrument_id"].values if "payment_instrument_id" in df.columns else (df["card_id"].values if "card_id" in df.columns else np.array([f"card_{i}" for i in range(n)]))
        
        for i in range(n):
            c = cust_ids[i]
            d = dev_ids[i]
            ip = ip_ids[i]
            card = card_ids[i]
            
            # Record state prior to adding current tx
            d_set = device_to_custs.setdefault(d, set())
            ip_set = ip_to_custs.setdefault(ip, set())
            card_set = card_to_custs.setdefault(card, set())
            
            dev_acc_count[i] = len(d_set)
            ip_acc_count[i] = len(ip_set)
            card_acc_count[i] = len(card_set)
            
            dev_tx_count[i] = device_txs.get(d, 0)
            ip_tx_count[i] = ip_txs.get(ip, 0)
            card_tx_count[i] = card_txs.get(card, 0)
            
            # Update state with current observation
            d_set.add(c)
            ip_set.add(c)
            card_set.add(c)
            device_txs[d] = dev_tx_count[i] + 1
            ip_txs[ip] = ip_tx_count[i] + 1
            card_txs[card] = card_tx_count[i] + 1

        # Calculate high-risk sharing indicators
        is_shared_device = (dev_acc_count >= 2).astype(float)
        is_high_shared_device = (dev_acc_count >= 5).astype(float)
        is_shared_card = (card_acc_count >= 2).astype(float)
        is_high_shared_card = (card_acc_count >= 4).astype(float)
        is_shared_ip = (ip_acc_count >= 3).astype(float)
        
        # Entity reuse index (composite score)
        reuse_index = (
            dev_acc_count * 1.5 + 
            card_acc_count * 2.0 + 
            np.log1p(ip_acc_count) * 0.8
        )

        return pd.DataFrame({
            "dev_acc_count": dev_acc_count,
            "ip_acc_count": ip_acc_count,
            "card_acc_count": card_acc_count,
            "dev_tx_count": dev_tx_count,
            "ip_tx_count": ip_tx_count,
            "card_tx_count": card_tx_count,
            "is_shared_device": is_shared_device,
            "is_high_shared_device": is_high_shared_device,
            "is_shared_card": is_shared_card,
            "is_high_shared_card": is_high_shared_card,
            "is_shared_ip": is_shared_ip,
            "reuse_index": reuse_index
        }, index=df.index)
