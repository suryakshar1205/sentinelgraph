"""
Unit tests for leakage-safe feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from features.pipeline import FeaturePipeline


def test_leakage_free_velocity():
    pipeline = FeaturePipeline()
    
    # Create 3 transactions for same customer
    data = [
        {
            "timestamp": "2026-08-01T10:00:00",
            "timestamp_unix": 1785578400.0,
            "customer_id": "cust_1",
            "account_created_at": "2026-08-01T08:00:00",
            "account_created_unix": 1785571200.0,
            "device_id": "dev_1",
            "ip_id": "ip_1",
            "payment_instrument_id": "card_1",
            "merchant_id": "m_1",
            "amount": 100.0,
            "location": "Mumbai",
            "transaction_type": "payment"
        },
        {
            "timestamp": "2026-08-01T10:05:00",
            "timestamp_unix": 1785578700.0,
            "customer_id": "cust_1",
            "account_created_at": "2026-08-01T08:00:00",
            "account_created_unix": 1785571200.0,
            "device_id": "dev_1",
            "ip_id": "ip_1",
            "payment_instrument_id": "card_1",
            "merchant_id": "m_1",
            "amount": 200.0,
            "location": "Mumbai",
            "transaction_type": "payment"
        },
        {
            "timestamp": "2026-08-01T10:50:00",
            "timestamp_unix": 1785581400.0,
            "customer_id": "cust_1",
            "account_created_at": "2026-08-01T08:00:00",
            "account_created_unix": 1785571200.0,
            "device_id": "dev_1",
            "ip_id": "ip_1",
            "payment_instrument_id": "card_1",
            "merchant_id": "m_1",
            "amount": 300.0,
            "location": "Mumbai",
            "transaction_type": "payment"
        }
    ]
    df = pd.DataFrame(data)
    b_df, e_df, t_df = pipeline.extract_full_features(df)

    # First transaction should have 0 prior velocity in 10m window
    assert b_df["cust_velocity_10m"].iloc[0] == 0
    # Second transaction at +5 min should have 1 prior velocity
    assert b_df["cust_velocity_10m"].iloc[1] == 1
    # Third transaction at +50 min should have 0 prior in 10m window, but 2 in 1h window
    assert b_df["cust_velocity_10m"].iloc[2] == 0
    assert b_df["cust_velocity_1h"].iloc[2] == 2
