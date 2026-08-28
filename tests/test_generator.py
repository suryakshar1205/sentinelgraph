"""
Unit tests for synthetic data generation and deterministic reproducibility.
"""

import pytest
import pandas as pd
import numpy as np
import datetime
from data.patterns import PatternGenerator


def test_pattern_generator_deterministic():
    rng1 = np.random.default_rng(20260827)
    start_t = datetime.datetime(2026, 8, 1, 0, 0, 0)
    pgen1 = PatternGenerator(rng1, start_t)
    prof1 = pgen1.generate_legitimate_user_profile("cust_0001")

    rng2 = np.random.default_rng(20260827)
    pgen2 = PatternGenerator(rng2, start_t)
    prof2 = pgen2.generate_legitimate_user_profile("cust_0001")

    assert prof1["customer_id"] == prof2["customer_id"]
    assert prof1["mean_amt"] == prof2["mean_amt"]
    assert prof1["primary_device"] == prof2["primary_device"]


def test_coordinated_ring_creation():
    rng = np.random.default_rng(42)
    start_t = datetime.datetime(2026, 8, 1, 0, 0, 0)
    pgen = PatternGenerator(rng, start_t)
    records = pgen.create_coordinated_ring(
        ring_id="RING_TEST_001",
        pattern_type="pattern_a_shared_device",
        base_time=start_t,
        merchants=["merch_001", "merch_002"]
    )

    assert len(records) > 10
    assert all(r["ring_id"] == "RING_TEST_001" for r in records)
    assert all(r["is_fraud"] == 1 for r in records)
    assert all(r["label"] == "coordinated_ring" for r in records)
