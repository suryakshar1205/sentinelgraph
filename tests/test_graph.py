"""
Unit tests for graph building and community detection.
"""

import pytest
import pandas as pd
from graph.builder import EntityGraphBuilder
from graph.communities import GraphCommunityDetector
from graph.signals import GraphSignalScorer


def test_entity_graph_builder_and_clusters():
    # 4 accounts sharing 1 device and 1 card
    data = []
    for i in range(4):
        data.append({
            "transaction_id": f"tx_{i}",
            "timestamp": "2026-08-01T12:00:00",
            "timestamp_unix": 1785585600.0,
            "customer_id": f"cust_{i}",
            "device_id": "shared_dev_99",
            "ip_id": f"ip_{i}",
            "payment_instrument_id": "shared_card_99",
            "merchant_id": "m_1",
            "amount": 1500.0
        })
    df = pd.DataFrame(data)

    builder = EntityGraphBuilder()
    graph = builder.build_from_dataframe(df)
    
    assert graph.has_node("DEV:shared_dev_99")
    assert graph.has_node("CARD:shared_card_99")
    assert graph.degree("DEV:shared_dev_99") >= 4

    detector = GraphCommunityDetector(min_cluster_accounts=3)
    clusters = detector.extract_candidate_clusters(graph)
    
    assert len(clusters) >= 1
    assert clusters[0]["num_accounts"] == 4
    assert clusters[0]["num_devices"] == 1
