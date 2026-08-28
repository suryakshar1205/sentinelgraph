"""
Dynamic entity graph builder for SentinelGraph.
Maintains a heterogeneous graph connecting accounts, devices, IPs, cards, and merchants.
"""

import networkx as nx
import pandas as pd
from typing import Dict, List, Set, Any, Optional


class EntityGraphBuilder:
    """Constructs and incrementally updates the entity relationship graph."""

    def __init__(self):
        self.graph = nx.Graph()
        self.node_metadata: Dict[str, Dict[str, Any]] = {}

    def reset(self):
        self.graph.clear()
        self.node_metadata.clear()

    def add_transaction(self, tx: Dict[str, Any]):
        """Incrementally add a transaction and connect its entities."""
        c_node = f"CUST:{tx['customer_id']}"
        d_node = f"DEV:{tx['device_id']}"
        ip_node = f"IP:{tx['ip_id']}"
        card_node = f"CARD:{tx['payment_instrument_id']}"
        m_node = f"MERCH:{tx['merchant_id']}"
        
        # Add nodes with types
        self._add_node(c_node, "customer", tx)
        self._add_node(d_node, "device", tx)
        self._add_node(ip_node, "ip", tx)
        self._add_node(card_node, "payment_instrument", tx)
        self._add_node(m_node, "merchant", tx)
        
        # Add edges connecting customer to infrastructure
        self._add_edge(c_node, d_node, tx)
        self._add_edge(c_node, ip_node, tx)
        self._add_edge(c_node, card_node, tx)
        self._add_edge(c_node, m_node, tx)
        self._add_edge(d_node, ip_node, tx)

    def _add_node(self, node_id: str, node_type: str, tx: Dict[str, Any]):
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, type=node_type, first_seen=tx["timestamp"], tx_count=1)
            self.node_metadata[node_id] = {
                "type": node_type,
                "first_seen": tx["timestamp"],
                "last_seen": tx["timestamp"],
                "tx_count": 1,
                "total_amount": float(tx["amount"]),
                "transactions": [tx["transaction_id"]]
            }
        else:
            meta = self.node_metadata[node_id]
            meta["tx_count"] += 1
            meta["last_seen"] = tx["timestamp"]
            meta["total_amount"] += float(tx["amount"])
            if len(meta["transactions"]) < 100:  # Cap list to save memory
                meta["transactions"].append(tx["transaction_id"])
            self.graph.nodes[node_id]["tx_count"] = meta["tx_count"]

    def _add_edge(self, u: str, v: str, tx: Dict[str, Any]):
        if self.graph.has_edge(u, v):
            self.graph[u][v]["weight"] += 1
            self.graph[u][v]["total_amount"] += float(tx["amount"])
            self.graph[u][v]["last_seen"] = tx["timestamp"]
        else:
            self.graph.add_edge(
                u, v,
                weight=1,
                total_amount=float(tx["amount"]),
                first_seen=tx["timestamp"],
                last_seen=tx["timestamp"]
            )

    def build_from_dataframe(self, df: pd.DataFrame) -> nx.Graph:
        """Build graph in batch from dataframe."""
        self.reset()
        records = df.to_dict(orient="records")
        for r in records:
            self.add_transaction(r)
        return self.graph
