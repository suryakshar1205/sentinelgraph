"""
Graph community and connected component extractor for SentinelGraph.
"""

import networkx as nx
from typing import List, Dict, Set, Any


class GraphCommunityDetector:
    """Discovers suspicious connected subgraphs and clusters."""

    def __init__(self, min_cluster_accounts: int = 3):
        self.min_cluster_accounts = min_cluster_accounts

    def extract_candidate_clusters(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """
        Extract connected components containing shared non-merchant infrastructure.
        Excludes merchant hubs from merging independent legitimate users.
        """
        # Create non-merchant projection subgraph to avoid giant component collapse across merchants
        non_merch_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") != "merchant"]
        subgraph = graph.subgraph(non_merch_nodes)
        
        components = list(nx.connected_components(subgraph))
        clusters = []
        
        for idx, comp in enumerate(components):
            # Tally entity types
            accounts = [n for n in comp if n.startswith("CUST:")]
            devices = [n for n in comp if n.startswith("DEV:")]
            ips = [n for n in comp if n.startswith("IP:")]
            cards = [n for n in comp if n.startswith("CARD:")]
            
            # If component has shared resources across multiple accounts
            if len(accounts) >= self.min_cluster_accounts and (len(devices) < len(accounts) or len(cards) < len(accounts) or len(ips) < len(accounts)):
                # Calculate internal density and edge weights
                comp_sub = subgraph.subgraph(comp)
                num_edges = comp_sub.number_of_edges()
                total_tx_volume = sum(
                    graph.nodes[n].get("tx_count", 1) for n in accounts
                )
                
                clusters.append({
                    "cluster_id": f"CLUSTER_{idx+1:04d}",
                    "nodes": list(comp),
                    "num_accounts": len(accounts),
                    "num_devices": len(devices),
                    "num_ips": len(ips),
                    "num_cards": len(cards),
                    "num_nodes": len(comp),
                    "num_edges": num_edges,
                    "accounts": accounts,
                    "devices": devices,
                    "ips": ips,
                    "cards": cards,
                    "total_tx_volume": total_tx_volume
                })
                
        # Sort by suspiciousness (ratio of accounts to shared entities)
        clusters.sort(key=lambda c: (c["num_accounts"] / max(1, c["num_devices"] + c["num_cards"])), reverse=True)
        return clusters
