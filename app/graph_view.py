"""
Network Graph Explorer for SentinelGraph.
"""

import streamlit as st
import pandas as pd
import networkx as nx
from typing import List

from graph.builder import EntityGraphBuilder
from graph.communities import GraphCommunityDetector
from graph.visualization import GraphVisualizer


def render_graph_explorer(transactions_df: pd.DataFrame):
    st.markdown("## 🕸️ Network Graph Explorer")
    st.caption("Interactive multi-entity relationship graph and community cluster isolation")

    if transactions_df is None or transactions_df.empty:
        st.warning("No transactions dataset loaded.")
        return

    # Filter controls
    c1, c2, c3 = st.columns(3)
    with c1:
        sample_size = st.slider("Sample Transaction Size", min_value=100, max_value=2000, value=600, step=100)
    with c2:
        filter_type = st.selectbox("Filter Entity Subgraph", ["All Coordinated Clusters", "Shared Device Rings", "Shared Card Rings", "Shared IP Hubs"])
    with c3:
        highlight_ring = st.checkbox("Highlight Suspicious Entities in Neon Red", value=True)

    # Subsample data
    subset = transactions_df.head(sample_size)
    
    # Build graph
    builder = EntityGraphBuilder()
    graph = builder.build_from_dataframe(subset)
    detector = GraphCommunityDetector(min_cluster_accounts=2)
    clusters = detector.extract_candidate_clusters(graph)

    st.markdown(f"**Discovered {len(clusters)} suspicious connected clusters** across {graph.number_of_nodes()} total entities and {graph.number_of_edges()} observed relationships.")

    # Highlighting nodes
    hl_nodes = []
    if highlight_ring and clusters:
        for cl in clusters[:3]:
            hl_nodes.extend(cl["nodes"])

    # Visualizer
    vis = GraphVisualizer()
    fig = vis.generate_plotly_figure(graph, highlight_cluster=hl_nodes, title=f"Entity Subgraph ({graph.number_of_nodes()} Nodes)")
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    # Cluster Inspector Table
    if clusters:
        st.subheader("🔍 Detected Candidate Abuse Clusters")
        cl_data = []
        for c in clusters:
            cl_data.append({
                "Cluster ID": c["cluster_id"],
                "Accounts": c["num_accounts"],
                "Devices": c["num_devices"],
                "IPs": c["num_ips"],
                "Cards": c["num_cards"],
                "Internal Edges": c["num_edges"],
                "Tx Volume": c["total_tx_volume"]
            })
        st.dataframe(pd.DataFrame(cl_data), use_container_width=True, hide_index=True)
