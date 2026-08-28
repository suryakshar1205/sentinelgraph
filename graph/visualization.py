"""
Graph visualization converter for SentinelGraph.
Generates Plotly interactive 2D network graphs from NetworkX topologies.
"""

import networkx as nx
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List


class GraphVisualizer:
    """Produces rich interactive Plotly network figures."""

    COLOR_MAP = {
        "customer": "#00d2ff",       # Cyan
        "device": "#ffd200",         # Gold / Yellow
        "ip": "#b200ff",             # Purple
        "payment_instrument": "#ff5e00", # Orange
        "merchant": "#00ff66"        # Green
    }

    SYMBOL_MAP = {
        "customer": "circle",
        "device": "square",
        "ip": "diamond",
        "payment_instrument": "triangle-up",
        "merchant": "hexagon"
    }

    def generate_plotly_figure(
        self,
        subgraph: nx.Graph,
        highlight_cluster: Optional[List[str]] = None,
        title: str = "Entity Relationship Graph"
    ) -> go.Figure:
        """Create Plotly Figure from NetworkX graph."""
        if subgraph.number_of_nodes() == 0:
            fig = go.Figure()
            fig.update_layout(
                title="No entities in selected graph",
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117"
            )
            return fig

        # Compute Spring Layout positions
        pos = nx.spring_layout(subgraph, k=0.35, iterations=40, seed=2026)

        # 1. Edge traces
        edge_x = []
        edge_y = []
        for u, v, data in subgraph.edges(data=True):
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1.2, color="#30363d"),
            hoverinfo="none",
            mode="lines"
        )

        # 2. Node traces grouped by type
        node_traces = []
        highlight_set = set(highlight_cluster) if highlight_cluster else set()

        for ntype, color in self.COLOR_MAP.items():
            nodes_of_type = [
                n for n, d in subgraph.nodes(data=True) if d.get("type") == ntype
            ]
            if not nodes_of_type:
                continue

            node_x = []
            node_y = []
            node_text = []
            node_sizes = []
            node_colors = []

            for n in nodes_of_type:
                if n in pos:
                    x, y = pos[n]
                    node_x.append(x)
                    node_y.append(y)
                    
                    data = subgraph.nodes[n]
                    tx_c = data.get("tx_count", 1)
                    first_s = data.get("first_seen", "N/A")
                    
                    is_hl = n in highlight_set
                    size = 14 if not is_hl else 22
                    node_sizes.append(size)
                    
                    node_col = color if not is_hl else "#ff0055" # High-alert neon pink/red
                    node_colors.append(node_col)

                    hover = f"<b>Entity:</b> {n}<br><b>Type:</b> {ntype.upper()}<br><b>Transactions:</b> {tx_c}<br><b>First Seen:</b> {first_s}"
                    node_text.append(hover)

            trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                name=ntype.capitalize(),
                hoverinfo="text",
                text=node_text,
                marker=dict(
                    symbol=self.SYMBOL_MAP.get(ntype, "circle"),
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=1.5, color="#ffffff"),
                    opacity=0.9
                )
            )
            node_traces.append(trace)

        fig = go.Figure(
            data=[edge_trace] + node_traces,
            layout=go.Layout(
                title=dict(text=title, font=dict(color="#f0f6fc", size=16)),
                showlegend=True,
                hovermode="closest",
                margin=dict(b=20, l=10, r=10, t=45),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template="plotly_dark",
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#0b0f19",
                legend=dict(
                    bgcolor="rgba(13,17,23,0.8)",
                    bordercolor="#30363d",
                    borderwidth=1,
                    font=dict(color="#8b949e")
                )
            )
        )
        return fig
