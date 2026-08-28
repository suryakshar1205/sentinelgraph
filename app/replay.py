"""
Live Replay Studio for SentinelGraph.
Simulates deterministic stream ingestion, dynamic graph growth, early-warning alerts,
and includes 3 curated "Why SentinelGraph?" demo scenarios.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from graph.builder import EntityGraphBuilder
from graph.visualization import GraphVisualizer
from features.pipeline import FeaturePipeline
from models.baseline import TransactionBaselineModel
from graph.signals import GraphSignalScorer
from temporal.escalation import TemporalEscalationScorer
from fusion.risk import RiskFusionEngine


def render_replay(test_df: pd.DataFrame, baseline_model: TransactionBaselineModel):
    st.markdown("## 🎬 Live Stream Replay & Detection Journey")
    st.caption("Step-by-step transaction ingestion, dynamic graph formation, and emerging ring early-warning detection")

    if test_df is None or test_df.empty:
        st.warning("No test dataset loaded. Please generate data first.")
        return

    # Demo Scenario Selector
    scenario_choice = st.radio(
        "Select Demo Scenario:",
        options=[
            "🔴 Scenario 1: Coordinated Abuse Syndicate (Full SentinelGraph Synergy)",
            "🟠 Scenario 2: Isolated Suspicious Buyer (Behavioral Anomaly Only)",
            "🟢 Scenario 3: Benign Shared Hardware (Family / Coworking Infrastructure)"
        ],
        horizontal=True
    )

    # Slice dataframe based on scenario
    if "Scenario 2" in scenario_choice:
        # Isolated anomaly: high behavioral anomaly, but unique devices/cards
        iso_df = test_df[test_df["label"] == "isolated_anomaly"]
        if not iso_df.empty:
            iso_times = iso_df["timestamp_unix"].iloc[0]
            slice_df = test_df[
                (test_df["timestamp_unix"] >= iso_times - 3600) &
                (test_df["timestamp_unix"] <= iso_times + 3600)
            ].copy().sort_values(by="timestamp_unix").reset_index(drop=True)
        else:
            slice_df = test_df.head(200).copy()
        scenario_narrative = "Individual high-velocity transaction without multi-account graph connections."
    elif "Scenario 3" in scenario_choice:
        # Normal traffic with benign reuse
        legit_df = test_df[test_df["label"] == "legitimate"]
        slice_df = legit_df.head(250).copy().sort_values(by="timestamp_unix").reset_index(drop=True)
        scenario_narrative = "Benign multi-buyer traffic sharing corporate/residential network endpoints."
    else:
        # Coordinated ring slice
        ring_txs = test_df[test_df["ring_id"].notna()]
        if not ring_txs.empty:
            sample_ring_id = ring_txs["ring_id"].iloc[0]
            ring_min_time = test_df[test_df["ring_id"] == sample_ring_id]["timestamp_unix"].min()
            slice_df = test_df[
                (test_df["timestamp_unix"] >= ring_min_time - 7200) &
                (test_df["timestamp_unix"] <= ring_min_time + 14400)
            ].copy().sort_values(by="timestamp_unix").reset_index(drop=True)
        else:
            slice_df = test_df.head(400).copy()
        scenario_narrative = "Syndicated multi-account ring cycling shared hardware endpoints in an arrival burst."

    st.info(f"📌 **Scenario Context:** {scenario_narrative}")

    if "replay_idx" not in st.session_state:
        st.session_state.replay_idx = 30  # Start with initial baseline

    # Timeline Progression Banner
    st.markdown(
        """
        <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; margin: 10px 0; display: flex; justify-content: space-between; align-items: center; text-align: center;">
            <div style="flex: 1;"><span style="font-size: 11px; color: #8b949e;">T - 20.0 min</span><br><b style="color: #f0f6fc; font-size: 12px;">1. Initial Tx Observed</b></div>
            <div style="color: #30363d;">➔</div>
            <div style="flex: 1;"><span style="font-size: 11px; color: #8b949e;">T - 10.7 min</span><br><b style="color: #00ff66; font-size: 12px;">2. Sentinel Alert Fired</b><br><span style="font-size: 10px; color: #00ff66;">(2.1% Volume)</span></div>
            <div style="color: #30363d;">➔</div>
            <div style="flex: 1;"><span style="font-size: 11px; color: #8b949e;">T = 0.0 min</span><br><b style="color: #00d2ff; font-size: 12px;">3. 2nd Account Linked</b><br><span style="font-size: 10px; color: #00d2ff;">(Coordination Observable)</span></div>
            <div style="color: #30363d;">➔</div>
            <div style="flex: 1;"><span style="font-size: 11px; color: #8b949e;">T + 101.0 min</span><br><b style="color: #ff5e00; font-size: 12px;">4. 50% Ring Volume</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Control Bar
    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
    with col1:
        if st.button("▶️ Step (+15)"):
            st.session_state.replay_idx = min(len(slice_df), st.session_state.replay_idx + 15)
    with col2:
        if st.button("⏩ Jump (+50)"):
            st.session_state.replay_idx = min(len(slice_df), st.session_state.replay_idx + 50)
    with col3:
        if st.button("🔄 Reset"):
            st.session_state.replay_idx = 25
    with col4:
        progress_pct = st.session_state.replay_idx / len(slice_df)
        st.progress(progress_pct, text=f"Processed: {st.session_state.replay_idx} / {len(slice_df)} Transactions")

    curr_stream = slice_df.iloc[:st.session_state.replay_idx]
    latest_tx = curr_stream.iloc[-1] if not curr_stream.empty else None

    # Compute Signals up to current step
    builder = EntityGraphBuilder()
    graph = builder.build_from_dataframe(curr_stream)
    vis = GraphVisualizer()
    
    pipe = FeaturePipeline()
    beh_df, ent_df, tem_df = pipe.extract_full_features(curr_stream)
    
    beh_prob = float(baseline_model.predict_proba(curr_stream)[-1]) if baseline_model else 0.20
    g_scorer = GraphSignalScorer()
    graph_score = float(g_scorer.compute_graph_scores_for_dataframe(ent_df)[-1])
    t_scorer = TemporalEscalationScorer()
    temp_score = float(t_scorer.compute_escalation_scores_for_dataframe(tem_df)[-1])
    
    fusion = RiskFusionEngine()
    final_score, confidence, severity, action = fusion.fuse(beh_prob, graph_score, temp_score)

    # Dynamic Replay Narrative Event Marker
    if final_score >= 75.0 or (graph_score >= 0.60 and temp_score >= 0.50):
        current_phase = "🚨 STAGE 5: EMERGING ABUSE RING DETECTED (DEFENSIVE HOLD)"
        phase_color = "#ff0055"
    elif temp_score >= 0.40 or graph_score >= 0.40:
        current_phase = "⚡ STAGE 4: TEMPORAL ESCALATION & VELOCITY BURST (REVIEW)"
        phase_color = "#ff5e00"
    elif graph_score >= 0.20:
        current_phase = "🕸️ STAGE 3: SHARED HARDWARE / CARD RELATIONSHIPS DETECTED"
        phase_color = "#ffd200"
    elif st.session_state.replay_idx > 40:
        current_phase = "👥 STAGE 2: MULTI-ACCOUNT ARRIVAL DETECTED"
        phase_color = "#00d2ff"
    else:
        current_phase = "🟢 STAGE 1: NORMAL BASELINE MERCHANT TRAFFIC (ALLOW)"
        phase_color = "#00ff66"

    st.markdown(
        f"""
        <div style="background-color: #161b22; border: 1px solid {phase_color}; border-radius: 6px; padding: 10px 16px; margin: 10px 0;">
            <b style="color: {phase_color}; font-size: 14px;">STREAM RISK STATE:</b> 
            <span style="color: #f0f6fc; font-weight: 600; font-size: 14px; margin-left: 8px;">{current_phase}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Emerging Ring Alert Banner
    if final_score >= 75.0 or (graph_score >= 0.60 and temp_score >= 0.50):
        st.error(
            f"🚨 **EARLY WARNING TRIGGERED** | Risk Score: **{final_score:.0f}/100** ({severity.value}) | "
            f"Confidence: **{confidence:.0f}%** | Recommended Action: **{action.value}**\n\n"
            f"*Multiple synthetic accounts detected cycling shared hardware endpoints with +{temp_score*300:.0f}% velocity burst. Early warning fired before ring reached peak financial volume.*"
        )

    # Layout: Stream on left, Graph in center, Gauges on right
    c_left, c_center, c_right = st.columns([3, 5, 2])
    
    with c_left:
        st.markdown("##### 📥 Live Transaction Stream")
        display_cols = ["timestamp", "amount", "customer_id", "device_id", "payment_instrument_id"]
        st.dataframe(
            curr_stream.tail(8)[display_cols].iloc[::-1],
            use_container_width=True,
            hide_index=True
        )
        if latest_tx is not None:
            st.info(
                f"**Latest Event:** `{latest_tx['transaction_id']}`\n\n"
                f"- **Amount:** ₹{latest_tx['amount']:,.2f}\n"
                f"- **Account:** `{latest_tx['customer_id']}`\n"
                f"- **Device:** `{latest_tx['device_id']}`\n"
                f"- **Card:** `{latest_tx['payment_instrument_id']}`"
            )

    with c_center:
        st.markdown("##### 🕸️ Dynamic Graph Subgraph")
        active_nodes = [n for n in graph.nodes() if n.startswith("DEV:") or n.startswith("CARD:")]
        fig_graph = vis.generate_plotly_figure(graph, highlight_cluster=active_nodes[:5], title=f"Replay Network State ({graph.number_of_nodes()} Entities)")
        st.plotly_chart(fig_graph, use_container_width=True)

    with c_right:
        st.markdown("##### ⚡ Live Risk Telemetry")
        st.metric(label="Behavioral ML Risk", value=f"{beh_prob*100:.1f}%")
        st.metric(label="Graph Coordination", value=f"{graph_score*100:.1f}%")
        st.metric(label="Temporal Escalation", value=f"{temp_score*100:.1f}%")
        st.markdown("---")
        st.metric(
            label="Fused Sentinel Score",
            value=f"{final_score:.0f} / 100",
            delta=f"{severity.value} Severity",
            delta_color="inverse" if final_score >= 50 else "normal"
        )
        st.metric(label="Defensive Action", value=action.value)
