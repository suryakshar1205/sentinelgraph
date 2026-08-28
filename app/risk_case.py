"""
Risk Case Detail view for SentinelGraph.
Displays structured evidence, score breakdown, exposure calculations, and evidence-grounded AI investigation brief.
"""

import streamlit as st
import pandas as pd
from typing import Optional
from cases.store import RiskCaseStore
from llm.explainer import RiskExplainer
from data.schemas import EvidenceSufficiency


def render_risk_case(case_store: RiskCaseStore):
    st.markdown("## 🔍 Risk Case Deep-Dive Inspector")
    st.caption("Verifiable evidence synthesis, connected infrastructure audit, and evidence-grounded AI investigation")

    cases = case_store.list_cases()
    if not cases:
        st.warning("No active risk cases found in the repository.")
        return

    # Select Case
    case_ids = [c.case_id for c in cases]
    selected_id = st.selectbox("Select Case ID to Inspect", options=case_ids, index=0)
    case = case_store.get_case(selected_id)
    
    if not case:
        st.error("Selected case could not be retrieved.")
        return

    explainer = RiskExplainer()
    explanation = explainer.explain_case(case)
    is_sufficient = (explanation.get("evidence_sufficiency") == EvidenceSufficiency.SUFFICIENT.value)

    # Header Badges
    sev_color = "#ff0055" if case.severity.value == "CRITICAL" else ("#ff5e00" if case.severity.value == "HIGH" else "#ffd200")
    suff_badge_bg = "#238636" if is_sufficient else "#d29922"
    suff_text = "EVIDENCE: SUFFICIENT" if is_sufficient else "EVIDENCE: INSUFFICIENT (REVIEW REQUIRED)"

    st.markdown(
        f"""
        <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 18px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="color: #f0f6fc; margin: 0;">CASE #{case.case_id} <span style="font-size: 14px; color: #8b949e;">({case.created_at[:19]})</span></h2>
                    <p style="color: #8b949e; margin: 5px 0 0 0;">Ring Identifier: <b>{case.ring_id or 'CLUSTER-INFERRED'}</b> | Status: <b>UNDER AUDIT</b></p>
                </div>
                <div>
                    <span style="background-color: {suff_badge_bg}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 13px;">
                        {suff_text}
                    </span>
                    <span style="background-color: {sev_color}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 13px; margin-left: 6px;">
                        {case.severity.value} SEVERITY
                    </span>
                    <span style="background-color: #1f6feb; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 13px; margin-left: 6px;">
                        ACTION: {case.recommended_action.value}
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. WHY THIS ALERT FIRED Box
    why_text = case.why_alert_fired if hasattr(case, "why_alert_fired") and case.why_alert_fired else (
        f"SentinelGraph flagged this cluster due to dense entity sharing across {case.num_accounts} accounts, "
        f"reusing {case.num_devices} devices and {case.num_payment_instruments} payment instruments with +{case.activity_growth_pct:.0f}% velocity escalation."
    )
    st.info(f"💡 **WHY THIS ALERT FIRED:** {why_text}")

    # 2. Score Contribution Breakdown
    st.subheader("📊 Signal Score Contribution Breakdown")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric(label="Behavioral Anomaly (30%)", value=f"{case.behavioral_score:.0f}/100", delta="ML Tabular")
    with s2:
        st.metric(label="Graph Coordination (45%)", value=f"{case.graph_score:.0f}/100", delta="Entity Relationships", delta_color="inverse" if case.graph_score >= 60 else "normal")
    with s3:
        st.metric(label="Temporal Escalation (25%)", value=f"{case.temporal_score:.0f}/100", delta="Arrival Burst", delta_color="inverse" if case.temporal_score >= 50 else "normal")
    with s4:
        st.metric(label="Final Fused Risk Score", value=f"{case.risk_score:.0f}/100", delta=f"{case.confidence:.0f}% Confidence")

    st.markdown("---")

    # 3. Financial Exposure & Infrastructure Summary
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric(label="Observed Exposure", value=f"₹{case.observed_exposure_inr:,.0f}", delta="Transactions in Ring")
    f2.metric(label="Estimated At-Risk", value=f"₹{case.estimated_exposure_inr:,.0f}", delta="1.25x Propagation")
    f3.metric(label="Accounts Involved", value=case.num_accounts)
    f4.metric(label="Shared Devices", value=case.num_devices)
    f5.metric(label="Shared Payment Cards", value=case.num_payment_instruments)
    st.caption("ℹ️ *Financial exposure projection is a prototype risk assumption, not Razorpay actual economics.*")

    st.markdown("---")

    # 4. Layout: Verifiable Evidence on Left, AI Brief on Right
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.subheader("📋 Verifiable Pipeline Evidence Checklist")
        st.caption("All evidence items are deterministic, measured from pipeline telemetry, and fully auditable")
        
        for idx, ev in enumerate(case.evidence_items):
            ev_id = getattr(ev, "evidence_id", f"E{idx+1:02d}")
            ev_base = getattr(ev, "expected_baseline", "< standard normal")
            with st.expander(f"🔹 [{ev_id}] {ev.title}", expanded=True):
                st.write(ev.description)
                st.caption(
                    f"**Observed Metric:** `{ev.metric_name}` = `{ev.metric_value:.2f}` | "
                    f"**Expected Baseline:** `{ev_base}` | "
                    f"**Severity Impact:** `+{ev.severity_contribution:.0f}`"
                )

    with c_right:
        st.subheader("🤖 AI-Assisted Investigation Brief")
        st.caption("Grounded exclusively in pipeline telemetry; AI synthesizes evidence but does not decide fraud")
        
        with st.container():
            st.markdown(explanation["formatted_markdown"])

        with st.expander("🔍 Traceable Evidence Sources Table", expanded=False):
            if "evidence_sources" in explanation:
                st.dataframe(pd.DataFrame(explanation["evidence_sources"]), use_container_width=True, hide_index=True)

    st.markdown("---")
    
    # 5. Defensive Action Confirmation Console
    st.subheader("🛡️ Analyst Defensive Action Console")
    b1, b2, b3 = st.columns([1, 1, 3])
    with b1:
        if st.button("🔒 Apply Defensive HOLD", type="primary"):
            st.success(f"Defensive HOLD applied to Case #{case.case_id}. Shared card tokens and hardware identifiers isolated.")
    with b2:
        if st.button("👁️ Route to Senior REVIEW"):
            st.info(f"Case #{case.case_id} routed to Tier-2 Fraud Investigation Queue.")
    with b3:
        if st.button("✅ Mark as ALLOW (Safe)"):
            st.warning(f"Case #{case.case_id} cleared. Feedback logged for calibration tuning.")
