"""
Methodology & Architecture View for SentinelGraph.
Presents technical design principles, mathematical formulation, trust boundaries, and defense-only guardrails.
"""

import streamlit as st


def render_methodology():
    st.markdown("## 🧠 Methodology & Architecture Deep-Dive")
    st.caption("Technical design principles, mathematical formulation, trust boundaries, and defense-only safety guarantees")

    # Callout: Why Not Just Fraud Detection?
    st.markdown(
        """
        <div style="background-color: #161b22; border-left: 4px solid #00d2ff; border-radius: 4px; padding: 14px 18px; margin-bottom: 20px;">
            <h4 style="color: #00d2ff; margin: 0 0 6px 0;">🎯 Judge FAQ: "Why Not Just Standard Transaction-Level Fraud Detection?"</h4>
            <p style="color: #f0f6fc; margin: 0; font-size: 14px; line-height: 1.6;">
                SentinelGraph is <b>not replacing</b> transaction-level fraud detection. The Baseline model in SentinelGraph represents what can be achieved with conventional transaction/customer behavior alone. 
                SentinelGraph measures the <b>incremental, empirical value of relationship and temporal escalation intelligence</b> on the exact same frozen held-out test set, capturing distributed syndicates that mimic organic individual buyer behavior.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
### 1. The SentinelGraph Operational Engine: DETECT → CONNECT → ESCALATE → EXPLAIN → ACT

```text
1. DETECT (Behavioral ML)
   Transaction-level amount deviations, cyclic hour profiles, and personal velocity anomalies.
        ↓
2. CONNECT (Dynamic Entity Graph)
   Bipartite multi-entity linkage (Device / Card / IP reuse) uncovering emerging syndicates.
        ↓
3. ESCALATE (Temporal Burst Engine)
   Point-in-time account arrival acceleration and rolling velocity surge (+1200% burst detection).
        ↓
4. EXPLAIN (Traceable Evidence Layer)
   Structured telemetry evidence synthesis mapping [E01], [E02] with safe abstention.
        ↓
5. ACT (Defense-Only Policy)
   Authoritative risk tiering: ALLOW (clean), REVIEW (queue), HOLD (restrict token).
```

---

### 1.1 🛡️ "Why Not Just Same IP = Fraud?" (Responsible Risk Modeling)

<div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 14px 18px; margin-bottom: 16px;">
    <h5 style="color: #ffd200; margin: 0 0 8px 0;">🌐 Shared Infrastructure Disambiguation Gate</h5>
    <p style="color: #f0f6fc; font-size: 13.5px; line-height: 1.6; margin: 0 0 10px 0;">
        A naive detector might flag every shared IP or Wi-Fi network as fraud, inflicting massive false positives on family households, offices, and coffee shops. SentinelGraph enforces a multi-signal corroboration gate:
    </p>
    <pre style="background-color: #0d1117; color: #58a6ff; padding: 10px; border-radius: 4px; font-size: 12.5px; margin: 0;">
Shared IP / Hardware Observed
        ↓
[ Behavioral ML: Normal ] + [ Device Reuse: None ] + [ Velocity Burst: 0% ]
        ↓
Multi-Signal Agreement Check: LOW CORROBORATION
        ↓
Result: Risk Score 8/100 → Action: ALLOW (Scenario A Validated)
    </pre>
</div>

---

### 2. Strict AI Safety & Trust Guardrails

<div style="background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 14px 18px; margin-bottom: 16px;">
    <h5 style="color: #00d2ff; margin: 0 0 8px 0;">🔒 Non-Negotiable System Safety Boundaries</h5>
    <ul style="color: #f0f6fc; font-size: 13.5px; line-height: 1.8; margin: 0; padding-left: 20px;">
        <li><b>AI CANNOT OVERRIDE RISK ENGINE:</b> The deterministic mathematical risk engine makes the authoritative decision boundary. The LLM is an investigator/explainer, not the decider.</li>
        <li><b>AI CANNOT ACCESS FUTURE EVENTS:</b> All inputs provided to the model and explainer are strictly constrained to timestamps <code>&le; t</code>.</li>
        <li><b>AI CANNOT EXECUTE MONEY MOVEMENT:</b> SentinelGraph produces strictly defensive risk telemetry (ALLOW / REVIEW / HOLD) and cannot execute financial transactions.</li>
        <li><b>AI CANNOT INVENT EVIDENCE:</b> The explainer only references verified pipeline evidence IDs (e.g. <code>[E01]</code>, <code>[E02]</code>) and explicitly abstains when evidence is insufficient.</li>
    </ul>
</div>

---

### 3. Leakage Prevention Architecture
To guarantee honest evaluation, SentinelGraph enforces strict point-in-time calculation rules:
- **Zero Lookahead**: For any transaction at timestamp $t$, feature calculations, graph edges, and temporal velocity are computed exclusively using events observed at or before $t$.
- **Frozen Held-Out Test Set**: The test set is split chronologically (70% Train, 15% Validation, 15% Test) and frozen before any hyperparameter tuning or threshold selection.
- **Validation-Only Tuning**: Thresholds for Alert triggers and probability calibration are determined strictly on the validation slice.

---

### 4. Mathematical Formulations

#### Dynamic Graph Coordination Score ($S_{\text{graph}}$)
Measures non-linear entity reuse and multi-account concentration:
$$S_{\text{graph}} = \min\left(1.0, \; \left(w_d \frac{\max(0, N_d - 1)}{4} + w_c \frac{\max(0, N_c - 1)}{2} + w_{\text{ip}} \frac{\max(0, N_{\text{ip}} - 2)}{10}\right) \times \mu_{\text{high}}\right)$$
where $N_d, N_c, N_{\text{ip}}$ are the distinct accounts linked to the device, payment card, and IP at timestamp $t$.

#### Temporal Escalation Score ($S_{\text{temp}}$)
Detects velocity acceleration and arrival bursts before ring saturation:
$$S_{\text{temp}} = w_{\text{dev}} \cdot \text{clip}\left(\frac{N_{\text{dev}, 1\text{h}}}{3}, 0, 1\right) + w_{\text{ip}} \cdot \text{clip}\left(\frac{\max(0, N_{\text{ip}, 1\text{h}}-1)}{6}, 0, 1\right) + w_{\text{cr}} \cdot \mathbb{I}(\text{new account})$$

#### Deterministic Calibrated Fusion
$$S_{\text{final}} = 100 \times \max\Big(S_{\text{beh}}, \; 0.30 \cdot S_{\text{beh}} + 0.45 \cdot S_{\text{graph}} + 0.25 \cdot S_{\text{temp}}, \; 0.50 \cdot S_{\text{graph}} + 0.50 \cdot S_{\text{temp}}\Big)$$

---

### 5. False-Positive & Business Cost Framework
$$\text{Total Expected Business Cost} = \text{FP} \times \text{Cost}_{\text{FP}} + \text{FN} \times \text{Cost}_{\text{FN}}$$
- **$\text{Cost}_{\text{FP}}$ (₹150)**: Estimated merchant friction & manual review overhead per legitimate transaction held.
- **$\text{Cost}_{\text{FN}}$ (₹2,500)**: Direct chargeback, merchandise loss, and scheme fees per unprevented fraud transaction.
- *Configured prototype assumptions — not realized production savings.*

---

### 6. Defense-Only Declaration
- **Action Vocabulary**: `ALLOW` (clear transaction), `REVIEW` (queue for analyst audit), `HOLD` (temporarily restrict card/device token).
- **Zero Offensive Tooling**: The system contains zero offensive fraud scripts, evasion helpers, or credential stuffing tools.

---

### 7. ⚖️ Judge Objection Panel: Transparent Answers to Technical Questions

<div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
    <h4 style="color: #00d2ff; margin: 0 0 12px 0;">Skeptical Judge Questions & Technical Defenses</h4>
    <details style="margin-bottom: 12px;">
        <summary style="cursor: pointer; font-weight: 600; color: #58a6ff;">1. "Isn't this just another fraud detector?"</summary>
        <p style="color: #f0f6fc; margin: 6px 0 0 14px; font-size: 13.5px; line-height: 1.6;">
            Traditional transaction-level models score individual events in isolation. Coordinated syndicates easily bypass them by rotating valid devices and cards with small amounts. SentinelGraph models the evolving bipartite relationship graph and arrival acceleration over time to identify emerging multi-account coordination before saturation.
        </p>
    </details>
    <details style="margin-bottom: 12px;">
        <summary style="cursor: pointer; font-weight: 600; color: #58a6ff;">2. "Why does Stage C PR-AUC show 0.8663 vs Stage B 0.8807?"</summary>
        <p style="color: #f0f6fc; margin: 6px 0 0 14px; font-size: 13.5px; line-height: 1.6;">
            We report this honestly. Temporal escalation is designed and evaluated as a point-in-time escalation and early-warning signal (~49x signal separation for coordinated rings: 0.4606 vs 0.0094) rather than a universal global ranking feature. Graph intelligence provides the primary classification improvement (F1 0.8689 vs Baseline 0.8530).
        </p>
    </details>
    <details style="margin-bottom: 12px;">
        <summary style="cursor: pointer; font-weight: 600; color: #58a6ff;">3. "Isn't the dataset synthetic?"</summary>
        <p style="color: #f0f6fc; margin: 6px 0 0 14px; font-size: 13.5px; line-height: 1.6;">
            Yes. Production payment network logs cannot be published due to PCI-DSS/PII regulations. We created a high-fidelity synthetic merchant simulator (295,661 events, seed 20260827) with 8 distinct abuse topologies to provide a controlled, leak-free benchmark for coordinated abuse.
        </p>
    </details>
    <details style="margin-bottom: 12px;">
        <summary style="cursor: pointer; font-weight: 600; color: #58a6ff;">4. "Can shared office Wi-Fi or family tablets trigger false HOLD actions?"</summary>
        <p style="color: #f0f6fc; margin: 6px 0 0 14px; font-size: 13.5px; line-height: 1.6;">
            No. Defensive robustness tests (Scenario A and Stress Test 4) confirm that shared hardware without behavioral anomaly or velocity burst produces low risk (score 8/100) and receives action ALLOW. Shared infrastructure alone is insufficient for escalation.
        </p>
    </details>
    <details style="margin-bottom: 12px;">
        <summary style="cursor: pointer; font-weight: 600; color: #58a6ff;">5. "Can the LLM hallucinate fraud reasons or override risk scores?"</summary>
        <p style="color: #f0f6fc; margin: 6px 0 0 14px; font-size: 13.5px; line-height: 1.6;">
            No. The LLM cannot determine fraud or alter the deterministic mathematical score. It summarizes only verified pipeline evidence items ([E01], [E02]) and explicitly abstains (<code>INSUFFICIENT_EVIDENCE — HUMAN REVIEW REQUIRED</code>) when evidence is sparse.
        </p>
    </details>
    <details style="margin-bottom: 12px;">
        <summary style="cursor: pointer; font-weight: 600; color: #58a6ff;">6. "Does the pipeline leak future information into current scoring?"</summary>
        <p style="color: #f0f6fc; margin: 6px 0 0 14px; font-size: 13.5px; line-height: 1.6;">
            Zero lookahead is strictly enforced. Features, graph edges, and temporal velocity at timestamp $t$ use only events occurring at or before $t$. Future volume milestones ($t_{\text{mat}}$) and final exposures are computed strictly post-hoc in evaluation modules.
        </p>
    </details>
    <details>
        <summary style="cursor: pointer; font-weight: 600; color: #58a6ff;">7. "How do you know the improvement is real and not overfitted?"</summary>
        <p style="color: #f0f6fc; margin: 6px 0 0 14px; font-size: 13.5px; line-height: 1.6;">
            The test set (44,350 events, 9 rings) is strictly frozen. All calibration and threshold optimization are conducted on the validation set. Stage A, Stage B, and Stage C are evaluated on the exact same held-out test events, accompanied by 95% bootstrap confidence intervals.
        </p>
    </details>
</div>

---

### 8. 🚀 Prototype → Production Engineering Roadmap

<div style="background-color: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
    <h5 style="color: #00ff66; margin: 0 0 10px 0;">Current Prototype vs Production Deployment Plan</h5>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; font-size: 13px;">
        <div style="background-color: #161b22; padding: 12px; border-radius: 6px;">
            <b style="color: #00d2ff;">Current Validated Prototype:</b>
            <ul style="margin: 6px 0 0 0; padding-left: 18px; color: #8b949e; line-height: 1.6;">
                <li>Controlled synthetic simulator (295,661 events)</li>
                <li>Frozen chronological held-out evaluation</li>
                <li>Point-in-time entity feature extractor (zero lookahead)</li>
                <li>Dynamic bipartite graph & temporal burst scoring</li>
                <li>Evidence-grounded brief generation with safe abstention</li>
                <li>Defense-only policy actions (ALLOW, REVIEW, HOLD)</li>
            </ul>
        </div>
        <div style="background-color: #161b22; padding: 12px; border-radius: 6px;">
            <b style="color: #ffd200;">Production Next Steps:</b>
            <ul style="margin: 6px 0 0 0; padding-left: 18px; color: #8b949e; line-height: 1.6;">
                <li>Validate against real historical merchant payment streams</li>
                <li>Calibrate probability curves using production base rates</li>
                <li>Continuous drift monitoring & distribution shift telemetry</li>
                <li>Analyst investigation feedback & active learning loops</li>
                <li>Live friction & intervention-cost calibration per merchant tier</li>
                <li>Production graph engine scaling via distributed graph DB</li>
            </ul>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
