# SentinelGraph — Final Competitive Hardening & Verification Audit Report

**Project:** **SentinelGraph — AI-Powered Early-Warning System for Coordinated Merchant Abuse**  
**Track:** Razorpay Buildathon 2026 • Track 02: AI Risk Manager  
**Safety Classification:** Strictly DEFENSE-ONLY  
**Evaluation Mode:** STRICTLY FROZEN HELD-OUT EVALUATION (Zero Data Leakage / Zero Lookahead)  

---

## A. Files Changed

1. [`data/schemas.py`](file:///c:/Users/surya/Desktop/razorpay_buildathon/data/schemas.py): Added `EvidenceSufficiency` enum (`SUFFICIENT_EVIDENCE`, `INSUFFICIENT_EVIDENCE`).
2. [`llm/explainer.py`](file:///c:/Users/surya/Desktop/razorpay_buildathon/llm/explainer.py): Implemented structured evidence reasoning, explicit abstention on insufficient evidence, deterministic guardrails, and traceable evidence mapping.
3. [`evaluation/ablation.py`](file:///c:/Users/surya/Desktop/razorpay_buildathon/evaluation/ablation.py): Integrated `RobustnessEvaluator` (Scenarios A-F) and `CostSensitivityEvaluator` into ablation suite output.
4. [`app/evaluation_view.py`](file:///c:/Users/surya/Desktop/razorpay_buildathon/app/evaluation_view.py): Reorganized into 4 distinct sections: Primary Frozen Test Results, Supplemental Robustness Scenarios, Cost Sensitivity Grid, and Diagnostic Curves.
5. [`app/risk_case.py`](file:///c:/Users/surya/Desktop/razorpay_buildathon/app/risk_case.py): Added Evidence Sufficiency badge, traceable evidence items `[E01]`, `[E02]`, and explicit AI assistant disclaimer.
6. [`app/replay.py`](file:///c:/Users/surya/Desktop/razorpay_buildathon/app/replay.py): Added 3 "Why SentinelGraph?" curated demo scenarios and clear 4-stage early-warning timeline banner.
7. [`app/methodology.py`](file:///c:/Users/surya/Desktop/razorpay_buildathon/app/methodology.py): Added Trust Boundary Diagram and 4 non-negotiable safety rules (`AI CANNOT OVERRIDE RISK ENGINE`, `AI CANNOT ACCESS FUTURE EVENTS`, etc.).
8. [`README.md`](file:///c:/Users/surya/Desktop/razorpay_buildathon/README.md): Enhanced executive summary headline, core metric callout, and added robustness evaluation matrix.

---

## B. New Files

1. [`evaluation/adversarial_cases.py`](file:///c:/Users/surya/Desktop/razorpay_buildathon/evaluation/adversarial_cases.py): Evaluates 6 defensive stress scenarios (Benign Shared Hardware, Flash Sale Burst, Individual Anomaly, Subtle Coordinated Ring, Gradual Formation, Insufficient Evidence).
2. [`evaluation/cost_sensitivity.py`](file:///c:/Users/surya/Desktop/razorpay_buildathon/evaluation/cost_sensitivity.py): Evaluates expected business cost grid across FP cost (₹50 - ₹500) and FN cost (₹500 - ₹10,000).
3. [`tests/test_ai_grounding.py`](file:///c:/Users/surya/Desktop/razorpay_buildathon/tests/test_ai_grounding.py): 5 automated tests for AI grounding, evidence traceability, and safe abstention.
4. [`tests/test_adversarial_robustness.py`](file:///c:/Users/surya/Desktop/razorpay_buildathon/tests/test_adversarial_robustness.py): 2 automated tests for defensive robustness boundaries and cost sensitivity monotonicity.
5. [`.gitignore`](file:///c:/Users/surya/Desktop/razorpay_buildathon/.gitignore): Excludes environment files, bytecode, and temporary logs.

---

## C. AI Contribution & Evidence Reasoning Layer

- **Deterministic Engine is Authoritative:** Mathematical risk fusion determines the final risk score and defensive recommendation (`ALLOW`, `REVIEW`, `HOLD`).
- **Structured Investigation Brief:** Generates risk hypothesis, multi-signal interpretation (behavioral, graph, temporal), and operational analyst guidance.
- **Traceable Evidence Mapping:** All statements reference deterministic pipeline evidence IDs (`[E01]`, `[E02]`).
- **Safe Abstention:** When telemetry is sparse or confidence is low, the AI explicitly abstains with `INSUFFICIENT_EVIDENCE — HUMAN REVIEW REQUIRED` without hallucinating fake fraud narratives.

---

## D. Supplemental Defensive Robustness Evaluation

| Scenario | Name | Key Condition | Assigned Risk | Action | Result |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **A** | Benign Shared Infrastructure | Family members sharing tablet/Wi-Fi | 8 / 100 | ALLOW | **PASSED** |
| **B** | Legitimate Traffic Burst | Flash sale volume surge (unlinked buyers) | 12 / 100 | ALLOW | **PASSED** |
| **C** | Individual Suspicious Account | High behavioral anomaly, 0 graph links | 88 / 100 | REVIEW | **PASSED** |
| **D** | Coordinated Low-Signal Ring | Subtle amounts + dense card/device reuse | 94 / 100 | HOLD | **PASSED** |
| **E** | Gradual Ring Formation | Progressive multi-account linkage | 48 / 100 | REVIEW | **PASSED** |
| **F** | Insufficient Evidence | Sparse telemetry, 0 evidence items | 25 / 100 | ALLOW (Abstain) | **PASSED** |

---

## E. Business Cost Sensitivity Analysis

- **Formula:** $\text{Total Cost} = FP \times \text{cost}_{FP} + FN \times \text{cost}_{FN}$
- **Primary Assumptions:** $\text{cost}_{FP} = \text{₹150}$, $\text{cost}_{FN} = \text{₹2,500}$
- **Primary Cost Reduction:** ₹72,500 lower expected prototype cost (-12.3%) on frozen test split.
- **Sensitivity Range:** Evaluated over FP cost [₹50, ₹100, ₹150, ₹250, ₹500] and FN cost [₹500, ₹1,000, ₹2,500, ₹5,000, ₹10,000]. Across all configurations where $\text{cost}_{FN} \ge \text{cost}_{FP}$, SentinelGraph consistently produces lower operational friction and financial exposure.
- *Disclaimer: Configured prototype assumptions — not realized production savings.*

---

## F. Frozen-Test Results (44,350 Events)

* **Dataset Split:** 206,962 Train / 44,349 Validation / 44,350 Frozen Test (Seed: `20260827`)
* **Confusion Matrix:** $TP = 908$, $FP = 72$, $TN = 43,168$, $FN = 202$ (Frauds: 1,110, Legitimate: 43,240)
* **Classification Performance:**
  * **Precision:** $\mathbf{0.9265}$ [95% CI: `0.9102`, `0.9419`]
  * **Recall:** $\mathbf{0.8180}$ [95% CI: `0.7938`, `0.8415`]
  * **F1 Score:** $\mathbf{0.8689}$ [95% CI: `0.8521`, `0.8847`]
  * **PR-AUC:** Stage A: `0.8836`, Stage B: `0.8807`, Stage C: `0.8663`
  * **False Positive Rate (FPR):** $\mathbf{0.17\%}$ on frozen test set
* **PR-AUC Interpretation:** *"Graph intelligence provides the primary classification gain. Temporal escalation is evaluated separately as an escalation and early-warning signal rather than a universal ranking improvement."*
* **Early-Warning Telemetry:**
  * **Ring Detection Rate:** `100.0%` (9 of 9 held-out test rings flagged)
  * **Predictive Lead Time:** `10.7 mins` (alert fired 10.7 min before multi-account coordination became observable)
  * **Post-Coordination Latency:** `0.0 mins` (immediate alert at moment of coordination observability)
  * **Alert Lead Time to 50% Volume:** `101.0 mins` (advance notice before 50% of final ring volume was reached)
  * **Exposure at First Alert:** `2.1%` (97.9% of eventual synthetic exposure had not yet occurred at first alert; early-detection indicator, not realized loss prevention)
* **Active Risk Cases:** `89` persisted cases in `cases/artifacts/cases.json`

---

## G. Zero-Lookahead & Leakage Verification

- **Point-in-Time Features:** All behavioral, graph, and temporal features computed strictly using events observed at or before transaction timestamp $t$.
- **Evaluation-Only Milestones:** $t_{\text{mat}}$ (50% financial volume) and final ring exposure are used strictly as post-hoc evaluation milestones and never as model features.
- **Validation-Only Tuning:** Threshold selection and probability calibration performed exclusively on the validation split.

---

## H. AI Grounding Verification

- Explainer strictly references supplied evidence IDs (`[E01]`, `[E02]`).
- Zero hallucination of entity IDs, account names, or financial amounts.
- Explicit abstention (`INSUFFICIENT_EVIDENCE`) when corroboration is weak.
- AI cannot alter mathematical risk scores or override risk decisions.

---

## I. API Verification

All 7 FastAPI endpoints tested and verified with HTTP 200 responses:
- `GET /health` -> `{"status": "healthy", "service": "SentinelGraph Risk Engine", "active_cases": 89}`
- `GET /cases` -> 89 cases returned
- `GET /cases/{case_id}` -> Case details, graph clusters, and evidence checklist returned
- `GET /clusters/{cluster_id}` -> Subgraph entity structures returned
- `GET /metrics` -> Frozen ablation and early-warning results returned
- `POST /explain/{case_id}` -> Deterministic evidence-grounded investigation brief generated
- `POST /score` -> Stateless real-time batch transaction scoring executed

---

## J. UI Verification

All 6 Streamlit views verified and operational:
1. **Executive Dashboard**: KPIs, Detection Journey, financial exposure by cluster, priority case queue.
2. **Live Replay Studio**: 3 curated "Why SentinelGraph?" demo scenarios, 4-stage early-warning timeline, dynamic graph growth.
3. **Risk Case Inspector**: Evidence sufficiency badge, 3-signal score breakdown, traceable evidence checklist, AI brief, defensive action console.
4. **Network Graph Explorer**: Zoomable Plotly 2D graph with cluster filters.
5. **Evaluation & Ablation View**: 4 clearly separated sections (Primary Frozen Results, Robustness Scenarios, Cost Sensitivity Grid, PR Curves).
6. **Methodology & Architecture**: Trust boundary flow, mathematical formulations, and defense-only safety guarantees.

---

## K. Test Results

- **`22 / 22 passed`** across all unit, integration, and robustness test modules (`pytest tests/` in 4.79s):
  - `test_adversarial_robustness.py` (2/2 passed)
  - `test_ai_grounding.py` (5/5 passed)
  - `test_audit_hardening.py` (4/4 passed)
  - `test_e2e_pipeline.py` (1/1 passed)
  - `test_evaluation.py` (2/2 passed)
  - `test_features.py` (1/1 passed)
  - `test_fusion.py` (2/2 passed)
  - `test_generator.py` (2/2 passed)
  - `test_graph.py` (1/1 passed)
  - `test_temporal.py` (2/2 passed)

---

## L. Remaining Limitations & Disclosures

1. **Synthetic Data**: Developed on a high-fidelity synthetic merchant simulator (295,661 events) with 8 distinct abuse topologies; production payment networks will introduce additional platform noise.
2. **Prototype Cost Parameters**: Business cost coefficients (₹150 / ₹2,500) and exposure multipliers (1.25x) are configured prototype parameters for transparent sensitivity analysis.
3. **Strictly Defense-Only**: SentinelGraph produces auditable early-warning intelligence, with zero real-money movement or offensive capability.

---

## M. Final Status

**FINAL SUBMISSION READY**

SentinelGraph has passed every audit requirement: zero data leakage, strict point-in-time safety, deterministic decision boundaries, evidence-grounded AI explanation with safe abstention, 6/6 defensive robustness scenarios passed, comprehensive cost sensitivity analysis, and 22/22 automated tests passing.
