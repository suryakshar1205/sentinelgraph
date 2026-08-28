"""
AI Explanation and Evidence Reasoning Layer for SentinelGraph.
Synthesizes verified pipeline evidence into analyst-grade investigation briefs.
Strictly adheres to deterministic guardrails with explicit abstention when evidence is insufficient.
"""

import os
import json
from typing import Dict, Any, Optional, List
from data.schemas import RiskCase, EvidenceSufficiency, DefensiveAction


class RiskExplainer:
    """
    Synthesizes structured risk evidence into analyst-grade investigation briefs.
    
    Guardrail Principles:
      1. Deterministic Engine is Authoritative: AI does not determine fraud or override the risk engine.
      2. Strictly Grounded: AI only references supplied evidence IDs ([E01], [E02], etc.) and telemetry.
      3. Zero Hallucination: AI never invents accounts, transactions, timestamps, or risk factors.
      4. Explicit Abstention: If evidence is insufficient (e.g. low signal / sparse cluster), AI abstains.
      5. Strictly Defensive: Recommendations are strictly ALLOW, REVIEW, or HOLD.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")

    def explain_case(self, case: RiskCase) -> Dict[str, Any]:
        """
        Generate structured investigation summary and evidence reasoning.
        Determines evidence sufficiency and performs deterministic synthesis.
        """
        # Determine evidence sufficiency
        is_sufficient, sufficiency_reason = self._evaluate_evidence_sufficiency(case)
        sufficiency_status = EvidenceSufficiency.SUFFICIENT if is_sufficient else EvidenceSufficiency.INSUFFICIENT

        return self._generate_structured_explanation(case, sufficiency_status, sufficiency_reason)

    def _evaluate_evidence_sufficiency(self, case: RiskCase) -> tuple[bool, str]:
        """
        Checks if available point-in-time evidence is sufficient to establish
        a strong coordinated abuse hypothesis.
        """
        if len(case.evidence_items) == 0:
            return False, "No distinct corroborating evidence items provided."
        
        if case.confidence < 40.0 and case.risk_score < 40.0:
            return False, "Low model confidence (<40%) with sparse corroborating signals."
            
        if case.num_accounts <= 1 and case.graph_score < 30.0:
            return False, "Single account observed with weak relationship telemetry; cannot infer coordination."

        return True, "Multi-signal corroboration established across behavioral, graph, and temporal telemetry."

    def _generate_structured_explanation(
        self,
        case: RiskCase,
        sufficiency_status: EvidenceSufficiency,
        sufficiency_reason: str
    ) -> Dict[str, Any]:
        """Creates a verified explanation referencing exact evidence IDs and metrics."""
        
        evidence_bullets = []
        evidence_sources = []
        for idx, item in enumerate(case.evidence_items):
            ev_id = getattr(item, "evidence_id", f"E{idx+1:02d}")
            evidence_bullets.append(
                f"**[{ev_id}] {item.title}**: {item.description} "
                f"(Baseline: `{item.expected_baseline}` | Severity Impact: `+{item.severity_contribution:.0f}`)"
            )
            evidence_sources.append({
                "id": ev_id,
                "type": item.evidence_type,
                "metric": item.metric_name,
                "value": item.metric_value,
                "baseline": item.expected_baseline
            })

        # When evidence is insufficient, explicitly abstain from asserting coordinated ring hypothesis
        if sufficiency_status == EvidenceSufficiency.INSUFFICIENT:
            risk_hypothesis = (
                "INCONCLUSIVE COORDINATION: Available telemetry does not provide sufficient multi-entity "
                "corroboration to establish a coordinated abuse operation."
            )
            behavioral_interp = f"Individual behavioral score is {case.behavioral_score:.0f}/100. Signal is isolated."
            graph_interp = f"Graph connectivity score is {case.graph_score:.0f}/100 across {case.num_accounts} account(s)."
            temporal_interp = f"Temporal escalation score is {case.temporal_score:.0f}/100."
            action_rationale = (
                f"Evidence sufficiency check failed: {sufficiency_reason}. "
                f"Automatic defensive escalation is withheld. Manual fraud analyst review is recommended "
                f"before taking high-friction actions."
            )
            rec_action = DefensiveAction.REVIEW.value if case.recommended_action == DefensiveAction.HOLD else case.recommended_action.value
            
            markdown_text = f"""#### ⚠️ Evidence Sufficiency Alert
- **Status:** `INSUFFICIENT_EVIDENCE — HUMAN REVIEW REQUIRED`  
- **Reason:** {sufficiency_reason}

#### 1. Risk Hypothesis & Executive Summary
> **{risk_hypothesis}**

#### 2. Available Evidence Checklist
""" + ("\n".join([f"- {b}" for b in evidence_bullets]) if evidence_bullets else "- *No formal corroborating evidence items.*") + f"""

#### 3. Multi-Signal Interpretation
- **Behavioral Signal:** {behavioral_interp}
- **Relationship Signal:** {graph_interp}
- **Temporal Signal:** {temporal_interp}

#### 4. Recommended Defensive Action
- **Action:** `{rec_action}`  
- **Analyst Guidance:** {action_rationale}
"""
        else:
            # Sufficient evidence: structured comprehensive brief
            risk_hypothesis = (
                f"ACTIVE COORDINATED ABUSE: High-confidence syndication across {case.num_accounts} customer accounts "
                f"sharing {case.num_devices} device(s) and {case.num_payment_instruments} payment instrument(s) "
                f"with +{case.activity_growth_pct:.0f}% velocity escalation."
            )
            behavioral_interp = (
                f"Behavioral ML model assigned risk score {case.behavioral_score:.0f}/100 based on transaction velocity, "
                f"amount deviations, and profile attributes."
            )
            graph_interp = (
                f"Dynamic entity graph identified dense cross-account linkage ({case.num_accounts} accounts connected "
                f"via {case.num_devices} device(s) and {case.num_payment_instruments} card(s)), producing graph score {case.graph_score:.0f}/100."
            )
            temporal_interp = (
                f"Temporal escalation scorer recorded score {case.temporal_score:.0f}/100 (+{case.activity_growth_pct:.0f}% burst "
                f"relative to merchant historical baseline)."
            )
            action_rationale = (
                f"Given the {case.severity.value} severity risk score ({case.risk_score:.0f}/100) and corroborated multi-signal confidence "
                f"({case.confidence:.0f}%), an immediate defensive `{case.recommended_action.value}` is advised to halt further "
                f"ring propagation and mitigate estimated at-risk exposure of INR {case.estimated_exposure_inr:,.2f}."
            )
            rec_action = case.recommended_action.value
            
            markdown_text = f"""#### 1. Risk Hypothesis & Executive Summary
> **{risk_hypothesis}**

#### 2. Primary Risk Evidence (Traceable to Pipeline Telemetry)
""" + "\n".join([f"- {b}" for b in evidence_bullets]) + f"""

#### 3. Multi-Signal Interpretation
- **Behavioral Signal:** {behavioral_interp}
- **Relationship / Graph Signal:** {graph_interp}
- **Temporal Escalation Signal:** {temporal_interp}

#### 4. Recommended Defensive Action
- **Action:** `{rec_action}`  
- **Operational Rationale:** {action_rationale}
"""

        return {
            "case_id": case.case_id,
            "evidence_sufficiency": sufficiency_status.value,
            "evidence_sufficiency_reason": sufficiency_reason,
            "risk_hypothesis": risk_hypothesis,
            "why_alert_fired": case.why_alert_fired,
            "score_breakdown": {
                "behavioral_score": case.behavioral_score,
                "graph_score": case.graph_score,
                "temporal_score": case.temporal_score,
                "final_risk_score": case.risk_score,
                "confidence": case.confidence
            },
            "evidence_bullets": evidence_bullets,
            "evidence_sources": evidence_sources,
            "behavioral_interpretation": behavioral_interp,
            "graph_interpretation": graph_interp,
            "temporal_interpretation": temporal_interp,
            "recommended_action": rec_action,
            "action_rationale": action_rationale,
            "formatted_markdown": markdown_text
        }
