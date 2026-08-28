"""
Tests for AI Investigation Grounding, Guardrails, and Evidence Sufficiency.
Verifies zero hallucination, strict evidence traceability, and safe abstention.
"""

import pytest
from data.schemas import RiskCase, RiskEvidence, SeverityLevel, DefensiveAction, EvidenceSufficiency
from llm.explainer import RiskExplainer


@pytest.fixture
def sample_risk_case():
    return RiskCase(
        case_id="1042",
        created_at="2026-08-27T12:00:00",
        ring_id="RING_01",
        severity=SeverityLevel.CRITICAL,
        risk_score=92.0,
        confidence=88.0,
        why_alert_fired="Dense hardware sharing across 6 accounts with rapid arrival velocity",
        behavioral_score=85.0,
        graph_score=95.0,
        temporal_score=90.0,
        num_accounts=6,
        num_devices=2,
        num_ips=2,
        num_payment_instruments=3,
        num_merchants=1,
        num_transactions=18,
        activity_growth_pct=340.0,
        earliest_event_time="2026-08-27T11:40:00",
        latest_event_time="2026-08-27T12:00:00",
        observed_exposure_inr=154200.0,
        estimated_exposure_inr=192750.0,
        evidence_items=[
            RiskEvidence(
                evidence_id="E01",
                evidence_type="graph_entity_sharing",
                title="Cross-Account Device Reuse",
                description="Device DEV_0042 shared across 6 distinct accounts.",
                metric_name="device_reuse_count",
                metric_value=6.0,
                expected_baseline="1.00",
                severity_contribution=40.0
            ),
            RiskEvidence(
                evidence_id="E02",
                evidence_type="temporal_arrival_burst",
                title="Rapid Account Creation Velocity",
                description="4 accounts created within 15-minute window.",
                metric_name="creation_burst_count",
                metric_value=4.0,
                expected_baseline="0.00",
                severity_contribution=35.0
            )
        ],
        recommended_action=DefensiveAction.HOLD
    )


def test_ai_explainer_references_only_supplied_evidence(sample_risk_case):
    """Verifies AI outputs only reference existing evidence IDs [E01], [E02]."""
    explainer = RiskExplainer()
    explanation = explainer.explain_case(sample_risk_case)

    assert explanation["evidence_sufficiency"] == EvidenceSufficiency.SUFFICIENT.value
    assert "E01" in explanation["formatted_markdown"]
    assert "E02" in explanation["formatted_markdown"]
    assert "E99" not in explanation["formatted_markdown"]  # No invented IDs


def test_ai_explainer_abstains_on_insufficient_evidence():
    """Verifies that sparse telemetry triggers explicit abstention without hallucinating."""
    sparse_case = RiskCase(
        case_id="SPARSE_01",
        created_at="2026-08-27T12:00:00",
        severity=SeverityLevel.LOW,
        risk_score=20.0,
        confidence=25.0,
        why_alert_fired="Single borderline transaction",
        behavioral_score=20.0,
        graph_score=10.0,
        temporal_score=5.0,
        num_accounts=1,
        num_devices=1,
        num_ips=1,
        num_payment_instruments=1,
        num_merchants=1,
        num_transactions=1,
        activity_growth_pct=0.0,
        earliest_event_time="2026-08-27T12:00:00",
        latest_event_time="2026-08-27T12:00:00",
        observed_exposure_inr=500.0,
        estimated_exposure_inr=625.0,
        evidence_items=[],  # Empty evidence items
        recommended_action=DefensiveAction.ALLOW
    )

    explainer = RiskExplainer()
    explanation = explainer.explain_case(sparse_case)

    assert explanation["evidence_sufficiency"] == EvidenceSufficiency.INSUFFICIENT.value
    assert "INSUFFICIENT_EVIDENCE" in explanation["formatted_markdown"]
    assert "HUMAN REVIEW REQUIRED" in explanation["formatted_markdown"]


def test_ai_cannot_alter_deterministic_score_or_decision(sample_risk_case):
    """Verifies that AI explanation cannot alter the mathematical risk score or decision."""
    explainer = RiskExplainer()
    explanation = explainer.explain_case(sample_risk_case)

    # Deterministic scores remain identical
    assert explanation["score_breakdown"]["final_risk_score"] == 92.0
    assert explanation["recommended_action"] == DefensiveAction.HOLD.value
    assert sample_risk_case.risk_score == 92.0


def test_ai_explainer_traceable_evidence_table(sample_risk_case):
    """Verifies that evidence sources table correctly maps all pipeline items."""
    explainer = RiskExplainer()
    explanation = explainer.explain_case(sample_risk_case)

    sources = explanation["evidence_sources"]
    assert len(sources) == 2
    assert sources[0]["id"] == "E01"
    assert sources[0]["metric"] == "device_reuse_count"
    assert sources[0]["value"] == 6.0


def test_ai_explainer_strictly_defensive():
    """Verifies that allowed recommendations are exclusively defensive."""
    for action in DefensiveAction:
        assert action.value in ["ALLOW", "REVIEW", "HOLD"]
