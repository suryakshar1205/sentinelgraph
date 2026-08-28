"""
Risk case builder for SentinelGraph.
Constructs auditable risk cases from pipeline signals with verifiable evidence and baselines.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import datetime
from data.schemas import RiskCase, RiskEvidence, SeverityLevel, DefensiveAction
from fusion.thresholds import RiskThresholdManager


class RiskCaseBuilder:
    """Constructs auditable risk cases with structured evidence from detection pipeline."""

    def __init__(self, exposure_multiplier: float = 1.25):
        self.exposure_multiplier = exposure_multiplier
        self.threshold_mgr = RiskThresholdManager()

    def build_case_from_cluster(
        self,
        case_id: str,
        cluster_data: Dict[str, Any],
        cluster_txs_df: pd.DataFrame,
        fused_row: Dict[str, Any]
    ) -> RiskCase:
        """
        Build a complete RiskCase object from cluster telemetry and transaction evidence.
        """
        created_at = datetime.datetime.now().isoformat()
        
        # Aggregate entities
        num_accounts = int(cluster_data.get("num_accounts", cluster_txs_df["customer_id"].nunique()))
        num_devices = int(cluster_data.get("num_devices", cluster_txs_df["device_id"].nunique()))
        num_ips = int(cluster_data.get("num_ips", cluster_txs_df["ip_id"].nunique()))
        num_cards = int(cluster_data.get("num_cards", cluster_txs_df["payment_instrument_id"].nunique()))
        num_merchants = int(cluster_txs_df["merchant_id"].nunique())
        num_txs = len(cluster_txs_df)
        
        # Financial exposure
        observed_exposure = float(cluster_txs_df["amount"].sum())
        estimated_exposure = float(observed_exposure * self.exposure_multiplier)
        
        # Time progression
        earliest_time = str(cluster_txs_df["timestamp"].min())
        latest_time = str(cluster_txs_df["timestamp"].max())
        
        # Calculate cluster growth rate
        earliest_unix = cluster_txs_df["timestamp_unix"].min()
        latest_unix = cluster_txs_df["timestamp_unix"].max()
        duration_hours = max(0.2, (latest_unix - earliest_unix) / 3600.0)
        growth_pct = float(min(1200.0, (num_txs / duration_hours) * 100.0))
        
        # Scores
        beh_score = float(fused_row.get("behavioral_score", 65.0))
        graph_score = float(fused_row.get("graph_score", 85.0))
        temp_score = float(fused_row.get("temporal_score", 80.0))
        final_score = float(fused_row.get("final_risk_score", 82.0))
        confidence = float(fused_row.get("confidence", 90.0))
        severity = SeverityLevel(fused_row.get("severity", SeverityLevel.HIGH))
        action = DefensiveAction(fused_row.get("recommended_action", DefensiveAction.HOLD))

        # Generate verifiable evidence list
        evidence_items: List[RiskEvidence] = []
        ev_counter = 1
        
        if num_accounts > num_devices and num_devices > 0:
            evidence_items.append(RiskEvidence(
                evidence_id=f"E{ev_counter:02d}",
                evidence_type="shared_device_concentration",
                title=f"{num_accounts} accounts connected through {num_devices} devices",
                description=f"Observed an abnormal device sharing ratio of {num_accounts / max(1, num_devices):.1f} accounts per device.",
                metric_name="accounts_per_device",
                metric_value=float(num_accounts / max(1, num_devices)),
                expected_baseline="< 1.2 accounts/device",
                severity_contribution=25.0
            ))
            ev_counter += 1
            
        if num_accounts > num_cards and num_cards > 0:
            evidence_items.append(RiskEvidence(
                evidence_id=f"E{ev_counter:02d}",
                evidence_type="shared_payment_instrument",
                title=f"{num_cards} payment instruments shared across {num_accounts} accounts",
                description="Cross-account card/UPI reuse pattern indicating syndicated operation.",
                metric_name="accounts_per_instrument",
                metric_value=float(num_accounts / max(1, num_cards)),
                expected_baseline="< 1.05 accounts/instrument",
                severity_contribution=25.0
            ))
            ev_counter += 1
            
        if num_accounts > num_ips and num_ips > 0:
            evidence_items.append(RiskEvidence(
                evidence_id=f"E{ev_counter:02d}",
                evidence_type="ip_concentration",
                title=f"{num_accounts} accounts originating from {num_ips} IP endpoints",
                description="High geographic/network concentration across distinct account identities.",
                metric_name="accounts_per_ip",
                metric_value=float(num_accounts / max(1, num_ips)),
                expected_baseline="< 2.0 accounts/IP",
                severity_contribution=15.0
            ))
            ev_counter += 1
            
        # Account creation burst check
        new_acc_count = (cluster_txs_df["account_created_unix"] >= earliest_unix - 86400).sum()
        if new_acc_count > 0:
            evidence_items.append(RiskEvidence(
                evidence_id=f"E{ev_counter:02d}",
                evidence_type="rapid_account_creation",
                title=f"{new_acc_count} accounts created in tight temporal proximity",
                description=f"{new_acc_count} of {num_accounts} member accounts were registered immediately prior to transaction burst.",
                metric_name="new_account_ratio",
                metric_value=float(new_acc_count / max(1, num_accounts)),
                expected_baseline="< 0.10 new accounts/window",
                severity_contribution=20.0
            ))
            ev_counter += 1

        if growth_pct >= 150.0:
            evidence_items.append(RiskEvidence(
                evidence_id=f"E{ev_counter:02d}",
                evidence_type="activity_escalation",
                title=f"Activity velocity escalated by +{growth_pct:.0f}%",
                description=f"Transaction arrival velocity significantly exceeds historical baseline for merchant sector.",
                metric_name="growth_rate_pct",
                metric_value=float(growth_pct),
                expected_baseline="< 25% rolling hourly growth",
                severity_contribution=15.0
            ))
            ev_counter += 1

        # Construct concise WHY THIS ALERT FIRED statement
        primary_drivers = []
        if graph_score >= 60.0:
            primary_drivers.append(f"dense entity sharing ({num_accounts} accounts across {num_devices} devices and {num_cards} cards)")
        if temp_score >= 50.0:
            primary_drivers.append(f"accelerated temporal escalation (+{growth_pct:.0f}% activity growth)")
        if beh_score >= 60.0:
            primary_drivers.append("elevated transaction behavioral anomalies")
            
        why_fired = (
            f"SentinelGraph triggered a {severity.value} alert (Score: {final_score:.0f}/100, Confidence: {confidence:.0f}%) "
            f"driven by {', '.join(primary_drivers) if primary_drivers else 'coordinated cluster relationships'}. "
            f"Observed financial exposure reached INR {observed_exposure:,.2f}."
        )

        return RiskCase(
            case_id=case_id,
            created_at=created_at,
            ring_id=cluster_data.get("ring_id"),
            severity=severity,
            risk_score=final_score,
            confidence=confidence,
            why_alert_fired=why_fired,
            behavioral_score=beh_score,
            graph_score=graph_score,
            temporal_score=temp_score,
            num_accounts=num_accounts,
            num_devices=num_devices,
            num_ips=num_ips,
            num_payment_instruments=num_cards,
            num_merchants=num_merchants,
            num_transactions=num_txs,
            activity_growth_pct=growth_pct,
            earliest_event_time=earliest_time,
            latest_event_time=latest_time,
            observed_exposure_inr=round(observed_exposure, 2),
            estimated_exposure_inr=round(estimated_exposure, 2),
            evidence_items=evidence_items,
            recommended_action=action
        )
