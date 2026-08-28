"""
Data schemas and entity representations for SentinelGraph.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class LabelType(str, Enum):
    LEGITIMATE = "legitimate"
    ISOLATED_ANOMALY = "isolated_anomaly"
    COORDINATED_RING = "coordinated_ring"


class AbusePatternType(str, Enum):
    NONE = "none"
    SHARED_DEVICE_RING = "pattern_a_shared_device"
    SHARED_IP_RING = "pattern_b_shared_ip"
    SHARED_PAYMENT_INSTRUMENT = "pattern_c_shared_payment_instrument"
    RAPID_ACCOUNT_CREATION = "pattern_d_rapid_account_creation"
    SYNCHRONIZED_ACTIVITY = "pattern_e_synchronized_activity"
    GRADUAL_RING_GROWTH = "pattern_f_gradual_growth"
    SUDDEN_ACTIVATION = "pattern_g_sudden_activation"
    MIXED_CLUSTER = "pattern_h_mixed_cluster"


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DefensiveAction(str, Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    HOLD = "HOLD"


class EvidenceSufficiency(str, Enum):
    SUFFICIENT = "SUFFICIENT_EVIDENCE"
    INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


class Transaction(BaseModel):
    transaction_id: str
    timestamp: str  # ISO-8601 string
    timestamp_unix: float
    amount: float
    merchant_id: str
    customer_id: str
    device_id: str
    ip_id: str
    payment_instrument_id: str
    location: str
    transaction_type: str = "payment"

    # Ground truth (not to be used in feature engineering)
    label: LabelType = LabelType.LEGITIMATE
    abuse_type: AbusePatternType = AbusePatternType.NONE
    ring_id: Optional[str] = None
    is_fraud: int = 0  # 0 or 1 for binary classification


class Customer(BaseModel):
    customer_id: str
    account_created_at: str
    account_created_unix: float
    customer_segment: str
    historical_transaction_count: int = 0
    historical_transaction_value: float = 0.0


class Device(BaseModel):
    device_id: str
    first_seen: str
    device_type: str


class IPNetwork(BaseModel):
    ip_id: str
    first_seen: str
    geographic_region: str
    asn: Optional[str] = None


class PaymentInstrument(BaseModel):
    payment_instrument_id: str
    instrument_type: str  # e.g., 'credit_card', 'upi', 'netbanking'
    first_seen: str


class Merchant(BaseModel):
    merchant_id: str
    merchant_segment: str
    baseline_volume: float
    baseline_risk: float


class RiskEvidence(BaseModel):
    evidence_id: str  # e.g., 'E01', 'E02'
    evidence_type: str
    title: str
    description: str
    metric_name: str
    metric_value: float
    expected_baseline: str
    severity_contribution: float


class RiskCase(BaseModel):
    case_id: str
    created_at: str
    ring_id: Optional[str] = None
    severity: SeverityLevel
    risk_score: float  # 0 - 100
    confidence: float  # 0 - 100
    why_alert_fired: str = ""
    
    # Sub-scores
    behavioral_score: float  # 0 - 100
    graph_score: float       # 0 - 100
    temporal_score: float    # 0 - 100
    
    # Connected entities summary
    num_accounts: int
    num_devices: int
    num_ips: int
    num_payment_instruments: int
    num_merchants: int
    num_transactions: int
    
    # Dynamics
    activity_growth_pct: float
    earliest_event_time: str
    latest_event_time: str
    
    # Financial Exposure
    observed_exposure_inr: float
    estimated_exposure_inr: float
    
    # Evidence & Recommendation
    evidence_items: List[RiskEvidence]
    recommended_action: DefensiveAction
    ai_explanation: Optional[Dict[str, Any]] = None
