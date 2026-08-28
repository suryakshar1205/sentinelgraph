"""
Risk case storage and query manager for SentinelGraph.
"""

import os
import json
from typing import List, Dict, Optional, Any
from data.schemas import RiskCase


class RiskCaseStore:
    """Manages storage, retrieval, and indexing of active risk cases."""

    def __init__(self, storage_path: str = "cases/artifacts/cases.json"):
        self.storage_path = storage_path
        self.cases: Dict[str, RiskCase] = {}
        self.load()

    def add_case(self, case: RiskCase):
        """Add or update a case."""
        self.cases[case.case_id] = case

    def get_case(self, case_id: str) -> Optional[RiskCase]:
        """Retrieve a single case by ID."""
        return self.cases.get(case_id)

    def list_cases(self) -> List[RiskCase]:
        """List all cases ordered by risk score descending."""
        return sorted(self.cases.values(), key=lambda c: c.risk_score, reverse=True)

    def save(self):
        """Save cases to disk."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = [case.model_dump() for case in self.cases.values()]
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Persisted {len(data)} risk cases to {self.storage_path}")

    def load(self):
        """Load cases from disk if existing."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                self.cases = {item["case_id"]: RiskCase.model_validate(item) for item in data}
            except Exception as e:
                print(f"Warning: could not load cases from {self.storage_path}: {e}")
                self.cases = {}
