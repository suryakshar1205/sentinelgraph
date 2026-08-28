"""
Pydantic API request/response schemas for SentinelGraph REST API.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from data.schemas import DefensiveAction, SeverityLevel


class ScoreRequest(BaseModel):
    transactions: List[Dict[str, Any]]


class ScoreResponse(BaseModel):
    scores: List[float]
    confidences: List[float]
    severities: List[str]
    actions: List[str]
    details: List[Dict[str, Any]]
