"""Recommendation output contract shared by every backend and the control plane.

Never add a field here that only makes sense for one backend (e.g. an
MLX-specific token id) — this schema is the seam between the model layer and
everything downstream, and it has to mean the same thing regardless of which
ModelBackend produced it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Action(str, Enum):
    APPROVE = "approve"
    DECLINE = "decline"
    ESCALATE_L2 = "escalate_l2"
    REQUEST_MORE_INFO = "request_more_info"


class Recommendation(BaseModel):
    alert_id: str
    action: Action
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    policy_flags: list[str] = Field(default_factory=list)
    rationale: str = Field(max_length=1000)
