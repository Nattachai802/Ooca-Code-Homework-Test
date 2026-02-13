from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TicketAnalysis(BaseModel):

    urgency: Literal["critical", "high", "medium", "low"] = Field(
        description="Ticket urgency level based on impact and time sensitivity"
    )
    sentiment: Literal["angry", "frustrated", "neutral", "positive"] = Field(
        description="Customer emotional tone detected from messages"
    )
    issue_type: str = Field(
        description="Category of the issue, e.g. 'billing', 'outage', 'bug', 'feature_request'"
    )
    product_area: str = Field(
        description="Product area affected, e.g. 'payments', 'infrastructure', 'ui/appearance'"
    )
    language: str = Field(
        description="Primary language detected in the ticket, e.g. 'en', 'th'"
    )
    summary: str = Field(
        description="One-line summary of the customer's issue"
    )


class SuggestedAction(BaseModel):
    action: Literal["auto_respond", "route_to_specialist", "escalate_to_human"] = Field(
        description="The recommended next action for this ticket"
    )
    suggested_reply: str = Field(
        description="Draft reply message to send to the customer, always provided regardless of action type"
    )
    reason: str = Field(
        description="Explanation of why this action was chosen"
    )
    priority_score: int = Field(
        ge=1, le=10,
        description="Priority score from 1 (lowest) to 10 (highest)"
    )
    auto_response: Optional[str] = Field(
        default=None,
        description="Draft response to send if action is auto_respond"
    )
    routing_department: Optional[str] = Field(
        default=None,
        description="Target department if action is route_to_specialist"
    )
    escalation_notes: Optional[str] = Field(
        default=None,
        description="Notes for the human agent if action is escalate_to_human"
    )


class TriageResult(BaseModel):
    """Complete triage result combining analysis and action."""

    ticket_id: str = Field(description="The ticket identifier")
    analysis: TicketAnalysis = Field(description="Extracted ticket analysis")
    action: SuggestedAction = Field(description="Recommended action")
    customer_context: str = Field(
        description="Summary of customer info relevant to this ticket"
    )
    kb_articles_used: list[str] = Field(
        default_factory=list,
        description="List of KB article IDs that were referenced"
    )

    @field_validator("customer_context", mode="before")
    @classmethod
    def coerce_customer_context(cls, v):
        """Convert dict to JSON string if LLM returns an object instead of string."""
        if isinstance(v, dict):
            return json.dumps(v, ensure_ascii=False)
        return v