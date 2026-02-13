from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from vector_store import KnowledgeBaseStore

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

_kb_store: KnowledgeBaseStore | None = None


def _get_kb_store() -> KnowledgeBaseStore:
    global _kb_store
    if _kb_store is None:
        _kb_store = KnowledgeBaseStore()
    return _kb_store


def fetch_customer_data(email: str) -> dict[str, Any]:
    customers_path = DATA_DIR / "customers.json"
    with open(customers_path, encoding="utf-8") as f:
        customers = json.load(f)

    customer = next((c for c in customers if c["email"] == email), None)
    if customer is None:
        return {"error": "not_found", "message": f"No customer found with email: {email}"}

    tiers_path = DATA_DIR / "plan_tiers.json"
    with open(tiers_path, encoding="utf-8") as f:
        tiers = json.load(f)

    plan_key = customer.get("plan", "free")
    tier_info = tiers.get(plan_key, {})

    return {
        **customer,
        "plan_details": {
            "label": tier_info.get("label", plan_key),
            "sla_hours": tier_info.get("sla_hours"),
            "priority": tier_info.get("priority", "low"),
            "support_channel": tier_info.get("support_channel", "email"),
            "features": tier_info.get("features", []),
            "auto_escalate": tier_info.get("auto_escalate", False),
        },
    }


def query_knowledge_base(query: str) -> list[dict[str, Any]]:
    store = _get_kb_store()
    results = store.search(query=query, n_results=3)
    logger.info("KB search for '%s' returned %d results.", query, len(results))
    return results


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fetch_customer_data",
            "description": (
                "Look up customer profile by email address. Returns customer info "
                "including plan type, region, usage history, and plan tier details "
                "(SLA, priority level, support channel). Always call this first to "
                "understand the customer context before making triage decisions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The customer's email address to look up.",
                    }
                },
                "required": ["email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_base",
            "description": (
                "Search the knowledge base for relevant FAQ articles, troubleshooting "
                "guides, and business guidelines. Returns matching articles with their "
                "recommended actions (auto_respond, escalate, route_to_specialist). "
                "Use this to find the appropriate resolution and action guidelines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of the customer's issue to search for.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_DISPATCH: dict[str, Any] = {
    "fetch_customer_data": fetch_customer_data,
    "query_knowledge_base": query_knowledge_base,
}