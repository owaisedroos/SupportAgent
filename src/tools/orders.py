"""
Order status lookup "tool" the agent calls via Gemini's tool-use API.

Constructs an explicit allow-listed result from the data-dictionary's 
"Customer-safe fields" list — customer.* and internal.* objects are structurally excluded.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from google.genai import types

ORDERS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "orders.json"

ORDER_ID_RE = re.compile(r"^ORD-\d{4,}$")
ORDER_ID_LOOSE_RE = re.compile(r"^ORD[\s\-_]?(\d{4,})$")

_NO_ETA_STATUSES = {"cancelled", "returned", "delivered"}
CANCELLATION_WINDOW_MINUTES = 30

_CUSTOMER_SAFE_TOP_LEVEL = {
    "order_id", "membership_tier", "placed_at", "status", "status_updated_at",
    "shipped_at", "delivered_at", "carrier", "tracking_number",
    "estimated_delivery", "customer_safe_message",
}


class OrderLookupResult(TypedDict, total=False):
    found: bool
    error: str | None
    order_id: str
    status: str
    membership_tier: str
    items: list
    placed_at: str
    status_updated_at: str
    shipped_at: str | None
    delivered_at: str | None
    carrier: str | None
    tracking_number: str | None
    estimated_delivery: str | None
    customer_safe_message: str
    requires_human_review: bool
    cancellation_window_open: bool


_dataset_cache: dict | None = None


def _load_dataset() -> dict:
    global _dataset_cache
    if _dataset_cache is None:
        with open(ORDERS_PATH, encoding="utf-8") as f:
            _dataset_cache = json.load(f)
    return _dataset_cache


def _orders_by_id() -> dict:
    return {o["order_id"]: o for o in _load_dataset()["orders"]}


def normalize_order_id(raw: str) -> str:
    cleaned = raw.strip().upper()
    m = ORDER_ID_LOOSE_RE.match(cleaned)
    if m:
        return f"ORD-{m.group(1)}"
    return cleaned


def _cancellation_window_open(record: dict) -> bool:
    if record.get("status") != "pending":
        return False
    try:
        snapshot_at = datetime.fromisoformat(_load_dataset()["snapshot_at"].replace("Z", "+00:00"))
        placed_at = datetime.fromisoformat(record["placed_at"].replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return False
    elapsed_minutes = (snapshot_at - placed_at).total_seconds() / 60
    return 0 <= elapsed_minutes <= CANCELLATION_WINDOW_MINUTES


def lookup_order(order_id: str) -> OrderLookupResult:
    if not order_id or not order_id.strip():
        return {"found": False, "error": "missing_order_id"}

    norm_id = normalize_order_id(order_id)

    if not ORDER_ID_RE.match(norm_id):
        return {"found": False, "error": "malformed_order_id", "order_id": norm_id}

    record = _orders_by_id().get(norm_id)
    if record is None:
        return {"found": False, "error": "not_found", "order_id": norm_id}

    result: OrderLookupResult = {
        "found": True,
        "error": None,
        "order_id": record["order_id"],
        "status": record["status"],
        "membership_tier": record.get("membership_tier"),
        "items": [
            {"name": i.get("name"), "quantity": i.get("quantity"), "final_sale": i.get("final_sale")}
            for i in record.get("items", [])
        ],
        "placed_at": record.get("placed_at"),
        "status_updated_at": record.get("status_updated_at"),
        "shipped_at": record.get("shipped_at"),
        "delivered_at": record.get("delivered_at"),
        "carrier": record.get("carrier"),
        "tracking_number": record.get("tracking_number"),
        "customer_safe_message": record.get("customer_safe_message"),
        "requires_human_review": record.get("status") == "exception",
        "cancellation_window_open": _cancellation_window_open(record),
    }

    if record["status"] in _NO_ETA_STATUSES:
        result["estimated_delivery"] = None
    else:
        result["estimated_delivery"] = record.get("estimated_delivery")

    leaked = set(result.keys()) - _CUSTOMER_SAFE_TOP_LEVEL - {
        "found", "error", "items", "requires_human_review", "cancellation_window_open",
    }
    assert not leaked, f"unexpected fields leaked into lookup_order result: {leaked}"

    return result


# Gemini Tool Declaration
LOOKUP_ORDER_DECLARATION = types.FunctionDeclaration(
    name="lookup_order",
    description=(
        "Look up the current status of a customer order by order ID. "
        "Returns status, dates, carrier/tracking, delivery estimate (only "
        "when meaningful), and a customer-safe status message. Never "
        "returns customer name, email, address, or any internal-only data."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "order_id": types.Schema(
                type=types.Type.STRING,
                description="The order ID, e.g. 'ORD-1007'. Case, whitespace, and separator don't matter.",
            )
        },
        required=["order_id"],
    ),
)

TOOL_SCHEMA = types.Tool(function_declarations=[LOOKUP_ORDER_DECLARATION])