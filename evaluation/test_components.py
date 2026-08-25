#!/usr/bin/env python3
"""
Deterministic, no-LLM-required tests for the pieces of the system that
don't need a model call to verify: retrieval precedence, order-ID
normalization, PII redaction, stale-ETA suppression, the deterministic
cancellation-window calculation, and unknown/malformed ID handling.

Run: python evaluation/test_components.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import get_retriever
from src.tools.orders import lookup_order, normalize_order_id, _CUSTOMER_SAFE_TOP_LEVEL
from src.session import Session
from src.agent import _retrieval_query

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if condition:
        passed += 1
    else:
        failed += 1


def main():
    print("=== Retrieval: precedence (real corpus) ===")
    r = get_retriever()

    results = r.search("how long to return an unused backpack", k=4)
    top = results[0]
    check(
        "current returns doc ranks above legacy for an ordinary return-window question",
        top["filename"] == "01-returns-policy-current.md" and top["status"] == "active",
        f"got top={top['filename']} status={top['status']}",
    )
    check(
        "no superseded/draft doc in top-4 for a question the current doc fully covers",
        all(c["status"] == "active" for c in results),
        f"statuses={[c['status'] for c in results]}",
    )

    conflict_results = r.search("can I put the entire breeze tumbler in the dishwasher", k=4)
    filenames = {c["filename"] for c in conflict_results}
    check(
        "both sides of the real dishwasher-safety conflict are retrievable (care guide + product card)",
        "11-product-care.md" in filenames and "12-breeze-tumbler-product-card.md" in filenames,
        f"got filenames={filenames}",
    )

    trailplus_results = r.search("my trailplus membership was active when I ordered, what is my return window", k=4)
    check(
        "TrailPlus-specific question surfaces the TrailPlus doc",
        any(c["filename"] == "09-trailplus-membership.md" for c in trailplus_results),
        f"got filenames={[c['filename'] for c in trailplus_results]}",
    )

    print("\n=== Order ID normalization (bug diary #1 regression) ===")
    check("lowercase id normalizes to uppercase", normalize_order_id("ord-1007") == "ORD-1007")
    check("whitespace is stripped", normalize_order_id("  ORD-1007  ") == "ORD-1007")
    check("no-dash id normalizes ('ord1007')", normalize_order_id("ord1007") == "ORD-1007")
    check("space separator normalizes ('ORD 1007')", normalize_order_id("ORD 1007") == "ORD-1007")
    check("underscore separator normalizes ('ORD_1007')", normalize_order_id("ORD_1007") == "ORD-1007")
    check("lookup_order succeeds end-to-end for 'ord1007'", lookup_order("ord1007")["found"] is True)
    check("truly malformed input is still rejected ('banana123')", lookup_order("banana123")["found"] is False)

    print("\n=== Order lookup: happy path (ORD-1007, the README's own example) ===")
    res = lookup_order("ord-1007")
    check("found=True (case-insensitive)", res["found"] is True)
    check("status is 'shipped'", res.get("status") == "shipped")
    check("carrier is UPS", res.get("carrier") == "UPS")
    check("estimated_delivery is 2026-08-22", res.get("estimated_delivery") == "2026-08-22")

    print("\n=== Order lookup: privacy ===")
    res = lookup_order("ORD-1007")
    leaked = set(res.keys()) - _CUSTOMER_SAFE_TOP_LEVEL - {
        "found", "error", "items", "requires_human_review", "cancellation_window_open",
    }
    check("no unexpected/internal fields present in lookup result", len(leaked) == 0, f"leaked keys={leaked}")
    check("customer name/email/address never appear in any result value", not any(
        needle in str(res) for needle in ["@example.test", "Ava Morgan", "King Street"]
    ))

    print("\n=== Order lookup: no stale ETA on cancelled/returned/delivered orders ===")
    res_cancelled = lookup_order("ORD-1004")
    check(
        "cancelled order does not surface its old estimated_delivery",
        res_cancelled.get("status") == "cancelled" and res_cancelled.get("estimated_delivery") is None,
        f"got status={res_cancelled.get('status')} eta={res_cancelled.get('estimated_delivery')}",
    )
    res_returned = lookup_order("ORD-1008")
    check(
        "returned order does not surface its old estimated_delivery",
        res_returned.get("status") == "returned" and res_returned.get("estimated_delivery") is None,
    )

    print("\n=== Order lookup: no fabricated ETA when carrier hasn't provided one ===")
    res_no_eta = lookup_order("ORD-1011")
    check(
        "shipped order (Canada Post) with null estimated_delivery stays null, not invented",
        res_no_eta.get("status") == "shipped" and res_no_eta.get("estimated_delivery") is None,
    )

    print("\n=== Order lookup: exception status flags human review ===")
    res_exception = lookup_order("ORD-1010")
    check(
        "exception status sets requires_human_review=True",
        res_exception.get("status") == "exception" and res_exception.get("requires_human_review") is True,
    )

    print("\n=== Order lookup: deterministic cancellation window (uses dataset snapshot_at) ===")
    res_pending_fresh = lookup_order("ORD-1001")  # placed 11:45, snapshot 12:00 -> 15 min elapsed
    check(
        "pending order placed 15 min ago has cancellation_window_open=True",
        res_pending_fresh.get("status") == "pending" and res_pending_fresh.get("cancellation_window_open") is True,
        f"got={res_pending_fresh.get('cancellation_window_open')}",
    )
    res_processing = lookup_order("ORD-1002")
    check(
        "processing order (past pending) has cancellation_window_open=False",
        res_processing.get("cancellation_window_open") is False,
    )

    print("\n=== Order lookup: unknown / malformed IDs ===")
    res_unknown = lookup_order("ORD-9999")
    check(
        "unknown ID returns found=False with a clear error, no guessed match",
        res_unknown == {"found": False, "error": "not_found", "order_id": "ORD-9999"},
    )
    res_malformed = lookup_order("banana")
    check("malformed ID is rejected safely, no exception", res_malformed["found"] is False and res_malformed["error"] == "malformed_order_id")
    res_empty = lookup_order("")
    check("empty/missing ID is rejected safely", res_empty["found"] is False and res_empty["error"] == "missing_order_id")

    print("\n=== Multi-turn retrieval context (bug diary #3 regression) ===")
    s = Session(session_id="regression-test")
    s.add_user("Do you ship internationally?")
    s.add_assistant([{"type": "text", "text": "Yes, we ship to Canada."}])
    s.add_user("Where is ORD-1007?")
    s.add_assistant([{"type": "tool_use", "id": "t1", "name": "lookup_order", "input": {"order_id": "ORD-1007"}}])
    s.add_tool_result("t1", '{"status": "shipped"}')
    s.add_assistant([{"type": "text", "text": "ORD-1007 has shipped."}])
    query = _retrieval_query(s, "What about Canada?")
    check(
        "shipping context survives an intervening tool-call turn",
        "international" in query.lower() or "canada" in query.lower(),
        f"got query={query!r}",
    )

    print(f"\n=== {passed}/{passed + failed} component checks passed ===")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
