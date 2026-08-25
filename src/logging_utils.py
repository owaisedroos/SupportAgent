"""
Structured trace logging for observability (README requirement #6).

Each turn produces one JSON line with: the user message, the retrieved
passages (with scores/metadata), tool calls and *sanitized* tool results,
the final response, and any fallback/handoff flag. Never logs secrets
(API keys are never part of the traced payload to begin with).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "trace.jsonl"


def log_turn(
    *,
    session_id: str,
    user_message: str,
    history_len: int,
    retrieved: List[Dict[str, Any]],
    tool_calls: List[Dict[str, Any]],
    final_response: str,
    handoff: bool,
    error: Optional[str] = None,
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id": session_id,
        "user_message": user_message,
        "history_len": history_len,
        "retrieved": [
            {
                "filename": r["filename"],
                "heading": r["heading"],
                "status": r["status"],
                "score": round(r["score"], 4),
            }
            for r in retrieved
        ],
        "tool_calls": tool_calls,
        "final_response": final_response,
        "handoff": handoff,
        "error": error,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
