"""
In-memory, per-session conversation history using Gemini Content objects.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List

from google.genai import types


@dataclass
class Session:
    session_id: str
    messages: List[types.Content] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append(
            types.Content(role="user", parts=[types.Part.from_text(text=text)])
        )

    def add_assistant(self, content) -> None:
        # Accepts both Gemini Content objects (from agent) and strings (from tests)
        if isinstance(content, str):
            self.messages.append(
                types.Content(role="model", parts=[types.Part.from_text(text=content)])
            )
        else:
            self.messages.append(content)

    def add_tool_result(self, tool_use_id: str, content: str) -> None:
        # Helper specifically for test_components.py to simulate tool returns
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"result": content}
            
        self.messages.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="lookup_order",
                        response=parsed
                    )
                ]
            )
        )


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def get(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)