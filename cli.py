#!/usr/bin/env python3
"""
Minimal CLI interface for the Aster & Row support agent (Gemini SDK).

Usage:
    python cli.py                 # interactive chat
    python cli.py --debug         # also print retrieval/tool trace per turn
    python cli.py --session foo   # pin a session id (default: random per run)

Type 'exit' or Ctrl-D to quit. Type 'new' to start a fresh session.
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid

from google import genai
from dotenv import load_dotenv

from src.session import SessionStore
from src.agent import run_turn

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="print retrieval + tool trace")
    parser.add_argument("--session", default=None, help="fixed session id")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set. Add GEMINI_API_KEY=your_key to your .env file.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    store = SessionStore()
    session_id = args.session or str(uuid.uuid4())[:8]

    print(f"Aster & Row support agent (session: {session_id}). Type 'exit' to quit, 'new' for a fresh session.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except EOFError:
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.lower() == "new":
            session_id = str(uuid.uuid4())[:8]
            print(f"(started new session: {session_id})\n")
            continue

        answer, trace = run_turn(store, session_id, user_input, client)

        if args.debug:
            print("\n--- trace ---")
            print("retrieved:", [(r["filename"], r["heading"], round(r["score"], 3), r["status"]) for r in trace["retrieved"]])
            print("tool_calls:", [(t["name"], t["args"]) for t in trace["tool_calls"]])
            print("handoff:", trace["handoff"])
            print("-------------\n")

        print(f"agent> {answer}\n")


if __name__ == "__main__":
    main()