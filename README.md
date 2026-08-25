
# Aster & Row Support Agent

A reliability-focused RAG customer support agent built for the Aster & Row ecommerce take-home assignment.

The agent answers policy and product questions grounded in the `knowledge-base/`, looks up real order status through a secure tool, maintains context across turns, resists prompt injection, and includes a custom evaluation suite.

This project uses **Google's Gemini API** (`gemini-3.5-flash-lite`) through the official `google-genai` SDK.

---

## Features

* Reliable RAG retrieval over the provided knowledge base
* Authority-aware document selection
* Secure order lookup that never guesses order information
* Multi-turn conversation support
* Prompt-injection resistance
* Privacy-safe tool responses using an explicit allow-list
* Conflict detection between authoritative sources
* Debug logging and observability
* Deterministic component tests and behavioral evaluation

---

## Setup

```bash
git clone <your-repo-url>
cd support-agent

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install google-genai python-dotenv

cp .env.example .env
# Add your GEMINI_API_KEY to .env

python -m src.ingest
python cli.py
```

### Debug Mode

Use `--debug` to inspect the retrieval and tool trace for each turn:

```bash
python cli.py --debug
```

---

## Environment Variables

Create a `.env` file based on `.env.example`:

```env
GEMINI_API_KEY=your_gemini_api_key_here

# Optional
# GEMINI_MODEL=gemini-3.5-flash-lite
# EMBED_BACKEND=sentence-transformers
```

### Embedding Backends

```env
# Default local embedding model
EMBED_BACKEND=sentence-transformers
```

Or use the fully offline fallback:

```env
EMBED_BACKEND=tfidf
```

---

# Tech Stack

| Layer                   | Choice                                       | Why                                                                  |
| ----------------------- | -------------------------------------------- | -------------------------------------------------------------------- |
| **LLM**                 | Gemini API (`gemini-3.5-flash-lite`)         | Native tool use, high free-tier rate limits, and easy model swapping |
| **Embeddings**          | `sentence-transformers` (`all-MiniLM-L6-v2`) | Free local embeddings with no per-query API cost                     |
| **Embeddings Fallback** | TF-IDF (`scikit-learn`)                      | Fully offline fallback when the embedding model cannot be downloaded |
| **Vector Index**        | FAISS (`IndexFlatIP`)                        | Exact cosine similarity search is sufficient for the small corpus    |
| **Framework**           | None — direct `google-genai` SDK             | Keeps the retrieval and tool-use logic auditable                     |
| **Session Storage**     | In-memory dictionary                         | Simple and sufficient for the assignment                             |
| **Interface**           | CLI                                          | Fast to build, test, and demonstrate                                 |

---

# Architecture

```text
                         ┌───────────────────┐
                         │    User Message   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │   Session Store   │
                         │ Last 2 user turns │
                         └─────────┬─────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
          ┌──────────────────┐          ┌──────────────────┐
          │   RAG Retrieval  │          │   Gemini Agent   │
          │  FAISS Search +  │─────────►│ System Prompt +  │
          │ Authority Rerank │          │ Untrusted Context│
          └────────┬─────────┘          └─────────┬────────┘
                   │                              │
                   │                        Tool Required?
                   │                              │
                   │                       ┌──────┴──────┐
                   │                       │             │
                   │                      No            Yes
                   │                       │             │
                   │                       ▼             ▼
                   │                Final Response  ┌───────────────┐
                   │                                │ lookup_order  │
                   │                                └──────┬────────┘
                   │                                       │
                   │                                       ▼
                   │                              ┌────────────────┐
                   │                              │ Normalize ID   │
                   │                              │ Lookup Order   │
                   │                              │ Redact Data    │
                   │                              │ Suppress ETA   │
                   │                              └───────┬────────┘
                   │                                      │
                   └──────────────────────────────────────┘
                                                          │
                                                          ▼
                                                 Final Response
                                                          │
                                                          ▼
                                              ┌──────────────────┐
                                              │ Structured Trace │
                                              │ logs/trace.jsonl │
                                              └──────────────────┘
```

## Architecture Flow

1. The user sends a message.
2. Relevant previous user turns are collected from the session.
3. A retrieval query is constructed using the current message and conversation context.
4. Relevant knowledge-base passages are retrieved using FAISS.
5. Documents are reranked using authority metadata.
6. Retrieved content is provided to Gemini as **untrusted context**.
7. Gemini either:

   * Returns a grounded response, or
   * Calls `lookup_order` when real order information is required.
8. The order tool returns only customer-safe information.
9. Gemini generates the final response.
10. The interaction is recorded in `logs/trace.jsonl`.

---

# Knowledge Base Authority Model

Each document's front matter contains:

* `status`
* `audience`
* `policy_authority`

These fields determine how the retrieved content should be used.

| Document Type                      | Agent Behavior                                                          |
| ---------------------------------- | ----------------------------------------------------------------------- |
| `active` + `official` + `customer` | Can be cited to customers                                               |
| `superseded`                       | Historically true but never treated as current policy                   |
| `draft`                            | Never authoritative                                                     |
| `policy_authority: none`           | Never authoritative                                                     |
| `audience: internal`               | Can guide internal agent behavior but is never a customer-facing source |

## Prompt Injection Protection

All user messages, retrieved passages, and tool results are treated as **untrusted data**.

The knowledge base includes an internal migration document designed as a prompt-injection test fixture. Even if this document is retrieved, the agent:

* Does not follow instructions embedded inside retrieved documents
* Follows application-level instructions instead
* Never reveals system prompts or hidden instructions
* Never treats draft or internal content as customer-facing policy

## Conflict Handling

Two documents may both be active and authoritative while still disagreeing.

Instead of silently choosing one, the agent is instructed to:

1. Identify the conflict.
2. Clearly explain the disagreement.
3. Avoid inventing a resolution.
4. Recommend human assistance when appropriate.

---

# Secure Order Lookup

Order information is accessed only through the `lookup_order` tool.

The full `orders.json` file is **never sent to the LLM**. The model receives order information only when a lookup is required.

```text
Customer asks about an order
          │
          ▼
     Order ID provided?
      │           │
     No          Yes
      │           │
      ▼           ▼
 Ask for ID   Normalize ID
                  │
                  ▼
             Validate Format
                  │
                  ▼
             Lookup Record
                  │
                  ▼
       Build Customer-Safe Result
                  │
                  ▼
        Return Sanitized Data Only
```

## Privacy Protection

The raw order records contain fields such as:

* Customer name
* Email
* Address
* Risk score
* Warehouse notes
* Support tags

The tool uses an explicit **customer-safe allow-list**. Sensitive nested objects are structurally excluded from the outgoing tool result.

The model therefore does not receive:

* Customer email addresses
* Physical addresses
* Internal notes
* Risk scores
* Support tags
* Other internal-only information

## Order Reliability

The order tool:

* Normalizes lowercase order IDs
* Accepts harmless separator variations
* Handles malformed IDs safely
* Handles unknown IDs safely
* Uses the current order status as authoritative
* Never invents delivery estimates
* Suppresses stale delivery information for cancelled or returned orders
* Computes whether the cancellation window is open

---

# Multi-Turn Conversation

The agent maintains relevant context using an in-memory session store.

Retrieval uses the most recent **actual user turns** rather than simply slicing the most recent messages.

This prevents tool calls and tool results from evicting useful conversational context.

Example:

```text
User: Do you ship internationally?

Agent: Yes, we ship internationally...

User: What about Canada?

Agent: Understands that "Canada" refers to the
       previous international shipping discussion.
```

The agent does not carry unrelated information indefinitely and keeps sessions separate.

---

# Evaluation

The project includes two evaluation layers.

## 1. Deterministic Component Tests

These tests make no LLM calls and do not require an API key.

Run:

```bash
python evaluation/test_components.py
```

### Results

**26/27 passing**

```text
Retrieval: precedence (real corpus)   3/4
Order ID normalization                7/7
Order lookup: happy path              4/4
Order lookup: privacy                 2/2
Order lookup: no stale ETA            2/2
Order lookup: no fabricated ETA       1/1
Order lookup: exception → handoff     1/1
Order lookup: cancellation window     2/2
Order lookup: unknown/malformed IDs   3/3
Multi-turn retrieval context          1/1
```

---

## 2. Full Behavioral Evaluation

The behavioral evaluation makes live Gemini API calls.

Run:

```bash
python evaluation/run_eval.py
```

To save the results:

```bash
python evaluation/run_eval.py --save results/baseline.json
```

### Final Results

**15/22 passed**

```text
=== Results by category ===

abstention                1/1
agent-limits              1/1
conversation              1/2
groundedness              2/2
multi-source-grounding    1/2
privacy                   1/2
prompt-security           2/2
retrieval                 1/2
source-conflict           1/1
tool-reliability          3/4
tool-use                  1/3
```

---

# Bug Diary

## Bug 1 — Order IDs With Different Separators Were Rejected

### Reproduce

```python
lookup_order("ord1007")
lookup_order("ORD 1007")
```

Both were rejected as malformed even though they clearly referred to:

```text
ORD-1007
```

### Root Cause

The ID regex required a literal `-` between `ORD` and the digits.

### Fix

Added `ORDER_ID_LOOSE_RE` to canonicalize:

* Missing separators
* Spaces
* Underscores
* Lowercase IDs

before validation.

### Regression Test

Order ID normalization tests now cover multiple harmless variations.

---

## Bug 2 — PII Sanity Check Could Never Fail

### Problem

The original assertion was:

```python
INTERNAL_ONLY_FIELDS & record.keys() - INTERNAL_ONLY_FIELDS
```

### Root Cause

Python operator precedence caused the expression to always produce an empty result regardless of the input.

### Fix

The check was replaced with an explicit validation of the actual outgoing allow-listed `result` dictionary.

### Regression Test

Privacy tests now verify that sensitive fields never appear in the tool result.

---

## Bug 3 — Tool Calls Could Evict Relevant Conversation Context

### Reproduce

1. Ask a shipping question.
2. Perform an unrelated order lookup.
3. Ask a follow-up about the original shipping question.

The retrieval query lost the shipping context.

### Root Cause

`_retrieval_query()` sliced the last four **messages** rather than counting meaningful **user turns**.

A tool-heavy turn could consume the entire context window.

### Fix

The retrieval logic now walks backward through conversation history and counts actual user text turns while skipping hidden `tool_result` messages.

### Regression Test

Multi-turn retrieval tests verify that useful context survives intermediate tool calls.

---

## Bug 4 — Strict Tier Priority Buried a Highly Relevant Low-Tier Document

### Reproduce

A query directly asking about the migration note could fail to retrieve it because the strict authority tier ordering prevented lower-tier documents from appearing in the top results.

### Root Cause

A strict:

```text
(tier, -score)
```

sort prioritized authority so strongly that high semantic relevance could not overcome the tier.

### Decision

A proposed bounded score-penalty approach was reverted because it weakened normal document precedence checks.

Surfacing stale or non-authoritative policy as current was considered worse than under-surfacing a decoy document.

Prompt-injection resistance is therefore primarily handled at the prompt layer by treating all retrieved content as untrusted.

---

## Bug 5 — Gemini API Rate Limits and Payload Errors

### Reproduce

Running the evaluation suite produced:

* `429 RESOURCE_EXHAUSTED`
* `404 NOT_FOUND`
* Python attribute errors

### Root Cause

The original model selection had insufficient request limits for the 22-case evaluation suite.

The evaluation suite also injected mock dictionaries instead of native Gemini `Content` objects, breaking the history parser.

### Fix

* Switched to `gemini-3.5-flash-lite`
* Added exponential backoff to handle rate limits
* Updated `_retrieval_query()` to support both mock dictionaries and native Gemini objects

---

# Known Limitations

## Heuristic Behavioral Evaluation

The behavioral evaluation relies on strict keyword matching.

This can mark functionally correct Gemini responses as failures when the model uses unexpected wording.

### Production Improvement

Use semantic evaluation while retaining deterministic assertions for critical behavior such as:

* Tool invocation
* Tool arguments
* PII protection
* Source selection
* Abstention
* Forbidden disclosures

---

## Handoff Detection

The `_looks_like_handoff()` logic relies on keyword heuristics.

It may miss valid handoff recommendations that do not contain expected marker phrases.

### Production Improvement

Use structured handoff state or a dedicated model output field.

---

## Session Storage

Sessions are stored in memory.

Restarting the CLI clears all session history.

### Production Improvement

Use persistent session storage such as Redis or a database.

---

## Single-Process Architecture

The current implementation is suitable for the assignment but does not address:

* Horizontal scaling
* Distributed session management
* Production-wide rate limiting
* Production observability infrastructure

---

# AI Tools Used

Gemini was used during development for:

* API migration
* Debugging

## Example of an Incomplete AI Suggestion

During the Gemini SDK migration, an AI suggestion recommended appending tool results to session history using:

```text
role="tool"
```

This pattern caused a:

```text
400 INVALID_ARGUMENT
```

error with the Gemini SDK.

The corrected implementation appends tool results under:

```text
role="user"
```

This was corrected and regression-tested.

---

# Demo Video
<p align="center">

  <img src="https://github.com/user-attachments/assets/f7b65d7f-0813-4dd2-8a41-883f244b65ed" width="350"/>

</p> 



---

# Project Structure

```text
.
├── cli.py
├── README.md
├── requirements.txt
├── .env.example
│
├── src/
│   ├── agent.py
│   ├── ingest.py
│   ├── retrieval.py
│   ├── session.py
│   ├── logging_utils.py
│   └── tools/
│       └── orders.py
│
├── knowledge-base/
│   ├── 01-returns-policy-current.md
│   ├── 02-returns-policy-legacy.md
│   ├── ...
│   └── 14-internal-content-migration-notes.md
│
├── data/
│   ├── orders.json
│   └── orders-data-dictionary.md
│
├── evaluation/
│   ├── visible-cases.json
│   ├── test_components.py
│   └── run_eval.py
│
├── logs/
│   └── trace.jsonl
│
└── results/
    └── baseline.json
```

---

# Reliability Principles

This project was built around four core failure modes:

1. **Conflicting policy answers**
2. **Invented order information**
3. **Lost conversation context**
4. **Unsafe retrieved content**

The design prioritizes:

> **Reliability over breadth. Groundedness over guessing. Safe abstention over fabricated answers.**

---

# Quick Start

```bash
git clone <your-repo-url>
cd support-agent

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install google-genai python-dotenv

cp .env.example .env
# Add GEMINI_API_KEY

python -m src.ingest
python cli.py
```

**Debug mode:**

```bash
python cli.py --debug
```

**Run deterministic tests:**

```bash
python evaluation/test_components.py
```

**Run full evaluation:**

```bash
python evaluation/run_eval.py
```
