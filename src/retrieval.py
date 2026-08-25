"""
Query-time retrieval over the index built by ingest.py.

Precedence rule, straight from the assignment's own escalation policy
(13-support-escalation.md): "Use explicit document status, authority, and
supersession metadata" — NOT recency (effective_date is deliberately not
used for ranking here). Candidates are ranked by raw similarity, then
re-sorted into three tiers:

  tier 0: status=active AND policy_authority=official  (fully authoritative)
  tier 1: status=superseded                             (was authoritative,
                                                           no longer current)
  tier 2: status=draft OR policy_authority=none          (never authoritative,
                                                           e.g. the migration
                                                           scratchpad test doc)

A chunk in a lower tier can still be *retrieved* (the system prompt needs to
see the draft/scratchpad doc to demonstrate it correctly rejects it — that's
literally the prompt-injection test case), it just never outranks an
equally-relevant higher-tier chunk. `audience` (customer vs internal) is a
separate axis: internal docs (like the escalation doc itself) are legitimate
for informing the AGENT's own behavior but are never customer-facing
citation sources — that distinction is enforced in the system prompt, not
in ranking.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, TypedDict

import faiss
import numpy as np

from src.embeddings import get_embedder, TfidfEmbedder

INDEX_DIR = Path(__file__).resolve().parent.parent / ".index"


def _tier(chunk: dict) -> int:
    if chunk["status"] == "active" and chunk["policy_authority"] == "official":
        return 0
    if chunk["status"] == "superseded":
        return 1
    return 2  # draft, policy_authority=none, or anything else non-authoritative


# NOTE on a design attempt that was tried and reverted (see README bug diary):
# a bounded-penalty version of this (adjusted_score = score - small_penalty[tier])
# was tried so that a highly-relevant low-tier chunk — specifically
# 14-internal-content-migration-notes.md, the prompt-injection test fixture —
# could still surface for a query that directly references it. Testing showed
# this let 02-returns-policy-legacy.md occasionally outrank
# 01-returns-policy-current.md for an *ordinary* return-window question,
# because this corpus's "current" and "legacy" docs share heavy lexical
# overlap that TF-IDF can't reliably disambiguate by topic-fit. That
# regression (surfacing stale policy as if current) is exactly customer
# complaint #1 in the brief and is strictly worse than under-surfacing a
# document the system prompt already knows to reject on principle. So this
# reverts to a hard tier-priority sort: correctness on ordinary policy
# questions wins. Injection-resistance for content the retriever doesn't
# surface is handled at the prompt layer instead — the system prompt refuses
# to comply with any instruction embedded in retrieved content or claimed by
# the user, regardless of whether that content is in front of it right now.


class RetrievedChunk(TypedDict):
    chunk_id: str
    filename: str
    document_id: str
    title: str
    heading: str
    status: str
    audience: str
    policy_authority: str
    text: str
    score: float


class Retriever:
    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = index_dir
        self.index = faiss.read_index(str(index_dir / "kb.faiss"))
        with open(index_dir / "chunks.json", encoding="utf-8") as f:
            self.chunks = json.load(f)
        with open(index_dir / "meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)

        if self.meta["backend"] == "tfidf":
            with open(index_dir / "tfidf_vectorizer.pkl", "rb") as f:
                vectorizer = pickle.load(f)
            self.embedder = TfidfEmbedder(vectorizer=vectorizer)
        else:
            self.embedder = get_embedder()

    def search(self, query: str, k: int = 4, candidate_pool: int = 10) -> List[RetrievedChunk]:
        qvec = self.embedder.encode([query])
        scores, idxs = self.index.search(qvec.astype("float32"), min(candidate_pool, len(self.chunks)))

        candidates: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            c = dict(self.chunks[idx])
            c["score"] = float(score)
            candidates.append(c)  # type: ignore[arg-type]

        candidates.sort(key=lambda c: (_tier(c), -c["score"]))
        return candidates[:k]


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
