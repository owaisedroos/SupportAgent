"""
Ingest the knowledge-base/*.md files into a local vector index.

- Parses YAML front matter (status, audience, policy_authority, document_id, etc.) from each file.
- Splits each document into chunks along ## section boundaries (falls back
  to the whole document if there are no ## headings) so retrieval returns
  focused passages, not entire files.
- Embeds every chunk and builds a FAISS IndexFlatIP (cosine similarity,
  since embeddings are normalized) over them.
- Persists the index + chunk metadata to disk so the CLI doesn't have to
  re-embed on every run.

Run directly: `python -m src.ingest`
"""
from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import faiss
import yaml

from src.embeddings import get_embedder, TfidfEmbedder

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge-base"
INDEX_DIR = Path(__file__).resolve().parent.parent / ".index"

FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    filename: str
    document_id: str
    title: str
    heading: str
    status: str            # active | superseded | draft
    audience: str          # customer | internal
    policy_authority: str  # official | none
    text: str


def parse_file(path: Path) -> List[Chunk]:
    raw = path.read_text(encoding="utf-8")
    m = FRONT_MATTER_RE.match(raw)
    if not m:
        raise ValueError(f"{path.name} is missing YAML front matter")
    meta = yaml.safe_load(m.group(1)) or {}
    body = m.group(2).strip()

    title = meta.get("title", path.stem)
    document_id = meta.get("document_id", path.stem)
    status = meta.get("status", "active")
    audience = meta.get("audience", "customer")
    policy_authority = meta.get("policy_authority", "official")

    # Split on H2 headings; keep the H1/preamble as its own chunk if present.
    parts = HEADING_RE.split(body)
    chunks: List[Chunk] = []

    if len(parts) == 1:
        # No ## headings at all -> one chunk for the whole doc.
        chunks.append(
            Chunk(
                chunk_id=f"{path.stem}::full",
                filename=path.name,
                document_id=document_id,
                title=title,
                heading=title,
                status=status,
                audience=audience,
                policy_authority=policy_authority,
                text=body,
            )
        )
        return chunks

    preamble = parts[0].strip()
    if preamble:
        chunks.append(
            Chunk(
                chunk_id=f"{path.stem}::intro",
                filename=path.name,
                document_id=document_id,
                title=title,
                heading=title,
                status=status,
                audience=audience,
                policy_authority=policy_authority,
                text=preamble,
            )
        )

    # parts = [preamble, heading1, body1, heading2, body2, ...]
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        section_body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        chunks.append(
            Chunk(
                chunk_id=f"{path.stem}::{re.sub(r'[^a-z0-9]+', '-', heading.lower()).strip('-')}",
                filename=path.name,
                document_id=document_id,
                title=title,
                heading=heading,
                status=status,
                audience=audience,
                policy_authority=policy_authority,
                text=f"{heading}\n{section_body}".strip(),
            )
        )
    return chunks


def build_index(kb_dir: Path = KB_DIR, index_dir: Path = INDEX_DIR) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        all_chunks.extend(parse_file(path))

    if not all_chunks:
        raise RuntimeError(f"No markdown files found in {kb_dir}")

    texts = [c.text for c in all_chunks]
    embedder = get_embedder()

    if isinstance(embedder, TfidfEmbedder):
        vectors = embedder.fit(texts)
        with open(index_dir / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(embedder.vectorizer, f)
        backend_name = "tfidf"
    else:
        vectors = embedder.encode(texts)
        backend_name = "sentence-transformers"

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(index_dir / "kb.faiss"))
    with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in all_chunks], f, indent=2)
    with open(index_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"backend": backend_name, "dim": dim, "n_chunks": len(all_chunks)}, f, indent=2)

    print(f"Indexed {len(all_chunks)} chunks from {len(list(kb_dir.glob('*.md')))} files "
          f"using backend={backend_name}, dim={dim}")


if __name__ == "__main__":
    build_index()
