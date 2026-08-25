"""
Pluggable embedding backend.

Primary path (recommended, used by default): sentence-transformers
(all-MiniLM-L6-v2), a small local model with real semantic embeddings.
No API key or network call needed at *query* time, only once to
download the model weights the first time it runs.

Fallback path: a TF-IDF vectorizer (scikit-learn), fully offline with
no model download at all. This exists for environments with no
internet access to huggingface.co (e.g. sandboxed CI). It's weaker
semantically (lexical overlap, not meaning) but keeps the whole
pipeline runnable end-to-end anywhere.

Select via the EMBED_BACKEND env var: "sentence-transformers" (default)
or "tfidf".
"""
from __future__ import annotations

import os
from typing import List

import numpy as np


class Embedder:
    """Common interface both backends implement."""

    dim: int

    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")


class TfidfEmbedder(Embedder):
    """
    Offline fallback. Fit once on the corpus at ingest time, then reused
    (pickled) for query-time transforms so query vectors live in the
    same space as the indexed document vectors.
    """

    def __init__(self, vectorizer=None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = vectorizer or TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=4096
        )
        self.dim = None

    def fit(self, texts: List[str]) -> np.ndarray:
        mat = self.vectorizer.fit_transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        self.dim = mat.shape[1]
        return mat

    def encode(self, texts: List[str]) -> np.ndarray:
        mat = self.vectorizer.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


def get_embedder() -> Embedder:
    backend = os.environ.get("EMBED_BACKEND", "sentence-transformers")
    if backend == "tfidf":
        return TfidfEmbedder()
    try:
        return SentenceTransformerEmbedder()
    except Exception as exc:  # network/model download unavailable, etc.
        print(
            f"[embeddings] sentence-transformers unavailable ({exc}); "
            "falling back to TF-IDF. Set EMBED_BACKEND=tfidf to silence this."
        )
        return TfidfEmbedder()
