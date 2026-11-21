"""Embedders used by every demo.

HashingEmbedder    : deterministic, offline, zero downloads (bag of words + bigrams, hashed).
                     Good enough to understand pipelines; NOT a semantic model.
get_embedder("bge"): real model through sentence-transformers (needs internet once).

Switch the whole repo with one env var:
    EMBEDDER=bge python 03_rag/01_rag_from_scratch.py
"""
import hashlib
import os
import re

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    return TOKEN.findall(text.lower())


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        toks = tokenize(text)
        for t in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        return v

    def encode(self, texts, normalize_embeddings: bool = True):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        M = np.stack([self._vec(t) for t in texts])
        if normalize_embeddings:
            M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
        return M[0] if single else M


def get_embedder(name=None):
    name = name or os.getenv("EMBEDDER", "hash")
    if name == "bge":
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("BAAI/bge-small-en-v1.5")
    return HashingEmbedder()
