"""STEP 3a - RAG with zero frameworks: chunk -> embed -> retrieve -> prompt -> generate.
Run:  python 03_rag/01_rag_from_scratch.py          (offline stub LLM)
      ANTHROPIC_API_KEY=... EMBEDDER=bge python 03_rag/01_rag_from_scratch.py   (real)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from common.embedder import get_embedder
from common.llm import llm
from common.data import DOCS

emb = get_embedder()


def chunk(text, size=300, overlap=50):
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text), 1), step)]


class RAG:
    def __init__(self, docs):
        self.chunks, self.meta = [], []
        for di, d in enumerate(docs):
            for c in chunk(d):
                self.chunks.append(c); self.meta.append(di)     # keep source id for citations
        self.E = emb.encode(self.chunks, normalize_embeddings=True)

    def retrieve(self, q, k=3):
        s = self.E @ emb.encode(q, normalize_embeddings=True)
        top = np.argsort(-s)[:k]
        return [(self.chunks[i], self.meta[i], float(s[i])) for i in top]

    def answer(self, q, k=3):
        hits = self.retrieve(q, k)
        ctx = "\n".join(f"[{m}] {c}" for c, m, _ in hits)
        prompt = (f"Answer ONLY from the context and cite ids like [3]. "
                  f"If the answer is not in the context say 'not found'.\n\nContext:\n{ctx}\n\nQuestion: {q}")
        return llm(prompt), hits


if __name__ == "__main__":
    rag = RAG(DOCS)
    ans, hits = rag.answer("How can I reduce the memory used by vectors?")
    for c, m, s in hits:
        print(f"{s:.3f} [{m}] {c[:80]}")
    print("\nANSWER:", ans)
