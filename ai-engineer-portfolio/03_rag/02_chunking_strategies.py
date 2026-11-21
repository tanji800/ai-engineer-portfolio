"""STEP 3b - Chunking: fixed vs sentence vs recursive (+ overlap). Chunking is where most RAG quality is won/lost.
Run:  python 03_rag/02_chunking_strategies.py"""
import re

TEXT = ("Vector search finds similar items. It uses embeddings. "
        "Approximate indexes trade recall for speed.\n\n"
        "HNSW builds a graph. IVF builds clusters. PQ compresses vectors.\n\n"
        "RAG retrieves passages and asks an LLM to answer from them. Good chunking keeps ideas intact.")


def fixed(text, size=80, overlap=20):
    return [text[i:i + size] for i in range(0, len(text), size - overlap)]


def sentences(text, per_chunk=2):
    s = re.split(r"(?<=[.!?])\s+", text.replace("\n\n", " "))
    return [" ".join(s[i:i + per_chunk]) for i in range(0, len(s), per_chunk)]


def recursive(text, max_len=110, seps=("\n\n", ". ", " ")):
    """Split on the biggest separator first, recurse only on pieces that are still too long."""
    if len(text) <= max_len or not seps:
        return [text.strip()] if text.strip() else []
    out = []
    for part in text.split(seps[0]):
        out += recursive(part, max_len, seps[1:])
    return out


for name, fn in [("fixed", fixed), ("sentence", sentences), ("recursive", recursive)]:
    chunks = fn(TEXT)
    print(f"\n== {name}: {len(chunks)} chunks")
    for c in chunks[:4]:
        print("  -", repr(c))
print("\nRule of thumb: start recursive ~300-500 tokens, 10-15% overlap; tune with retrieval metrics (see 04_hybrid_search).")
