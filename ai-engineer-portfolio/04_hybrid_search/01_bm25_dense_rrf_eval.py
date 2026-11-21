"""STEP 4 - BM25 from scratch + dense + Reciprocal Rank Fusion + EVALUATION (recall@k, MRR).
The point: never claim 'hybrid is better' without numbers. This file produces them.
Run:  python 04_hybrid_search/01_bm25_dense_rrf_eval.py
      EMBEDDER=bge python 04_hybrid_search/01_bm25_dense_rrf_eval.py   (compare with a real model)
"""
import math, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from common.embedder import get_embedder, tokenize
from common.data import DOCS, QUERIES


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tok = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.avgdl = sum(map(len, self.tok)) / self.N
        self.df = Counter(t for d in self.tok for t in set(d))
        self.tf = [Counter(d) for d in self.tok]

    def scores(self, q):
        out = np.zeros(self.N)
        for i, (tf, d) in enumerate(zip(self.tf, self.tok)):
            for t in tokenize(q):
                if t not in tf:
                    continue
                idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
                out[i] += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
        return out


def rrf(rank_lists, k=60):
    """score(d) = sum over lists of 1/(k+rank). Needs ranks only -> no score normalization problem."""
    score = {}
    for ranks in rank_lists:
        for r, doc in enumerate(ranks, 1):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + r)
    return sorted(score, key=score.get, reverse=True)


def evaluate(rank_fn, k=3):
    hits, rr = [], []
    for q, rel in QUERIES:
        ranked = rank_fn(q)
        hits.append(len(set(ranked[:k]) & rel) / len(rel))                      # recall@k
        first = next((i for i, d in enumerate(ranked, 1) if d in rel), None)
        rr.append(1 / first if first else 0.0)                                  # reciprocal rank
    return float(np.mean(hits)), float(np.mean(rr))


if __name__ == "__main__":
    emb = get_embedder()
    E = emb.encode(DOCS, normalize_embeddings=True)
    bm25 = BM25(DOCS)
    dense = lambda q: list(np.argsort(-(E @ emb.encode(q, normalize_embeddings=True))))
    sparse = lambda q: list(np.argsort(-bm25.scores(q)))
    hybrid = lambda q: rrf([dense(q), sparse(q)])
    print(f"{'method':<8} recall@3   MRR")
    for name, fn in [("dense", dense), ("bm25", sparse), ("hybrid", hybrid)]:
        r, m = evaluate(fn)
        print(f"{name:<8} {r:.3f}     {m:.3f}")
