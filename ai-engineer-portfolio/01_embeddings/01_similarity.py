"""STEP 1 - What an embedding is and why cosine similarity works.

Run:  python 01_embeddings/01_similarity.py
Then: EMBEDDER=bge python 01_embeddings/01_similarity.py   (real semantic model)
Interview point: normalized vectors => cosine == dot product == (2 - L2^2)/2 relationship.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from common.embedder import get_embedder
from common.data import DOCS

emb = get_embedder()
D = emb.encode(DOCS, normalize_embeddings=True)
q = emb.encode("how can I shrink vector storage", normalize_embeddings=True)

cos = D @ q
l2 = np.linalg.norm(D - q, axis=1)
# For unit vectors: ||a-b||^2 = 2 - 2*cos(a,b)  -> ranking by L2 == ranking by cosine
assert np.allclose(l2**2, 2 - 2 * cos, atol=1e-4)

for i in np.argsort(-cos)[:3]:
    print(f"cos={cos[i]:.3f}  l2={l2[i]:.3f}  {DOCS[i][:70]}")
print("\nKey facts: dim =", D.shape[1], "| memory for 1M vectors (float32) =",
      round(1_000_000 * D.shape[1] * 4 / 1e9, 2), "GB")
