"""STEP 2d - Filtered vector search: the #1 production gotcha.
Post-filtering (search top-k then filter) returns too few results when the filter is selective.
Pre-filtering (restrict candidates first) returns k results. Real DBs do this inside the index."""
import numpy as np

rng = np.random.default_rng(0)
N, d, k = 50_000, 64, 10
X = rng.standard_normal((N, d)).astype("float32"); X /= np.linalg.norm(X, axis=1, keepdims=True)
tenant = rng.choice(["a", "b", "c", "d"], N, p=[0.01, 0.29, 0.35, 0.35])  # tenant "a" = 1% of data
q = X[123] + 0.1 * rng.standard_normal(d).astype("float32"); q /= np.linalg.norm(q)

sims = X @ q
post = [i for i in np.argsort(-sims)[:k] if tenant[i] == "a"]               # search -> then filter
mask = np.where(tenant == "a")[0]
pre = mask[np.argsort(-sims[mask])[:k]]                                      # filter -> then search

print(f"post-filter returned {len(post)} of {k} results   <- the classic bug")
print(f"pre-filter  returned {len(pre)} of {k} results")
print("Fix in practice: filterable HNSW (Qdrant), partitions per tenant, or over-fetch k*N then filter.")
