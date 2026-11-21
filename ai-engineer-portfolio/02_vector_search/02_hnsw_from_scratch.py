"""STEP 2b - Graph index (the idea behind HNSW) from scratch.

This is a single-layer Navigable Small World graph. Real HNSW adds hierarchical layers on top
(long-range "express lanes") so the greedy search starts near the target. The knobs are the same:
  M (links per node), efConstruction (build beam), efSearch (query beam).
Run:  python 02_vector_search/02_hnsw_from_scratch.py
"""
import heapq
import time
import numpy as np


class NSW:
    def __init__(self, X, M=12, ef_construction=48):
        self.X, self.M, self.efc = X, M, ef_construction
        self.nbrs = [[] for _ in range(len(X))]
        for i in range(1, len(X)):
            self._insert(i)

    def _beam(self, q, entry, ef):
        d0 = -float(self.X[entry] @ q)             # distance = -similarity (smaller is better)
        visited = {entry}
        cand = [(d0, entry)]                        # min-heap of candidates to expand
        res = [(-d0, entry)]                        # max-heap (by distance) of best ef results
        while cand:
            d, c = heapq.heappop(cand)
            if len(res) >= ef and d > -res[0][0]:
                break                               # closest candidate is worse than worst result
            for n in self.nbrs[c]:
                if n in visited:
                    continue
                visited.add(n)
                dn = -float(self.X[n] @ q)
                if len(res) < ef or dn < -res[0][0]:
                    heapq.heappush(cand, (dn, n))
                    heapq.heappush(res, (-dn, n))
                    if len(res) > ef:
                        heapq.heappop(res)
        return sorted((-nd, i) for nd, i in res)    # ascending distance

    def _insert(self, i):
        found = self._beam(self.X[i], 0, self.efc)
        for _, j in found[: self.M]:
            self.nbrs[i].append(j)
            self.nbrs[j].append(i)
            if len(self.nbrs[j]) > 2 * self.M:      # prune: keep the closest 2M links
                sims = self.X[self.nbrs[j]] @ self.X[j]
                keep = np.argsort(-sims)[: 2 * self.M]
                self.nbrs[j] = [self.nbrs[j][t] for t in keep]

    def search(self, q, k, ef=32):
        return [i for _, i in self._beam(q, 0, max(ef, k))[:k]]


if __name__ == "__main__":
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "ivf", os.path.join(os.path.dirname(__file__), "01_flat_and_ivf.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

    X = m.make_data(n=3000, d=32, clusters=30)
    Q = X[np.random.default_rng(1).choice(len(X), 100, replace=False)] + 0.05
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    k = 10
    gt = [m.flat_search(X, q, k) for q in Q]

    t = time.perf_counter(); g = NSW(X); print(f"build: {time.perf_counter()-t:.1f}s for {len(X)} vectors")
    for ef in (10, 20, 50, 100):
        t = time.perf_counter()
        res = [g.search(q, k, ef) for q in Q]
        dt = time.perf_counter() - t
        rec = np.mean([len(set(r) & set(x)) / k for r, x in zip(res, gt)])
        print(f"efSearch={ef:<3} recall@{k}={rec:.3f}  {dt*1000/len(Q):.2f} ms/query")
