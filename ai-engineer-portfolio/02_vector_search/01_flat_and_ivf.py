"""STEP 2a - Exact (Flat) vs IVF index, from scratch with NumPy.

Learn: why ANN exists, what nlist/nprobe do, how to MEASURE recall vs speed.
Run:  python 02_vector_search/01_flat_and_ivf.py
"""
import time
import numpy as np


def make_data(n=20000, d=64, clusters=50, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((clusters, d))
    X = centers[rng.integers(0, clusters, n)] + 0.35 * rng.standard_normal((n, d))
    X = X.astype("float32")
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def flat_search(X, q, k):                      # exact, O(N*d) per query
    return np.argsort(-(X @ q))[:k]


def kmeans(X, k, iters=10, seed=0):             # spherical k-means (cosine)
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmax(X @ C.T, axis=1)
        for j in range(k):
            m = X[assign == j]
            if len(m):
                C[j] = m.mean(0)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    return C


class IVF:
    def __init__(self, nlist=64):
        self.nlist = nlist

    def train_add(self, X):
        self.X = X
        self.C = kmeans(X, self.nlist)
        assign = np.argmax(X @ self.C.T, axis=1)
        self.lists = [np.where(assign == j)[0] for j in range(self.nlist)]  # inverted lists

    def search(self, q, k, nprobe=4):
        cells = np.argsort(-(self.C @ q))[:nprobe]          # 1) pick closest clusters
        cand = np.concatenate([self.lists[c] for c in cells])  # 2) scan only those
        return cand[np.argsort(-(self.X[cand] @ q))[:k]]


if __name__ == "__main__":
    X = make_data()
    Q = X[np.random.default_rng(1).choice(len(X), 200, replace=False)] + 0.05
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    k = 10

    t = time.perf_counter()
    gt = [flat_search(X, q, k) for q in Q]
    t_flat = time.perf_counter() - t
    print(f"FLAT  exact: {t_flat*1000/len(Q):.2f} ms/query")

    ivf = IVF(nlist=64)
    ivf.train_add(X)
    for nprobe in (1, 2, 4, 8, 16):
        t = time.perf_counter()
        res = [ivf.search(q, k, nprobe) for q in Q]
        dt = time.perf_counter() - t
        rec = np.mean([len(set(r) & set(g)) / k for r, g in zip(res, gt)])
        print(f"IVF   nprobe={nprobe:<2} recall@{k}={rec:.3f}  {dt*1000/len(Q):.2f} ms/query")
    print("\nTakeaway: nprobe is the recall/latency knob. Always report recall next to latency.")
