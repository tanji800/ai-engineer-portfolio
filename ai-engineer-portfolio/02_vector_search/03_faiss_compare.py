"""STEP 2c - The industry tool: FAISS. Needs `pip install faiss-cpu`.
Compare Flat / HNSW / IVF / IVF-PQ on recall, latency and memory."""
import time
import numpy as np
import faiss

d, n, nq, k = 128, 200_000, 500, 10
rng = np.random.default_rng(0)
centers = rng.standard_normal((200, d))
xb = (centers[rng.integers(0, 200, n)] + 0.4 * rng.standard_normal((n, d))).astype("float32")
xq = (xb[rng.choice(n, nq, replace=False)] + 0.05).astype("float32")
faiss.normalize_L2(xb); faiss.normalize_L2(xq)

flat = faiss.IndexFlatIP(d); flat.add(xb)
_, gt = flat.search(xq, k)
recall = lambda I: np.mean([len(set(I[i]) & set(gt[i])) / k for i in range(nq)])

def bench(name, index, setter, values):
    for v in values:
        setter(index, v)
        t = time.perf_counter(); _, I = index.search(xq, k); dt = time.perf_counter() - t
        print(f"{name:<8} param={v:<4} recall={recall(I):.3f}  {dt*1000/nq:.3f} ms/q")

hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efConstruction = 200; hnsw.add(xb)
bench("HNSW", hnsw, lambda ix, v: setattr(ix.hnsw, "efSearch", v), (16, 64, 256))

ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, 1024, faiss.METRIC_INNER_PRODUCT)
ivf.train(xb); ivf.add(xb)
bench("IVF", ivf, lambda ix, v: setattr(ix, "nprobe", v), (1, 8, 32))

# PQ: 128 floats (512 B) -> 16 bytes. ~32x less memory, lossy distances.
ivfpq = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, 1024, 16, 8)
ivfpq.train(xb); ivfpq.add(xb)
bench("IVF-PQ", ivfpq, lambda ix, v: setattr(ix, "nprobe", v), (8, 32))
print("\nMemory/vector: Flat 512 B | HNSW ~512 B + links | IVF-PQ 16 B + overhead")
