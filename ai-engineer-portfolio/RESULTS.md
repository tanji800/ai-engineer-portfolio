# Results (measured by running this repo)

Environment: GitHub Codespaces, Python 3.14, offline HashingEmbedder (512-d). Timings vary by machine.

## 02 Vector search (20,000 vectors, 64-d, k=10)
| Index | Setting | recall@10 | ms/query |
|---|---|---|---|
| Flat (exact) | - | 1.000 | 0.82 |
| IVF | nprobe=1 | 0.907 | 0.06 |
| IVF | nprobe=2 | 1.000 | 0.09 |
| IVF | nprobe=4 | 1.000 | 0.13 |
| IVF | nprobe=16 | 1.000 | 0.30 |

Takeaway: nprobe=2 reached full recall at about 9x lower latency than exact search on this clustered data.

## 02 Graph index from scratch (3,000 vectors, 32-d, k=10)
| efSearch | recall@10 | ms/query |
|---|---|---|
| 10 | 0.651 | 0.34 |
| 20 | 0.856 | 0.62 |
| 50 | 0.958 | 0.81 |
| 100 | 0.989 | 0.76 |

Takeaway: higher efSearch buys recall at the cost of latency; gains flatten near the top.

## 02 Metadata filtering (tenant = 1% of data)
- Post-filter returned 0 of 10 results; pre-filter returned 10 of 10.

## 04 Retrieval evaluation (8 labelled queries, offline embedder)
| Method | recall@3 | MRR |
|---|---|---|
| dense | 0.562 | 0.461 |
| BM25 | 0.875 | 0.821 |
| hybrid (RRF) | 0.875 | 0.696 |

Caveat: the offline embedder is lexical, not semantic, so this does NOT show BM25 beating real dense models.

## TODO: rerun 04 with EMBEDDER=bge (pip install sentence-transformers) and record it here
| Method | recall@3 | MRR |
|---|---|---|
| dense (bge) |  |  |
| BM25 |  |  |
| hybrid (RRF) |  |  |

## TODO: record outputs of steps 05, 06, 07 after the scipy fix
