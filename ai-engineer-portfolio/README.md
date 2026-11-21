# AI Engineer Portfolio: Vector Search, RAG, Graph RAG, Agents, Data Pipelines

Hands-on notes + runnable code + interview prep for **Data Engineer / AI Engineer** roles.
Every module runs **offline** (NumPy only) so you can learn the mechanics first, then switch to real models
(`EMBEDDER=bge`, `ANTHROPIC_API_KEY=...`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy networkx pytest            # core
./run_all.sh                                 # runs everything + tests
```

## Where to start (in this order, each step builds on the last)
| Step | Folder | Learn | Output you should write in `NOTES.md` |
|---|---|---|---|
| 1 | `01_embeddings` | embeddings, cosine/dot/L2, memory math | why normalized cosine = dot; GB for 10M vectors |
| 2 | `02_vector_search` | Flat, IVF, HNSW, PQ, filtering | recall-vs-latency table you generated |
| 3 | `03_rag` | chunking, retrieval, prompting, citations | your chunk-size experiment |
| 4 | `04_hybrid_search` | BM25, RRF, reranking, **evaluation** | recall@k/MRR for dense vs BM25 vs hybrid (try `EMBEDDER=bge`) |
| 5 | `05_graph_rag` | knowledge graphs, PPR, LightRAG / GraphRAG | 5-mode comparison on one corpus |
| 6 | `06_agents` | adaptive RAG, tool-calling loop, guardrails | trace of a failed-then-corrected query |
| 7 | `07_data_pipelines` | incremental indexing, deletes, re-index | cost saved by hashing (embed calls per run) |
| 8 | `interview/` | question bank + system design | answer each question aloud, then check |

**Fast schedule (4 weeks, ~2-3 h/day):** W1 steps 1-2 · W2 steps 3-4 · W3 steps 5-6 · W4 step 7 + a capstone + daily mock questions.
Rule for every module: **read → run → change a parameter → record numbers → explain it out loud in 2 minutes.**
If you can't explain it without notes, it's not interview-ready.

## Turn this into a real portfolio (what hiring managers actually look at)
The scaffold teaches mechanics. Add **3 capstones** with real data, each in its own folder with a README that has
architecture diagram, **measured results table**, trade-offs, and "what I'd do next":
1. **Hybrid search + reranker service** over a real corpus (docs/wiki/papers). FastAPI + Docker, eval harness, before/after metrics.
2. **Incremental ingestion pipeline** into pgvector/Qdrant (Airflow/Dagster or CocoIndex), with deletes, retries, and freshness monitoring.
3. **Adaptive/graph RAG assistant** with citations, an LLM grader, a golden-set eval in CI (GitHub Actions), compared against a plain-RAG baseline.

Quality bar: tests, pinned `requirements.txt`, clear README, a results table, no secrets committed.

## Push to GitHub (github.com/tanji800)
```bash
cd ai-engineer-portfolio
git init && git add . && git commit -m "Initial commit: AI engineer learning portfolio"
git branch -M main
# create an empty repo named ai-engineer-portfolio on github.com first, then:
git remote add origin https://github.com/tanji800/ai-engineer-portfolio.git
git push -u origin main
```

## Honest notes
- The offline `HashingEmbedder` is lexical, not semantic. That's why BM25 beats "dense" in step 4 offline. Rerun with `EMBEDDER=bge` for a realistic result and record both.
- The adaptive-RAG grader is a similarity threshold: crude on purpose. In production use an LLM/trained grader and evaluate it.
- `02_vector_search/03_faiss_compare.py` and `05_graph_rag/02_lightrag_quickstart.py` need extra installs/keys and were not run here; LightRAG's API changes often, so follow its current README.

## External repos to study next (verify each repo's recent activity before investing time)
`NirDiamant/RAG_Techniques` · `HKUDS/LightRAG` · `microsoft/graphrag` · `cocoindex-io/cocoindex` ·
`erikbern/ann-benchmarks` · `HandsOnLLM/Hands-On-Large-Language-Models` · `facebookresearch/faiss`
