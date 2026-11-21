# Interview Question Bank (answer out loud, then check)

Format: **Q** then a short model answer. Each section links to the code that lets you *demonstrate* the answer.

## 1. Embeddings  (`01_embeddings/`)
**Q: What is an embedding and why cosine similarity?**
A dense vector from a model trained so similar items are close. Cosine compares direction, ignoring length. For L2-normalized vectors cosine = dot product and L2 ranking = cosine ranking (`||a-b||² = 2-2cos`), so we normalize once and use fast dot products.

**Q: How do you choose an embedding model?**
Test candidates on *your* labelled queries (recall@k, MRR), not just the MTEB leaderboard. Weigh dimension (memory, latency), max tokens, language/domain fit, license, cost, and whether query/passage instructions are needed.

**Q: Cost of 10M vectors, 768-d, float32?**
10M × 768 × 4 B ≈ 30.7 GB raw, before index overhead (HNSW links add more). Mitigations: lower dims (Matryoshka truncation), float16/int8 scalar quantization, PQ.

**Q: What happens if you change the embedding model?**
Old and new vectors live in different spaces. You must re-embed everything and re-index (blue/green index swap). Store the model version with each vector (`07_data_pipelines`).

**Q: Dense vs sparse vectors?**
Dense = semantic, learned, low-dim. Sparse (BM25, SPLADE) = lexical, high-dim, exact-term matching (IDs, error codes, rare words). They fail differently, hence hybrid search.

**Q: Bi-encoder vs cross-encoder vs late interaction (ColBERT)?**
Bi-encoder embeds query and doc separately (fast, indexable). Cross-encoder reads both together (accurate, too slow to scan the corpus, used to rerank top-k). ColBERT keeps per-token vectors and does MaxSim: between the two in cost and accuracy.

## 2. Vector search & indexing  (`02_vector_search/`)
**Q: Why not brute force?**
O(N·d) per query. Fine up to ~100k vectors; at millions/billions latency and cost explode, so we use ANN (trade a little recall for big speedups).

**Q: Explain HNSW.**
A multi-layer proximity graph. Upper layers are sparse "express lanes"; search greedily descends layer by layer, then does a beam search (`efSearch`) at layer 0. Knobs: `M` (links/node → recall + memory), `efConstruction` (build quality/time), `efSearch` (query recall/latency). Weaknesses: RAM heavy, slow/expensive deletes, build time.

**Q: Explain IVF and the nprobe trade-off.**
k-means partitions vectors into `nlist` cells; query scans only the `nprobe` closest cells. More `nprobe` → higher recall, higher latency. Needs a training step; recall drops if data drifts from the clusters.

**Q: What is product quantization?**
Split each vector into m sub-vectors, quantize each to one of 256 centroids → m bytes per vector (e.g. 512 B → 16 B). Distances computed via lookup tables (ADC). Lossy; often followed by re-scoring top candidates with full vectors.

**Q: HNSW vs IVF vs DiskANN: when?**
HNSW: best latency/recall when it fits RAM. IVF-PQ: huge datasets, memory-constrained. DiskANN/Vamana: billion-scale on SSD with small RAM. Flat: <100k or when you need exact results.

**Q: How do you measure ANN quality?**
Recall@k vs exact search on held-out queries, plotted against QPS/latency, plus build time and memory (`ann-benchmarks` style). Never quote latency without recall.

**Q: Metadata-filtered search: what goes wrong?**
Post-filtering a top-k can return 0 results for selective filters (`04_metadata_filtering.py`). Fixes: in-index filtering (filterable HNSW), per-tenant partitions/collections, over-fetch then filter, or pre-filter + exact scan when the filtered set is small.

**Q: How do updates/deletes work?**
HNSW: soft-delete (tombstones) + periodic rebuild/compaction. IVF: append to lists, retrain occasionally. LSM-style engines (e.g., segment merges) batch writes. Know your DB's story, deletes are a real production issue.

**Q: pgvector vs a dedicated vector DB?**
pgvector: one system, transactions, joins, SQL filters, great to ~millions of vectors. Dedicated (Qdrant/Milvus/etc.): better scale-out, filtering performance, hybrid features, ops tooling. Choose by scale, latency SLO, team ops capacity.

**Q: Cosine vs dot vs L2?**
Same ranking for normalized vectors. Use what the model was trained with. Dot product on non-normalized vectors is also affected by magnitude.

## 3. RAG  (`03_rag/`)
**Q: Walk me through a RAG pipeline.**
Ingest (parse, clean, chunk, embed, index + metadata) → query (rewrite, retrieve, rerank) → generate (prompt with context + citations) → evaluate/monitor. Most failures are retrieval failures, not generation.

**Q: How do you pick chunk size?**
Test it. Start ~300–500 tokens, 10–15% overlap, structure-aware (headings/paragraphs). Small chunks → precise but lose context; large → noisy and costly. Consider parent-document retrieval (retrieve small, return larger context) or sentence-window.

**Q: RAG vs fine-tuning?**
RAG for fresh/private/changing knowledge, citations, access control. Fine-tuning for style, format, behavior, latency. They combine well. RAG is cheaper to update and auditable.

**Q: How do you reduce hallucinations?**
Better retrieval + reranking, "answer only from context / say not found", citations, groundedness check (second LLM pass or NLI), low temperature, refusing below a relevance threshold.

**Q: How do you evaluate a RAG system?**
Separate retrieval and generation. Retrieval: recall@k, MRR, nDCG on a labelled set. Generation: faithfulness/groundedness, answer relevance, context precision/recall (RAGAS/DeepEval-style), plus human review on a golden set. Run in CI to catch regressions.

**Q: Query transformations?**
Rewrite/clarify, multi-query expansion, HyDE (embed a hypothetical answer), decomposition into sub-questions, step-back prompts. Each adds latency/cost, so measure the gain.

**Q: What is "lost in the middle"?**
LLMs use information at the start/end of long contexts better than the middle. Order by relevance, limit context, rerank, and don't stuff 50 chunks.

**Q: Handle tables, PDFs, images?**
Layout-aware parsing (keep tables as units/markdown), OCR for scans, captioning or multimodal embeddings for images, store source/page metadata for citations.

**Q: Security concerns?**
Prompt injection via retrieved docs (treat retrieved text as untrusted data), document-level ACLs enforced at retrieval time (filter by user permissions, not after generation), PII handling, tenant isolation.

**Q: How do you cut latency and cost?**
Cache embeddings and frequent queries, smaller/faster models for routing and grading, ANN tuning, rerank only top 20–50, streaming responses, batch embedding at ingest.

## 4. Hybrid search & reranking  (`04_hybrid_search/`)
**Q: Why hybrid search?**
Dense misses exact tokens (SKUs, error codes, names); BM25 misses paraphrases. Combining raises recall across query types.

**Q: Explain RRF. Why not just add scores?**
`score(d)=Σ 1/(k+rank_i(d))`, k≈60. BM25 scores and cosine scores are on different scales, so adding is meaningless without normalization. RRF uses ranks only: robust, parameter-light.

**Q: Where does reranking fit?**
Retrieve broad (top 50–200, cheap, high recall) → rerank (cross-encoder / LLM, top 5–10, high precision). Watch added latency.

**Q: How do you know hybrid helped?**
Run `01_bm25_dense_rrf_eval.py` style evals on your own query set. Note: with the offline hashing embedder BM25 wins because that embedder is lexical, rerun with `EMBEDDER=bge` for a realistic comparison.

## 5. Graph RAG  (`05_graph_rag/`)
**Q: When does graph RAG beat vector RAG?**
Multi-hop questions (A→B→C), global/"what are the themes" questions over a whole corpus, entity-centric queries, explainability needs. Not always better: higher indexing cost, more complexity.

**Q: Microsoft GraphRAG vs LightRAG vs HippoRAG?**
GraphRAG: entity graph + hierarchical community summaries (Leiden) → strong global answers, expensive indexing. LightRAG: entities+relations with dual-level (local/global) retrieval and cheaper incremental updates. HippoRAG: entity graph + Personalized PageRank for multi-hop with few LLM calls at query time.

**Q: What is Personalized PageRank doing?**
Random walk with restart from the query's entities: nodes close in the graph get high scores, surfacing passages 2–3 hops away that share no words with the query (`01_ppr_graph_retrieval.py`).

**Q: Biggest practical issues of graph RAG?**
Extraction quality (entity resolution/dedup), indexing cost (LLM calls per chunk), keeping the graph in sync on updates and deletes, and evaluating gains honestly vs a strong hybrid baseline.

**Q: Knowledge graph store choices?**
Neo4j/Memgraph/Kuzu for native graph traversal; Postgres + pgvector for simple setups; or in-memory NetworkX for prototypes.

## 6. Agents & dynamic search  (`06_agents/`)
**Q: What is an agent vs a workflow?**
Workflow: predefined code path with LLM steps. Agent: the LLM decides the next action in a loop with tools. Prefer the simplest thing that works; use agents where the path can't be predefined.

**Q: What is adaptive / corrective RAG?**
Route each query (vector, web, direct), grade retrieved context, rewrite and retry on failure, fall back to another source, and check the answer is grounded (`01_adaptive_rag_loop.py`). Note its toy grader is a similarity threshold, which is crude; production uses an LLM or trained grader and still needs evaluation.

**Q: How do you make agents reliable?**
Step/budget limits, typed tool schemas with validation, error feedback to the model, idempotent tools, human approval for risky actions, tracing/observability, evals on task suites, guardrails against prompt injection.

**Q: Agent memory types?**
Short-term (conversation window), long-term (vector store/KG of facts), episodic (past runs), procedural (learned instructions). Decide what to write, when to forget, and how to avoid stale/poisoned memories.

**Q: MCP?**
Model Context Protocol: an open standard for exposing tools/data to LLM apps through servers, so integrations are written once and reused across clients.

**Q: Multi-agent: worth it?**
Helps with parallelizable or role-separated tasks, but adds cost, latency, and failure modes. Start single-agent with good tools.

## 7. Data engineering for AI  (`07_data_pipelines/`)
**Q: Design ingestion for a RAG system.**
Connectors/CDC → parse → clean/dedupe → chunk → embed (batched, rate-limited, retried) → upsert with stable IDs + metadata + content hash → index. Idempotent, incremental, observable.

**Q: How do you keep the index fresh cheaply?**
Content hashes (+ model version) so you embed only changes; tombstone deletes; CDC or event-driven triggers; periodic full reconciliation to fix drift (`01_incremental_indexer.py`).

**Q: Batch vs streaming indexing?**
Batch (Airflow/Dagster/Spark): simple, cheap, minutes-hours freshness. Streaming (Kafka/Flink): seconds freshness, more operational complexity. Match to the freshness SLO.

**Q: Zero-downtime re-index?**
Build a new index/collection in the background, validate with eval set, switch an alias atomically, keep the old one for rollback.

**Q: What do you monitor?**
Ingest lag, failed docs, embedding cost, index size, query latency p50/p95/p99, retrieval metrics on a canary set, "no result"/low-score rate, user feedback, drift in query distribution.

**Q: Handling PII/compliance?**
Classify/redact before embedding, encrypt at rest, ACL metadata filtered at query time, retention/deletion workflows (right to be forgotten must remove vectors too).

## 8. Behavioral / project questions (use your repo!)
- "Tell me about a retrieval bug you debugged." → the post-filter 0-results case.
- "How did you decide on an index?" → your recall/latency table from `02_vector_search`.
- "How do you know your RAG is good?" → your eval harness + numbers.
- Always structure: context → what you measured → decision → result → what you'd change.
