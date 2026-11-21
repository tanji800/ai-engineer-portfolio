# System Design Prompts (practice 30 min each, draw on a whiteboard)

Use this skeleton every time: **Requirements → Scale math → Architecture → Deep dives → Failure modes → Evaluation → Cost.**

## A. Enterprise RAG over 10M documents with per-user permissions
- Clarify: freshness SLO, QPS, latency target (e.g. p95 < 2 s), languages, document types, compliance.
- Math: 10M docs × ~20 chunks = 200M chunks × 768 d × 4 B ≈ 614 GB raw → quantization / sharding needed. Do the arithmetic out loud.
- Ingestion: connectors + CDC → parse → chunk → embed (batch) → upsert (stable IDs, hash, ACL metadata).
- Retrieval: hybrid (BM25 + dense) → ACL filter inside the index → rerank top 50 → generate with citations.
- Deep dives: filtered ANN, sharding/replication, re-index with alias swap, cache, access control leakage.
- Eval: golden set, recall@k/nDCG, faithfulness, online feedback, canary queries in CI.

## B. Semantic search for an e-commerce catalog (500M items, real-time updates)
- Two-stage: ANN candidate generation (IVF-PQ or HNSW sharded) → reranker with business signals.
- Real-time inventory/price: keep in a separate store, join at rerank (don't re-embed on price change).
- Filters (category, brand, in-stock) → filterable index / partitions.
- Cold start, multilingual, query understanding (typos, synonyms), A/B testing, click feedback loops.

## C. Customer-support agent with tools (refund, order lookup, KB search)
- Router → RAG for policy questions, tools for actions, human handoff for risky ones.
- Guardrails: tool schemas, authorization per action, step limit, PII redaction, prompt-injection defence.
- Memory: conversation + customer profile. Evaluation: task success rate on simulated tickets, escalation rate, CSAT.

## D. Knowledge graph + RAG for multi-hop analytics ("which suppliers of recalled parts are in Ohio?")
- Extraction (LLM/NER) → entity resolution → graph store + vector index on chunks/entities.
- Query: entity linking → PPR / graph traversal → fetch passages → answer with path explanation.
- Update path: incremental extraction, deletions, versioned graph. Compare against a strong hybrid baseline before committing.

## Cheat numbers to memorize
- float32 vector: `4 × dim` bytes. 1M × 768 d ≈ 3 GB. 
- PQ with m=16, 8-bit codes: 16 B per vector.
- RRF constant k = 60. Typical rerank candidates: 20–100. Typical chunk: 300–500 tokens, 10–15% overlap.
