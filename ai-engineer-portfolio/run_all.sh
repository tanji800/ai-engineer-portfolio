#!/usr/bin/env bash
# Runs every offline demo (no internet, no API keys). Use EMBEDDER=bge for a real model.
set -e
for f in 01_embeddings/01_similarity.py 02_vector_search/01_flat_and_ivf.py 02_vector_search/02_hnsw_from_scratch.py \
         02_vector_search/04_metadata_filtering.py 03_rag/01_rag_from_scratch.py 03_rag/02_chunking_strategies.py \
         04_hybrid_search/01_bm25_dense_rrf_eval.py 05_graph_rag/01_ppr_graph_retrieval.py \
         06_agents/01_adaptive_rag_loop.py 06_agents/02_tool_calling_agent.py 07_data_pipelines/01_incremental_indexer.py; do
  echo; echo "=================== $f"; python "$f"
done
python -m pytest -q tests
