"""Small labelled corpus so every retrieval demo can be EVALUATED, not just eyeballed."""

DOCS = [
    "HNSW is a graph based approximate nearest neighbor index built from layered navigable small world graphs.",
    "IVF partitions vectors into clusters with k-means and searches only the nprobe closest clusters.",
    "Product quantization compresses vectors into short codes to cut memory and speed up distance computation.",
    "BM25 is a lexical ranking function based on term frequency and inverse document frequency.",
    "Reciprocal rank fusion merges rankings from dense and sparse retrievers without score normalization.",
    "Retrieval augmented generation grounds an LLM answer in retrieved passages to reduce hallucination.",
    "Chunking splits documents into passages; chunk size and overlap strongly affect retrieval quality.",
    "A cross encoder reranker scores query and passage together and is more accurate but slower than bi-encoders.",
    "GraphRAG extracts entities and relationships from text and builds a knowledge graph for multi-hop questions.",
    "Personalized PageRank spreads activation from query entities across a knowledge graph to find related nodes.",
    "Agents use an LLM loop with tools, memory and planning to decide which action to take next.",
    "Incremental indexing re-embeds only changed documents using content hashes instead of rebuilding everything.",
    "pgvector adds vector similarity search to PostgreSQL with HNSW and IVFFlat indexes.",
    "Embedding models map text to dense vectors so that semantically similar texts are close in cosine similarity.",
    "Recall at k and MRR measure whether retrieval returns the relevant documents near the top.",
    "Metadata filtering restricts vector search to documents matching attributes such as tenant or date.",
]

# (query, set of relevant doc ids)
QUERIES = [
    ("which index uses a graph for nearest neighbour search", {0}),
    ("how to reduce the memory used by vectors", {2}),
    ("merge keyword and semantic rankings", {4}),
    ("multi hop questions over entities", {8, 9}),
    ("avoid re-embedding unchanged documents", {11}),
    ("how do we measure retrieval quality", {14}),
    ("store vectors inside postgres", {12}),
    ("rerank passages more accurately", {7}),
]
