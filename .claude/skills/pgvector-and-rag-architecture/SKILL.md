---
name: pgvector-and-rag-architecture
description: RAG system design with pgvector — chunking strategies, embedding model selection, vector indexing, hybrid search, re-ranking, prompt injection prevention, and PII handling.
---

**When to Use**
- Building RAG (Retrieval-Augmented Generation) pipelines.
- Implementing vector search or semantic search with pgvector.
- Selecting embedding models or configuring vector indexes.
- Combining SQL metadata filtering with semantic similarity.
- Designing secure retrieval systems with PII redaction.

**Core Principles**
1. **Chunking Strategy Matters**: Choose chunk size by content type (code: 100-300 tokens, prose: 500-1000, legal: 1000-2000). Overlap 10-20%.
2. **Embedding Model Selection**: Balance cost, latency, quality (OpenAI text-embedding-3-small for general, local models for privacy).
3. **Vector Index Selection**: IVFFlat for <1M rows, HNSW for >1M. Tune ef_construction and m parameters.
4. **Hybrid Search**: Always combine vector similarity with metadata filtering (SQL WHERE) for production systems.
5. **Re-ranking**: Apply cross-encoder re-ranking on top-K candidates before LLM injection.
6. **Security**: Sanitize retrieved context. Detect prompt injection. Redact PII before storage.
7. **Evaluate Retrieval**: Measure precision@k, recall@k, MRR alongside generation quality.

**References**
- Load `reference.md` for chunking strategies, embedding models, pgvector config, hybrid search architecture, and security patterns.
- Load `examples.md` for SQL schemas, queries, Python pipelines, and evaluation scripts.

**Scripts**
- `scripts/setup_pgvector.sh`: Initialize pgvector extension on PostgreSQL (env-var configurable, no sudo required).
- `scripts/estimate_vector_storage.py`: Estimate storage from document count and embedding dimensions.
- `scripts/evaluate_retrieval.py`: Evaluate retrieval quality (precision@k, recall@k, MRR).
