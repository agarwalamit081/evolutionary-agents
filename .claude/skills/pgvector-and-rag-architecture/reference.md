---
description: pgvector and RAG Architecture Reference
---

## Chunking Strategies

| Strategy | How It Works | Best For |
|---|---|---|
| Fixed-size | Split at N characters/tokens | Simple, uniform content |
| Recursive character | Split on `\n\n`, `\n`, ` ` with overlap | General purpose (LangChain default) |
| Semantic | Embed, then split on similarity drops | Topically diverse documents |
| Document-aware | Respect markdown headers, code blocks, table rows | Structured content |

**Recommended overlap**: 10-20% of chunk size to preserve context at boundaries.

## Embedding Models

| Model | Dimensions | Cost/1M tokens | Best For |
|---|---|---|---|
| text-embedding-3-small | 1536 | $0.02 | General purpose, cost-effective |
| text-embedding-3-large | 3072 | $0.13 | High accuracy needs |
| Cohere embed-v3 | 1024 | $0.10 | Multilingual, search-optimized |
| all-MiniLM-L6-v2 (local) | 384 | Free | Privacy-first, low latency |

## pgvector Index Configuration

- **HNSW** (recommended): `ef_construction=128, m=16` (good balance). For higher recall: `ef_construction=256`.
- **IVFFlat**: `lists = sqrt(rows)`, `probes = 10` for queries. Only for datasets that rarely change.
- **Distance metrics**: `cosine` (normalized vectors, most common), `L2` (geometric), `inner_product` (pre-normalized).

## Hybrid Search Architecture

1. User query → embed → vector search (top 50-100)
2. Apply SQL metadata filters (date range, category, access level, user permissions)
3. Re-rank remaining with cross-encoder (Cohere rerank, or local cross-encoder)
4. Inject top-K (3-5) chunks into LLM prompt with source metadata
5. Generate answer with citation tracking

## Security Patterns

- **Prompt injection detection**: Classify retrieved chunks before injection (look for instruction-like patterns).
- **PII redaction**: Run NER on chunks before storing; mask sensitive entities at retrieval time.
- **Access control**: Filter by user/team permissions in SQL WHERE clause.
- **Content moderation**: Scan generated output for toxic/harmful content.

## Evaluation Metrics

- **precision@k**: Fraction of retrieved items that are relevant.
- **recall@k**: Fraction of relevant items that are retrieved.
- **MRR** (Mean Reciprocal Rank): Inverse rank of first relevant result.
- **Faithfulness**: Does the answer use only the provided context?
- **Answer Relevancy**: Does the answer address the question?
