---
description: pgvector and RAG Architecture Examples
---

**Example 1: pgvector Table Schema with Metadata and HNSW Index**

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    metadata JSONB DEFAULT '{}',
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_documents_embedding
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (ef_construction = 128, m = 16);

CREATE INDEX idx_documents_source ON documents(source);
CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata);
```

---

**Example 2: Hybrid Search Query (Vector + Metadata Filter)**

```sql
SELECT id, content, source, metadata,
       1 - (embedding <=> $1::vector) AS similarity
FROM documents
WHERE metadata->>'category' = $2
  AND metadata->>'access_level' <= $3
  AND created_at > NOW() - INTERVAL '90 days'
ORDER BY embedding <=> $1::vector
LIMIT 50;
```

---

**Example 3: Python Embedding Generation with Batching**

```python
import openai
from tqdm import tqdm

BATCH_SIZE = 100

def generate_embeddings(texts: list[str], model="text-embedding-3-small") -> list[list[float]]:
    """Generate embeddings in batches to respect rate limits."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i : i + BATCH_SIZE]
        response = openai.embeddings.create(input=batch, model=model)
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings
```

---

**Example 4: LangChain RAG Pipeline with pgvector**

```python
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

vectorstore = PGVector(
    connection_string="postgresql://user:pass@localhost:5432/vectordb",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5, "filter": {"access_level": "public"}}
)

prompt = ChatPromptTemplate.from_template("""
Answer based only on the following context. Cite sources.
<context>
{context}
</context>

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

answer = rag_chain.invoke("What is our refund policy?")
```

---

**Example 5: Re-ranking with Cohere Rerank API**

```python
import cohere

def rerank_results(query: str, documents: list[dict], top_n: int = 5) -> list[dict]:
    co = cohere.Client("COHERE_API_KEY")
    results = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=[doc["content"] for doc in documents],
        top_n=top_n,
    )
    return [
        {**documents[r.index], "relevance_score": r.relevance_score}
        for r in results.results
    ]
```

---

**Example 6: PII Redaction Before Storage**

```python
import re

PII_PATTERNS = {
    "email": (r"\b[\w.-]+@[\w.-]+\.\w+\b", "[EMAIL]"),
    "phone": (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]"),
    "ssn": (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    "credit_card": (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CC]"),
}

def redact_pii(text: str) -> tuple[str, dict[str, str]]:
    """Redact PII and return mapping for potential un-redaction."""
    mapping = {}
    for pii_type, (pattern, placeholder) in PII_PATTERNS.items():
        for match in re.finditer(pattern, text):
            token = f"{placeholder}_{len(mapping)}"
            mapping[token] = match.group()
            text = text.replace(match.group(), token, 1)
    return text, mapping
```

---

**Example 7: Retrieval Evaluation (precision@k, recall@k, MRR)**

```python
def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / k

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0

def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0

# Usage
retrieved_ids = ["doc1", "doc3", "doc5", "doc7", "doc9"]
relevant_ids = {"doc1", "doc2", "doc5"}

print(f"P@5: {precision_at_k(retrieved_ids, relevant_ids, 5):.2f}")  # 0.40
print(f"R@5: {recall_at_k(retrieved_ids, relevant_ids, 5):.2f}")     # 0.67
print(f"MRR: {mrr(retrieved_ids, relevant_ids):.2f}")                 # 1.00
```

---

**Example 8: Streaming RAG Response with Source Citations**

```python
from anthropic import Anthropic

def stream_rag_answer(query: str, context_docs: list[dict]):
    client = Anthropic()
    context = "\n\n".join(
        f"[Source {i+1}: {d['source']}] {d['content']}"
        for i, d in enumerate(context_docs)
    )

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer with citations [Source N]."}],
    ) as stream:
        for text in stream.text_stream:
            yield text
```
