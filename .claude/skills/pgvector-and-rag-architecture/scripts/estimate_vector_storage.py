"""Estimate vector storage requirements based on document count and embedding dimensions.
Usage: python estimate_vector_storage.py <num_documents> <chunk_size_tokens> <embedding_dims>
"""
import sys


def estimate(num_docs: int, chunk_size: int, embedding_dims: int, avg_tokens_per_doc: int = 5000):
    chunks_per_doc = max(1, avg_tokens_per_doc // chunk_size)
    total_chunks = num_docs * chunks_per_doc

    # Each float32 = 4 bytes
    embedding_bytes = embedding_dims * 4
    vector_storage_gb = (total_chunks * embedding_bytes) / (1024**3)

    # Rough text storage (4 bytes per token avg)
    text_storage_gb = (total_chunks * chunk_size * 4) / (1024**3)

    # HNSW index overhead ~1.5x vector size
    index_overhead_gb = vector_storage_gb * 1.5

    total_gb = vector_storage_gb + text_storage_gb + index_overhead_gb

    print("--- Vector Storage Estimate ---")
    print(f"Documents:          {num_docs:,}")
    print(f"Chunks per doc:     {chunks_per_doc}")
    print(f"Total chunks:       {total_chunks:,}")
    print(f"Embedding dims:     {embedding_dims}")
    print(f"Vector storage:     {vector_storage_gb:.2f} GB")
    print(f"Text storage:       {text_storage_gb:.2f} GB")
    print(f"HNSW index (~1.5x): {index_overhead_gb:.2f} GB")
    print(f"Total estimated:    {total_gb:.2f} GB")


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        estimate(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    else:
        print("Usage: python estimate_vector_storage.py <num_docs> <chunk_size_tokens> <embedding_dims>")
        print("Example: python estimate_vector_storage.py 100000 500 1536")
