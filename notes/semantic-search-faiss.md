# Semantic Search with FAISS

## Focus points

- Embeddings convert text into numeric vectors.
- Similar meanings produce vectors close to each other.
- FAISS indexes vectors for fast similarity search.
- This is the foundation of retrieval in RAG systems.

## RAG flow

```text
Document text
-> chunks
-> embeddings
-> vector index
-> query embedding
-> top-k retrieval
-> LLM answer generation
```
