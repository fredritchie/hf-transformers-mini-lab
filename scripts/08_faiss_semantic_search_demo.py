"""Semantic search using sentence embeddings and FAISS.

This is the foundation for RAG systems: embed documents, index them, and
retrieve the most relevant documents for a query.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError as exc:
    raise SystemExit(
        "faiss-cpu is not installed. Run: pip install faiss-cpu"
    ) from exc


def main() -> None:
    documents = [
        "Kubernetes helps deploy and scale containerized AI services.",
        "MLflow tracks experiments, model parameters, metrics, and artifacts.",
        "Qdrant is a vector database used for semantic search and RAG.",
        "Prometheus and Grafana are useful for monitoring inference latency and errors.",
        "FastAPI can expose a model serving endpoint for prediction requests.",
    ]

    query = "How can I monitor an AI inference service?"

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    doc_embeddings = model.encode(documents, normalize_embeddings=True)
    query_embedding = model.encode([query], normalize_embeddings=True)

    doc_embeddings = np.asarray(doc_embeddings, dtype="float32")
    query_embedding = np.asarray(query_embedding, dtype="float32")

    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(doc_embeddings)

    top_k = 3
    scores, indices = index.search(query_embedding, top_k)

    print("Query:", query)
    print("\nTop results:")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        print(f"{rank}. score={score:.4f} | {documents[idx]}")


if __name__ == "__main__":
    main()
