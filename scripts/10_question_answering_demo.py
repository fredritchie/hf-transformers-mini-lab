"""Question answering demo.

Question answering is a foundation for document QA and RAG systems.
"""

from transformers import pipeline


def main() -> None:
    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    context = (
        "A production RAG platform usually includes document ingestion, chunking, "
        "embedding generation, vector database storage, retrieval, answer generation, "
        "authentication, monitoring, and deployment automation."
    )

    questions = [
        "What does a production RAG platform include?",
        "Where are embeddings stored?",
        "What comes before embedding generation?",
    ]

    print("Context:\n", context)
    for question in questions:
        answer = qa(question=question, context=context)
        print("\nQuestion:", question)
        print("Answer:", answer["answer"])
        print("Score:", round(answer["score"], 4))


if __name__ == "__main__":
    main()
