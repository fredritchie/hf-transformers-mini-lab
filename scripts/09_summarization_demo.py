"""Summarization demo using a pretrained model.

Summarization is useful for document processing, report generation, and AI
assistant workflows.
"""

from transformers import pipeline


def main() -> None:
    summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

    text = (
        "AI infrastructure engineers build platforms for deploying and operating "
        "machine learning and large language model workloads. These platforms often "
        "include model serving APIs, Kubernetes deployments, monitoring dashboards, "
        "CI/CD pipelines, vector databases, and operational runbooks. The goal is to "
        "make AI systems reliable, scalable, observable, and easy for product teams to use."
    )

    result = summarizer(text, max_length=55, min_length=15, do_sample=False)
    print("Original text:\n", text)
    print("\nSummary:\n", result[0]["summary_text"])


if __name__ == "__main__":
    main()
