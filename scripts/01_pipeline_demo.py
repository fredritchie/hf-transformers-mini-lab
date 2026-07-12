"""Basic Hugging Face pipeline examples.

Run:
    python scripts/01_pipeline_demo.py

This script demonstrates quick pretrained model usage. The first run may download
models from Hugging Face.
"""

from transformers import pipeline


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    print_section("1. Sentiment analysis pipeline")
    sentiment = pipeline("sentiment-analysis")
    print(sentiment("AI infrastructure is an exciting path for DevOps engineers."))

    print_section("2. Text generation pipeline")
    generator = pipeline("text-generation", model="distilgpt2")
    prompt = "AI infrastructure engineers are responsible for"
    output = generator(prompt, max_new_tokens=30, do_sample=True, temperature=0.7)
    print(output[0]["generated_text"])

    print_section("3. Question answering pipeline")
    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    context = (
        "AI infrastructure focuses on deploying, scaling, monitoring, and operating "
        "machine learning and large language model workloads in production."
    )
    question = "What does AI infrastructure focus on?"
    print(qa(question=question, context=context))

    print_section("4. Summarization pipeline")
    summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
    text = (
        "AI infrastructure combines cloud infrastructure, Kubernetes, model serving, "
        "observability, automation, and reliability engineering. Engineers in this "
        "area build platforms that allow teams to deploy and operate AI systems in "
        "production with consistent performance and reliability."
    )
    summary = summarizer(text, max_length=45, min_length=10, do_sample=False)
    print(summary[0]["summary_text"])


if __name__ == "__main__":
    main()
