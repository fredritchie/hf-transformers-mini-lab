"""Batch inference with padding and truncation.

This is important for production inference APIs because requests are often
processed in batches to improve throughput.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    texts = [
        "The AI inference platform is reliable and fast.",
        "The deployment failed because the service was misconfigured.",
        "Monitoring helped us detect latency issues early.",
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    print("Batch input_ids shape:", encoded["input_ids"].shape)
    print("Batch attention_mask shape:", encoded["attention_mask"].shape)

    with torch.no_grad():
        output = model(**encoded)
        probabilities = torch.softmax(output.logits, dim=-1)

    for text, probs in zip(texts, probabilities):
        predicted_id = int(torch.argmax(probs).item())
        label = model.config.id2label[predicted_id]
        confidence = float(probs[predicted_id])
        print(f"\nText: {text}")
        print(f"Prediction: {label} | Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
