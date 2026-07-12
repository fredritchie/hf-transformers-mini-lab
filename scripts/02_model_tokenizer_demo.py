"""Understand AutoTokenizer, AutoModel, input_ids, attention_mask, and hidden states."""

import torch
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    model_name = "distilbert-base-uncased"
    text = "Kubernetes can run scalable AI inference services."

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    encoded = tokenizer(text, return_tensors="pt")

    print("Input text:", text)
    print("Input IDs:", encoded["input_ids"])
    print("Attention mask:", encoded["attention_mask"])
    print("Tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))

    with torch.no_grad():
        output = model(**encoded)

    print("Last hidden state shape:", output.last_hidden_state.shape)
    print("Meaning: [batch_size, sequence_length, hidden_size]")


if __name__ == "__main__":
    main()
