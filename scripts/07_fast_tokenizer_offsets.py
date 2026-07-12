"""Demonstrate fast tokenizer offset mapping.

Offset mapping connects tokens back to character positions in the original text.
This is useful for question answering, document extraction, and RAG debugging.
"""

from transformers import AutoTokenizer


def main() -> None:
    model_name = "distilbert-base-uncased"
    text = "Inference platform engineers monitor latency, throughput, and errors."

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Is fast tokenizer:", tokenizer.is_fast)

    encoded = tokenizer(text, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    offsets = encoded["offset_mapping"]

    print("\nToken to character offsets:\n")
    for token, (start, end) in zip(tokens, offsets):
        original_span = text[start:end]
        print(f"{token:15s} -> ({start:2d}, {end:2d}) -> '{original_span}'")


if __name__ == "__main__":
    main()
