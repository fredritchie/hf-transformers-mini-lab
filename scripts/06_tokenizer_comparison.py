"""Compare how different tokenizers split the same text.

This helps explain why tokenization affects model behavior, context length,
and inference cost.
"""

from transformers import AutoTokenizer


def show_tokens(model_name: str, text: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)
    print("Tokens:", tokens)
    print("Token count:", len(tokens))
    print("Token IDs:", token_ids[:20], "..." if len(token_ids) > 20 else "")


def main() -> None:
    text = "AI infrastructure engineers deploy Kubernetes-based inference platforms."

    models = [
        "bert-base-uncased",
        "gpt2",
        "roberta-base",
    ]

    for model_name in models:
        show_tokens(model_name, text)


if __name__ == "__main__":
    main()
