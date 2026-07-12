"""Optional translation demo using T5.

Translation is not core for AI infrastructure, but it is useful to understand
how pipelines support different NLP tasks.
"""

from transformers import pipeline


def main() -> None:
    translator = pipeline("translation_en_to_de", model="t5-small", tokenizer="t5-small")
    text = "AI infrastructure engineers deploy and monitor machine learning systems."
    result = translator(text, max_length=60)

    print("English:", text)
    print("German:", result[0]["translation_text"])


if __name__ == "__main__":
    main()
