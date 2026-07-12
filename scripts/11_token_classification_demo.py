"""Optional token classification demo.

Token classification is useful for named entity recognition and structured
extraction from documents.
"""

from transformers import pipeline


def main() -> None:
    ner = pipeline("token-classification", model="dslim/bert-base-NER", grouped_entities=True)
    text = "Fredrick works on Kubernetes and AWS in Chennai for AI infrastructure projects."

    print("Text:", text)
    print("\nEntities:")
    for entity in ner(text):
        print(entity)


if __name__ == "__main__":
    main()
