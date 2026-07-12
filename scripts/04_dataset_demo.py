"""Hugging Face Dataset operations without requiring internet.

Demonstrates select, filter, map, shuffle, and train_test_split.
"""

from datasets import Dataset


def main() -> None:
    records = {
        "text": [
            "Kubernetes runs containers at scale.",
            "MLflow tracks experiments and model artifacts.",
            "Qdrant stores vector embeddings for search.",
            "Prometheus collects metrics from services.",
            "FastAPI exposes model inference endpoints.",
        ],
        "category": ["platform", "mlops", "rag", "operations", "serving"],
    }

    dataset = Dataset.from_dict(records)
    print("Original dataset:")
    print(dataset)
    print(dataset[0])

    selected = dataset.select(range(3))
    print("\nSelected first 3 rows:")
    print(selected)

    filtered = dataset.filter(lambda row: row["category"] in ["platform", "serving"])
    print("\nFiltered rows:")
    print(filtered)

    def add_text_length(row):
        row["text_length"] = len(row["text"])
        return row

    mapped = dataset.map(add_text_length)
    print("\nDataset after map():")
    print(mapped[0])

    split = dataset.train_test_split(test_size=0.4, seed=42)
    print("\nTrain/test split:")
    print(split)


if __name__ == "__main__":
    main()
