"""Load custom local datasets from CSV, JSON, and TXT files."""

from pathlib import Path
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def main() -> None:
    csv_path = DATA_DIR / "sample.csv"
    json_path = DATA_DIR / "sample.json"
    txt_path = DATA_DIR / "sample.txt"

    print("Loading CSV dataset")
    csv_dataset = load_dataset("csv", data_files=str(csv_path))
    print(csv_dataset)
    print(csv_dataset["train"][0])

    print("\nLoading JSON dataset")
    json_dataset = load_dataset("json", data_files=str(json_path))
    print(json_dataset)
    print(json_dataset["train"][0])

    print("\nLoading text dataset")
    text_dataset = load_dataset("text", data_files=str(txt_path))
    print(text_dataset)
    print(text_dataset["train"][0])


if __name__ == "__main__":
    main()
