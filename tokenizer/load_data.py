from datasets import load_dataset

from tokenizer.config import DATA_DIR


def export_dataset():
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
    )
    print(dataset)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation", "test"]:
        lines = dataset[split]["text"]
        text = "\n".join(lines)

        output_file = DATA_DIR / f"{split}.txt"
        output_file.write_text(text, encoding="utf-8")

        print(f"{split}: {output_file}")


if __name__ == "__main__":
    export_dataset()
