import os
import argparse
from datasets import load_dataset, DatasetDict


def convert_jsonl_to_arrow(input_dir, output_dir):
    data_files = {
        "train": os.path.join(input_dir, "train.jsonl"),
        "validation": os.path.join(input_dir, "validation.jsonl"),
        "test": os.path.join(input_dir, "test.jsonl")
    }

    for split, file_path in data_files.items():
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

    dataset = load_dataset("json", data_files=data_files)

    if isinstance(dataset, DatasetDict):
        dataset.save_to_disk(output_dir)
        print(f"Saved dataset to: {output_dir}")
    else:
        raise ValueError("Dataset cannot convert to DatasetDict.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse Jsonl to Arrow")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="folder including file train.jsonl, validation.jsonl, test.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory")
    args = parser.parse_args()

    convert_jsonl_to_arrow(args.input_dir, args.output_dir)
