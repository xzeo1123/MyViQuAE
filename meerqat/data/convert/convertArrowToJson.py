import json
from datasets import load_from_disk

DATASET_PATH = "../../../data/backup2/train/7_meerqat_train_triviaqa_filter"
OUTPUT_FILE = "../../../data/backup2/train/7_meerqat_train_triviaqa_filter/converted_dataset.json"
NUM_SAMPLES = 20


def extract_relevant_info(item):
    return dict(item)


def convert_dataset_to_json(dataset_path, output_file="converted_dataset.json", num_samples=100):
    try:
        dataset = load_from_disk(dataset_path)
        print(f"Loaded dataset with {dataset.num_rows} samples.")

        # If you want to get all samples instead of few samples, change it with max(num_samples, dataset.num_rows)
        num_samples = max(num_samples, dataset.num_rows)
        selected_data = [extract_relevant_info(dataset[i]) for i in range(num_samples)]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(selected_data, f, indent=4, ensure_ascii=False)

        print(f"Successfully saved {num_samples} samples to '{output_file}'")

    except Exception as e:
        print(f"Error while processing dataset: {e}")


if __name__ == "__main__":
    convert_dataset_to_json(DATASET_PATH, OUTPUT_FILE, NUM_SAMPLES)
