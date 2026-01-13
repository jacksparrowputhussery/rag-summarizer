from datasets import load_dataset
import json
from tqdm import tqdm

print("Downloading QASPER dataset from Hugging Face...")

dataset = load_dataset("allenai/qasper")



print("\n✓ Dataset loaded!")
print(f"Train: {len(dataset['train'])}")
print(f"Validation: {len(dataset['validation'])}")
print(f"Test: {len(dataset['test'])}")

# Convert validation split
print("\nConverting validation split to JSON format...")

dev_data = {}

for item in tqdm(dataset['validation'], desc="Processing"):
    paper_id = item["id"]

    dev_data[paper_id] = {
        "title": item["title"],
        "abstract": item["abstract"],
        "full_text": item["full_text"],
        "qas": item["qas"]
    }

output_file = "qasper-dev-v0.3.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dev_data, f, indent=2)

print(f"\n✓ Saved: {output_file}")
print(f"Papers: {len(dev_data)}")
print(f"Total questions: {sum(len(p['qas']) for p in dev_data.values())}")

