import json
import random

input_path = "data/fine_tuning/combined_prompt.jsonl"
train_path = "data/fine_tuning/train.jsonl"
test_path = "data/fine_tuning/test.jsonl"

# Use encoding="utf-8" to avoid UnicodeDecodeError
with open(input_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

random.shuffle(data)
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]

with open(train_path, "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(test_path, "w", encoding="utf-8") as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"✅ Split complete. Train: {len(train_data)} | Test: {len(test_data)}")
