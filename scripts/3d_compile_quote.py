import os
import json
from glob import glob

# === CONFIG ===
input_dir = "data/cleaned/knowledge_base"
output_path = "data/processed/quote/quote.json"

# === UTILITY FUNCTIONS ===
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# === EXTRACTION FUNCTION ===
def extract_factual_and_thematic_info(play_json):
    return {
        "title": play_json.get("title", ""),
        "famous_quotes": play_json.get("famous_quotes", ""),
    }

# === MAIN SCRIPT ===
def main():
    play_files = glob(os.path.join(input_dir, "*.json"))
    extracted_data = []

    for play_path in play_files:
        try:
            play_data = load_json(play_path)
            structured_info = extract_factual_and_thematic_info(play_data)
            extracted_data.append(structured_info)
            print(f"✅ Extracted: {structured_info['title']}")
        except Exception as e:
            print(f"❌ Failed to process {play_path}: {e}")

    save_json(extracted_data, output_path)
    print(f"\n🎉 Done! Saved extracted data to: {output_path}")

# === RUN ===
if __name__ == "__main__":
    main()