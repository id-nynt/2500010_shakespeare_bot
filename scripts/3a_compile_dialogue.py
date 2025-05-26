import os
import json
from glob import glob

# === CONFIG ===
input_dir = "data/cleaned/dialogue"
output_dir = "data/processed/dialogue"

# === UTILITY FUNCTIONS ===
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# === PROCESS FUNCTION ===
def add_scene_characters_to_play(play_data):
    for act in play_data.get("acts", []):
        for scene in act.get("scenes", []):
            dialogues = scene.get("dialogues", [])
            characters = sorted(set(d["speaker"] for d in dialogues if "speaker" in d and d["speaker"].strip()))
            scene["scene_characters"] = characters
    return play_data

# === MAIN SCRIPT ===
def main():
    files = glob(os.path.join(input_dir, "*.json"))
    
    for filepath in files:
        try:
            data = load_json(filepath)
            updated_data = add_scene_characters_to_play(data)
            filename = os.path.basename(filepath)
            output_path = os.path.join(output_dir, filename)
            save_json(updated_data, output_path)
            print(f"✅ Updated scene_characters in: {filename}")
        except Exception as e:
            print(f"❌ Failed to process {filepath}: {e}")

    print(f"\n🎉 Done! All files updated with scene_characters.")

# === RUN ===
if __name__ == "__main__":
    main()