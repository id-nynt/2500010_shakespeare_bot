import json
import os
from glob import glob

# === FOLDER PATHS ===
knowledge_base_dir = "data/cleaned/knowledge_base"
act_summary_dir = "data/cleaned/summary_act"
scene_summary_dir = "data/cleaned/summary_scene"
output_dir = "data/processed/full_summary"

# === UTILITY FUNCTIONS ===
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def create_structured_play(play_title, play_summary, act_summaries, scene_list):
    structured = {
        "title": play_title,
        "play_summary": play_summary,
        "acts": []
    }

    for act_num_str, act_summary in act_summaries.get("acts", {}).items():
        act_number = int(act_num_str.replace("act_", ""))
        scenes = []

        for scene in scene_list:
            if scene["title"].lower() != play_title.lower():
                continue
            if scene["act"] == act_number:
                scenes.append({
                    "scene": scene["scene"],
                    "location": scene.get("location", ""),
                    "scene_summary": scene.get("scene_summary", "")
                })

        act_entry = {
            "act": act_number,
            "act_summary": act_summary,
            "scenes": sorted(scenes, key=lambda x: x["scene"])
        }

        structured["acts"].append(act_entry)

    structured["acts"] = sorted(structured["acts"], key=lambda x: x["act"])
    return structured

# === MAIN LOOP ===
knowledge_files = glob(os.path.join(knowledge_base_dir, "*.json"))

for kb_path in knowledge_files:
    filename = os.path.basename(kb_path)
    play_name = filename.replace(".json", "")

    try:
        act_path = os.path.join(act_summary_dir, filename)
        scene_path = os.path.join(scene_summary_dir, filename)

        # Skip if required files are missing
        if not os.path.exists(act_path) or not os.path.exists(scene_path):
            print(f"Skipping {play_name} (missing act or scene file)")
            continue

        # Load files
        knowledge_data = load_json(kb_path)
        act_data = load_json(act_path)
        scene_data = load_json(scene_path)

        # Extract main title and summary
        play_title = knowledge_data.get("title", play_name)
        play_summary = knowledge_data.get("summary", "")

        # Create structured play
        structured_play = create_structured_play(play_title, play_summary, act_data, scene_data)

        # Output file path
        output_path = os.path.join(output_dir, f"{play_name}.json")
        save_json(structured_play, output_path)

        print(f"✅ Saved structured summary for: {play_name}")

    except Exception as e:
        print(f"❌ Error processing {play_name}: {e}")