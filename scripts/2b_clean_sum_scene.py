import os
import json
import re

input_folder = "data/raw/summary_scene"
output_folder = "data/cleaned/summary_scene"
os.makedirs(output_folder, exist_ok=True)

def sanitize_title(title):
    # Replace non-word characters with underscores and convert to lowercase
    return re.sub(r"[^\w]+", "_", title).strip("_").lower()

def parse_summary_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        # Normalize apostrophes and strip lines
        lines = [line.strip().replace("’", "'") for line in f if line.strip()]

    play_title = None
    act = None
    scene_data = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Match title
        if i == 0 and line.startswith("#"):
            play_title = line.replace("#", "").strip().replace("Scene Summaries", "").strip()

        # Match Act
        elif line.startswith("## Act"):
            act_match = re.search(r"Act\s+(\d+)", line)
            act = int(act_match.group(1)) if act_match else None

        # Match Scene
        elif line.startswith("### Scene"):
            scene_match = re.search(r"Scene\s+(\d+)", line)
            scene = int(scene_match.group(1)) if scene_match else None

            location = None
            summary = None

            # Look ahead for Location and Summary
            j = i + 1
            while j < len(lines) and not lines[j].startswith("### Scene") and not lines[j].startswith("## Act"):
                loc_match = re.match(r"\*\*Location\*\*: (.+)", lines[j])
                sum_match = re.match(r"\*\*Summary\*\*: (.+)", lines[j])

                if loc_match:
                    location = loc_match.group(1).strip()
                if sum_match:
                    summary = sum_match.group(1).strip()
                j += 1

            scene_id = f"{sanitize_title(play_title)}_{act}_{scene}"

            scene_data.append({
                "scene_id": scene_id,
                "title": play_title,
                "act": act,
                "scene": scene,
                "location": location,
                "scene_summary": summary
            })

            i = j - 1  # move pointer forward

        i += 1

    return scene_data

# Process all markdown files
for filename in os.listdir(input_folder):
    if filename.endswith(".markdown"):
        path = os.path.join(input_folder, filename)
        scenes = parse_summary_file(path)

        output_filename = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(output_folder, output_filename)

        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(scenes, out_f, indent=2, ensure_ascii=False)

print("✅ Scene summaries cleaned and saved to:", output_folder)