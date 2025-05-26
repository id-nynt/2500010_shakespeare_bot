import os
import re
import json
import logging

# === CONFIG ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = "data/raw/dialogue"
PROCESSED_DIR = "data/cleaned/dialogue"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# === STAGE DIRECTION PATTERNS ===
# Match standalone stage directions (e.g., "Enter Horatio.", "Exit.", "[Aside]")
STAGE_DIRECTION_PATTERN = re.compile(
    r'^(Enter|Exit|Exeunt|Aside|Alarums?|Flourish|Sennet|Dies|Within|Re-enter|Sound|Knocking|Bell rings)(\s+[A-Z][A-Za-z\s,.]+)?[\.\!]?$|\[.*?\]$',
    re.IGNORECASE
)

def normalize_text(text):
    return (
        text.replace("’", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("–", "-")
            .replace("—", "-")
    )

def clean_and_parse_play(text, filename):
    # Normalize entire text first
    text = normalize_text(text)

    # Extract title BEFORE removing header/footer
    title_match = re.search(r'^Title:\s*(.+)$', text, re.MULTILINE | re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else filename.replace(".txt", "").replace("_", " ").title()
    title_slug = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')

    # Remove Project Gutenberg header/footer
    start = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*", text)
    end = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*", text)
    if not start or not end:
        logger.error(f"Invalid format in {filename}: Missing Project Gutenberg markers")
        return None
    text = text[start.end():end.start()]

    # Normalize whitespace
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{2,}', '\n\n', text).strip()

    # Remove dramatis personae, contents, etc.
    text = re.split(r"(?i)\bACT I\b", text)[-1]
    text = "ACT I\n" + text

    acts = []
    current_act = None
    current_scene = None
    scene_lines = []

    lines = text.splitlines()
    for line in lines:
        line = normalize_text(line.strip())  # Normalize individual line
        act_match = re.match(r'^ACT ([IVX]+)$', line)
        scene_match = re.match(r'^SCENE ([IVX]+)\.?\s*(.*)$', line, re.IGNORECASE)

        if act_match:
            if current_scene and scene_lines:
                current_scene['dialogues'] = extract_dialogues(scene_lines)
                current_act['scenes'].append(current_scene)
            if current_act:
                acts.append(current_act)
            current_act_number = roman_to_int(act_match.group(1))
            current_act = {"act": current_act_number, "scenes": []}
            current_scene = None
            scene_lines = []

        elif scene_match:
            if current_scene and scene_lines:
                current_scene['dialogues'] = extract_dialogues(scene_lines)
                current_act['scenes'].append(current_scene)
            scene_number = roman_to_int(scene_match.group(1))
            location = scene_match.group(2).strip()
            scene_id = f"{title_slug}_{current_act['act']}_{scene_number}"
            current_scene = {"scene": scene_number, "location": location, "scene_id": scene_id, "dialogues": []}
            scene_lines = []

        elif line == "":
            continue
        else:
            scene_lines.append(line)

    if current_scene and scene_lines:
        current_scene['dialogues'] = extract_dialogues(scene_lines)
        current_act['scenes'].append(current_scene)
    if current_act:
        acts.append(current_act)

    return {"title": title, "acts": acts}

def extract_dialogues(lines):
    dialogues = []
    current_speaker = None
    current_lines = []
    in_dialogue_block = False

    for line in lines:
        stripped = normalize_text(line.strip())
        if not stripped:
            continue

        # Match speaker names (all caps, optional period)
        speaker_match = re.match(r'^([A-Z][A-Z\s\-]+)\.?$', stripped)
        if speaker_match:
            if current_speaker and current_lines:
                cleaned_line = re.sub(r'\[.*?\]', '', " ".join(current_lines)).strip()
                if cleaned_line:
                    dialogues.append({
                        "speaker": current_speaker.title(),
                        "line": cleaned_line
                    })
            current_speaker = speaker_match.group(1).strip()
            current_lines = []
            in_dialogue_block = True
            logger.debug(f"Speaker detected: {current_speaker}")
            continue

        # Skip stage directions if not in a dialogue block or if they match the pattern
        if not in_dialogue_block or STAGE_DIRECTION_PATTERN.match(stripped):
            logger.debug(f"Skipped stage direction: {stripped}")
            continue

        if current_speaker and stripped:
            current_lines.append(stripped)

    if current_speaker and current_lines:
        cleaned_line = re.sub(r'\[.*?\]', '', " ".join(current_lines)).strip()
        if cleaned_line:
            dialogues.append({
                "speaker": current_speaker.title(),
                "line": cleaned_line
            })

    return dialogues

def roman_to_int(roman):
    roman_numerals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
    result, prev = 0, 0
    for char in reversed(roman):
        value = roman_numerals.get(char.upper(), 0)
        if value < prev:
            result -= value
        else:
            result += value
        prev = value
    return result

def main():
    for fname in os.listdir(RAW_DIR):
        if not fname.endswith(".txt"):
            continue
        raw_path = os.path.join(RAW_DIR, fname)
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        parsed_play = clean_and_parse_play(raw_text, fname)
        if parsed_play:
            save_path = os.path.join(PROCESSED_DIR, fname.replace(".txt", ".json"))
            with open(save_path, "w", encoding="utf-8") as out:
                json.dump(parsed_play, out, indent=2, ensure_ascii=False)
            logger.info(f"✅ Processed: {fname}")
        else:
            logger.info(f"⚠️ Skipped (invalid format): {fname}")

if __name__ == "__main__":
    main()