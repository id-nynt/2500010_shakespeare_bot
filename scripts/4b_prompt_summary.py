import os
import json
import random
import logging

# === CONFIG ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

plays = [
    "hamlet",
    "romeo_and_juliet",
    "macbeth",
    "midsummer_nights_dream",
    "much_ado_about_nothing",
    "king_lear",
    "othello",
    "twelfth_night",
    "the_tempest",
    "julius_caesar",
]
input_dir = "data/processed/full_summary"
output_file = "data/prompt_response/summary_qa.jsonl"

# === QUESTION TYPES ===
PLAY_QUESTION_TEMPLATES = [
    lambda t: f"What is the storyline of *{t}*?",
    lambda t: f"Summarise the entire play *{t}*?",
    lambda t: f"What are the main events in *{t}*?",
    lambda t: f"Tell me about the plot of *{t}*.",
    lambda t: f"What are the key points in *{t}*’s story?"
]

ACT_QUESTION_TEMPLATES = [
    lambda t, a: f"What happens in Act {a} of *{t}*?",
    lambda t, a: f"Summarize Act {a} from *{t}*.",
    lambda t, a: f"Can you describe the main events in Act {a} of *{t}*?",
    lambda t, a: f"What are the main events in Act {a} in *{t}*’s story?",
    lambda t, a: f"Give a brief overview of Act {a} in *{t}*."
]

SCENE_QUESTION_TEMPLATES = [
    lambda t, a, s: f"What happens in Act {a}, Scene {s} of *{t}*?",
    lambda t, a, s: f"Summarize Act {a}, Scene {s} from *{t}*?",
    lambda t, a, s: f"Give me a summary of *{t}*, Act {a}, Scene {s}.",
    lambda t, a, s: f"What happens in Scene {s} of Act {a} in *{t}*.",
    lambda t, a, s: f"What are the main event in Act {a}, Scene {s} of *{t}*?"
]

# === HELPERS ===
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        # Attempt to fix common issues like trailing commas
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                # Simple fix: remove trailing commas before closing brackets
                content = content.replace("},]", "}]").replace("},}", "}}")
                return json.loads(content)
        except Exception as e2:
            logger.error(f"Failed to fix JSON in {path}: {e2}")
            return None

def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

# === MAIN FUNCTION ===
def generate_qa_data():
    qa_dataset = []
    play_counts = {play: {"play": 0, "acts": 0, "scenes": 0} for play in plays}

    for play in plays:
        path = os.path.join(input_dir, f"{play}.json")
        data = load_json(path)
        if not data:
            logger.warning(f"Skipping {play} due to file loading error")
            continue

        title = data.get("title", play.title())
        logger.info(f"Processing play: {title}")

        # Play-level summary
        play_summary = data.get("play_summary", "")
        if play_summary:
            play_question_template = random.choice(PLAY_QUESTION_TEMPLATES)
            play_question = play_question_template(title)
            qa_dataset.append({
                "prompt": play_question,
                "response": play_summary
            })
            play_counts[play]["play"] += 1
        else:
            logger.warning(f"No play summary found for {title}")

        # Process acts
        acts = data.get("acts", [])
        if not acts:
            logger.warning(f"No acts found for {title}")
        
        for act in acts:
            act_num = act.get("act")
            if not act_num:
                logger.warning(f"Skipping act in {title} due to missing act number")
                continue

            # Act-level summary
            act_summary = act.get("act_summary", "")
            if act_summary:
                act_question_template = random.choice(ACT_QUESTION_TEMPLATES)
                act_question = act_question_template(title, act_num)
                qa_dataset.append({
                    "prompt": act_question,
                    "response": act_summary
                })
                play_counts[play]["acts"] += 1
            else:
                logger.warning(f"No act summary for Act {act_num} in {title}")

            # Scene-level summaries
            scenes = act.get("scenes", [])
            if not scenes:
                logger.warning(f"No scenes found for Act {act_num} in {title}")
            
            for scene in scenes:
                scene_num = scene.get("scene")
                scene_summary = scene.get("scene_summary", "")
                if not scene_num:
                    logger.warning(f"Skipping scene in {title}, Act {act_num} due to missing scene number")
                    continue
                if not scene_summary:
                    logger.warning(f"No scene summary for {title}, Act {act_num}, Scene {scene_num}")
                    continue
                question_template = random.choice(SCENE_QUESTION_TEMPLATES)
                question = question_template(title, act_num, scene_num)
                qa_dataset.append({
                    "prompt": question,
                    "response": scene_summary
                })
                play_counts[play]["scenes"] += 1

    save_jsonl(qa_dataset, output_file)
    logger.info(f"✅ Generated {len(qa_dataset)} prompt-response pairs and saved to {output_file}")
    for play, counts in play_counts.items():
        logger.info(f"Summary for {play}: {counts['play']} play, {counts['acts']} acts, {counts['scenes']} scenes")

# === RUN ===
if __name__ == "__main__":
    generate_qa_data()