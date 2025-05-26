import os
import json
import random
import logging
from pathlib import Path

# === CONFIG ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GLOSSARY_FILE = "data/glossary/glossary.json"
OUTPUT_FILE = "data/prompt_response/glossary_qa.jsonl"

# === QUESTION TEMPLATES ===
QUESTION_TEMPLATES = {
    "meaning_direct": [
        lambda word: f"What does the word *{word}* mean in modern English?",
        lambda word: f"Explain the meaning of *{word}* in contemporary English?",
        lambda word: f"What is the meaning of the word *{word}*?"
    ]
}

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
        return None

def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

# === MAIN FUNCTION ===
def generate_glossary_qa():
    qa_dataset = []
    glossary_data = load_json(GLOSSARY_FILE)
    
    if not glossary_data:
        logger.error("No glossary data loaded, exiting")
        return 0

    logger.info(f"Processing glossary with {len(glossary_data)} terms")
    
    for word, meaning in glossary_data.items():
        # Randomly select a question template
        question_template = random.choice(QUESTION_TEMPLATES["meaning_direct"])
        question = question_template(word)
        response = meaning
        
        qa_dataset.append({"prompt": question, "response": response})

    save_jsonl(qa_dataset, OUTPUT_FILE)
    logger.info(f"✅ Generated {len(qa_dataset)} prompt-response pairs and saved to {OUTPUT_FILE}")
    return len(qa_dataset)

# === RUN ===
if __name__ == "__main__":
    total_pairs = generate_glossary_qa()
    print(f"Generated {total_pairs} prompt-response pairs")