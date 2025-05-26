import os
import json
import logging

# === CONFIG ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR = "data/prompt_response"
OUTPUT_DIR = "data/fine_tuning"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "combined_prompt.jsonl")

# === HELPERS ===
def is_valid_jsonl_line(line, file_path):
    try:
        data = json.loads(line.strip())
        if not isinstance(data, dict) or "prompt" not in data or "response" not in data:
            logger.warning(f"Invalid JSONL line in {file_path}: Missing prompt or response")
            return False
        return True
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON in {file_path}: {line.strip()}")
        return False

def format_pair(prompt, response):
    prompt = prompt.strip()
    response = response.strip().rstrip("?")  # Remove trailing question mark if any
    formatted_prompt = f"{prompt}\n" #"Question: {prompt}\nAnswer:"
    return {"prompt": formatted_prompt, "response": response}

def combine_jsonl_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_DIR):
        logger.error(f"Input directory {INPUT_DIR} does not exist")
        return 0

    jsonl_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jsonl")]
    if not jsonl_files:
        logger.error(f"No JSONL files found in {INPUT_DIR}")
        return 0

    total_pairs = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for filename in jsonl_files:
            file_path = os.path.join(INPUT_DIR, filename)
            logger.info(f"Processing file: {file_path}")
            file_pairs = 0

            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    for line in infile:
                        if line.strip() and is_valid_jsonl_line(line, file_path):
                            data = json.loads(line.strip())
                            formatted = format_pair(data["prompt"], data["response"])
                            outfile.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                            file_pairs += 1
                            total_pairs += 1
                logger.info(f"Added {file_pairs} pairs from {filename}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    logger.info(f"✅ Combined {total_pairs} prompt-response pairs into {OUTPUT_FILE}")
    return total_pairs

# === RUN ===
if __name__ == "__main__":
    total = combine_jsonl_files()
    print(f"Total prompt-response pairs: {total}")