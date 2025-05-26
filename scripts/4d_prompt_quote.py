import os
import json
import random
import logging

# === CONFIG ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

QUOTE_FILE = "data/processed/quote/quote.json"
FACTUAL_FILE = "data/processed/factual/factual.json"
OUTPUT_FILE = "data/processed/training_data/quote_qa_dataset.jsonl"
PLAYS = [
    "Hamlet, Prince of Denmark",
    "Romeo and Juliet",
    "Macbeth",
    "A Midsummer Night's Dream",
    "Much Ado About Nothing",
    "The Tragedy of King Lear",
    "Othello, the Moor of Venice",
    "Twelfth Night; Or, What You Will",
    "The Tempest",
    "Julius Caesar",
]

# === QUESTION TEMPLATES ===
QUESTION_TEMPLATES = {
    "famous_quotes": [
        lambda t: f"Can you list some well-known quotes from *{t}*?",
        lambda t: f"What are the most famous quotes from *{t}*?",
        lambda t: f"Which lines are iconic in *{t}*?"
    ],
    "quote_explanation": [
        lambda t, q: f"What does the quote '{q}' mean in *{t}*?",
        lambda t, q: f"Can you explain the meaning of '{q}' in *{t}*?",
        lambda t, q: f"What is the meaning of the quote '{q}' in *{t}*?"
    ],
    "main_characters": [
        lambda t: f"Who are the primary characters in *{t}*?",
        lambda t: f"Can you list the key characters in *{t}*?",
        lambda t: f"What are the names of the main figures in *{t}*?"
    ],
    "quote_speaker": [
        lambda p, q: f"Who is the speaker of '{q}' in *{p}*?",
        lambda p, q: f"Which character said '{q}' in *{p}*?",
        lambda p, q: f"In *{p}*, who said the line '{q}'"
    ],
    "identify_play": [
        lambda q: f"Which Shakespeare play includes the quote '{q}'?",
        lambda q: f"From which play is the line '{q}' taken?",
        lambda q: f"Which is the play that has the quote '{q}'?"
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
def generate_quote_qa():
    qa_dataset = []
    quote_data = load_json(QUOTE_FILE)
    factual_data = load_json(FACTUAL_FILE)
    if not quote_data or not factual_data:
        logger.error("No data loaded, exiting")
        return

    # Filter for requested plays (quotes)
    quote_plays = [play for play in quote_data if play.get("title") in PLAYS]
    missing_quote_plays = [title for title in PLAYS if title not in [p.get("title") for p in quote_plays]]
    if missing_quote_plays:
        logger.warning(f"Missing plays in quote.json: {missing_quote_plays}")

    # Create factual data lookup by title
    factual_lookup = {play.get("title", ""): play for play in factual_data if play.get("title")}
    missing_factual_plays = [title for title in PLAYS if title not in factual_lookup]
    if missing_factual_plays:
        logger.warning(f"Missing plays in factual.json: {missing_factual_plays}")

    for play in quote_plays:
        title = play.get("title", "")
        if not title:
            logger.warning("Skipping play with no title")
            continue

        logger.info(f"Processing play: {title}")
        quotes = play.get("famous_quotes", [])
        if not quotes:
            logger.warning(f"No quotes found for {title}")
            continue

        # Question 1: List all famous quotes
        question_template = random.choice(QUESTION_TEMPLATES["famous_quotes"])
        question = question_template(title)
        response = "; ".join([q.get("quote", "") for q in quotes if q.get("quote")]) + "."
        qa_dataset.append({"prompt": question, "response": response})

        # Questions 2, 4, 5: Per quote
        for quote_entry in quotes:
            quote = quote_entry.get("quote", "")
            explanation = quote_entry.get("explanation", "")
            speaker = quote_entry.get("speaker", "")
            act = quote_entry.get("act", "")
            scene = quote_entry.get("scene", "")
            if not quote:
                logger.warning(f"Skipping empty quote in {title}")
                continue

            # Question 2: Explain the quote
            if explanation:
                question_template = random.choice(QUESTION_TEMPLATES["quote_explanation"])
                question = question_template(title, quote)
                qa_dataset.append({"prompt": question, "response": explanation})

            # Question 4: Who said the quote
            if speaker and act and scene:
                question_template = random.choice(QUESTION_TEMPLATES["quote_speaker"])
                question = question_template(title, quote)
                response = f"{speaker} in Act {act}, Scene {scene}."
                qa_dataset.append({"prompt": question, "response": response})

            # Question 5: Identify the play
            question_template = random.choice(QUESTION_TEMPLATES["identify_play"])
            question = question_template(quote)
            qa_dataset.append({"prompt": question, "response": title})

        # Question 3: Main characters
        factual_play = factual_lookup.get(title, {})
        main_characters = factual_play.get("main_characters", [])
        if main_characters:
            question_template = random.choice(QUESTION_TEMPLATES["main_characters"])
            question = question_template(title)
            response = ", ".join(main_characters) + "."
            qa_dataset.append({"prompt": question, "response": response})
        else:
            logger.warning(f"No main characters found for {title} in factual.json")

    save_jsonl(qa_dataset, OUTPUT_FILE)
    logger.info(f"✅ Generated {len(qa_dataset)} prompt-response pairs and saved to {OUTPUT_FILE}")

# === RUN ===
if __name__ == "__main__":
    generate_quote_qa()