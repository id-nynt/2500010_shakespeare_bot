import json
import random
import os

# List of plays to include
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

# Paths to input and output files
RELATIONSHIPS_JSON_PATH = "data/glossary/character_relationship.json"
OUTPUT_JSONL_PATH = "data/prompt_response/relationship.jsonl"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_JSONL_PATH), exist_ok=True)

# Load character relationships from JSON, filtering by specified plays
def load_relationships():
    try:
        with open(RELATIONSHIPS_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Handle list of plays with character_descriptions
        if isinstance(data, list) and data and "character_descriptions" in data[0]:
            # Extract relationships only for specified plays
            all_relationships = []
            for play in data:
                if play.get("title") in PLAYS:
                    all_relationships.extend(play["character_descriptions"])
            return all_relationships
        # Handle flat list of relationships (fallback, though not expected per format)
        return [rel for rel in data if rel.get("play") in PLAYS]
    except FileNotFoundError:
        print(f"Error: {RELATIONSHIPS_JSON_PATH} not found.")
        return []
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in character_relationship.json.")
        return []

# Generate five types of questions for a given character pair and play
def generate_questions(play, character_1, character_2):
    questions = [
        {
            "type": "direct_relationship",
            "question": f"What is the relationship between {character_1} and {character_2} in {play}?",
            "answer_field": "relationship_type"
        },
        {
            "type": "description_based",
            "question": f"How does {character_1} relate to {character_2} in {play}?",
            "answer_field": "description"
        },
        {
            "type": "role_based",
            "question": f"Who is {character_2} to {character_1} in {play}?",
            "answer_field": "relationship_type"
        },
        {
            "type": "conflict",
            "question": f"Is there any conflict in the relationship between {character_1} and {character_2} in {play}?",
            "answer_field": "description"
        },
        {
            "type": "contextual",
            "question": f"Why is the relationship between {character_1} and {character_2} significant in {play}?",
            "answer_field": "description"
        }
    ]
    return questions

# Get answer from relationship data based on question type
def get_answer(relationship, question_type, question):
    if question_type in ["direct_relationship", "role_based"]:
        return f"{relationship['character_2']} is {relationship['character_1']}'s {relationship['relationship_type']} in {relationship['play']}. {relationship['description']}"
    elif question_type in ["description_based", "contextual"]:
        return relationship['description']
    elif question_type == "conflict":
        # Check if description mentions conflict-related terms
        conflict_keywords = ["antagonistic", "tension", "betrayal", "conflict", "strained"]
        has_conflict = any(keyword in relationship['description'].lower() for keyword in conflict_keywords)
        if has_conflict:
            return f"Yes, there is conflict in the relationship between {relationship['character_1']} and {relationship['character_2']} in {relationship['play']}. {relationship['description']}"
        else:
            return f"No significant conflict is noted in the relationship between {relationship['character_1']} and {relationship['character_2']} in {relationship['play']}. {relationship['description']}"
    return "Answer not found."

# Save question-answer pair to JSONL
def save_to_jsonl(question, answer):
    output_data = {"prompt": question, "response": answer}
    with open(OUTPUT_JSONL_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(output_data) + '\n')

def main():
    # Load relationships
    relationships = load_relationships()
    if not relationships:
        print("No relationships loaded for the specified plays. Exiting.")
        return

    # Clear output file if it exists
    if os.path.exists(OUTPUT_JSONL_PATH):
        os.remove(OUTPUT_JSONL_PATH)

    # Process each relationship
    total_pairs = 0
    for relationship in relationships:
        character_1 = relationship["character_1"]
        character_2 = relationship["character_2"]
        play = relationship["play"]

        # Generate five question types
        questions = generate_questions(play, character_1, character_2)

        # Randomly select one question
        selected_question = random.choice(questions)
        question_text = selected_question["question"]
        question_type = selected_question["type"]

        # Get the answer with explanation
        answer = get_answer(relationship, question_type, question_text)

        # Save the selected question-answer pair to JSONL
        save_to_jsonl(question_text, answer)
        total_pairs += 1

    print(f"Total question-answer pairs saved: {total_pairs}")

if __name__ == "__main__":
    main()