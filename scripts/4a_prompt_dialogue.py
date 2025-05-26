import os
import json
import random
from pathlib import Path

# Define input and output paths
input_dir = Path("data/processed/dialogue")
output_file = Path("data/prompt_response/dialogue_qa.jsonl")
output_file.parent.mkdir(parents=True, exist_ok=True)

# List of target plays
plays = [
    "hamlet", "romeo_and_juliet", "macbeth", "midsummer_nights_dream",
    "much_ado_about_nothing", "king_lear", "othello", "twelfth_night",
    "the_tempest", "julius_caesar",
]

# Question paraphrases
question_types = {
    1: [
        "What is the quote after '{line}'?",
        "What comes next after the line: '{line}'?",
        "Which quote follows this: '{line}'?",
        "Tell me the next line after: '{line}'",
        "Who speaks next and what do they say after: '{line}'?"
    ],
    2: [
        "Who are the characters in scene {scene}, act {act} of '{title}'?",
        "List characters in Act {act}, Scene {scene} of '{title}'",
        "Which characters appear in Act {act}, Scene {scene} of the play '{title}'?",
        "Who is present in Act {act}, Scene {scene} of '{title}'?",
        "Can you name the characters in Act {act}, Scene {scene} from '{title}'?"
    ],
    3: [
        "Who said the line '{line}' in '{title}'?",
        "Identify the speaker of '{line}' in the play '{title}'",
        "Which character said: '{line}' in '{title}'?",
        "Find the speaker of the quote '{line}' from '{title}'",
        "Who speaked this line in '{title}': '{line}'?"
    ],
    4: [
        "Which play contains the quote: '{line}'?",
        "In which Shakespeare play does the line '{line}' appear?",
        "What play has this line: '{line}'?",
        "Find the play that includes the line: '{line}'",
        "Which work includes the following quote: '{line}'?"
    ],
    5: [
        "Where does Act {act}, Scene {scene} of '{title}' take place?",
        "What is the setting of Act {act}, Scene {scene} in '{title}'?",
        "Can you describe the location of Act {act}, Scene {scene} in '{title}'?",
        "In what place is Act {act}, Scene {scene} set in '{title}'?",
        "What is the location for Act {act}, Scene {scene} of the play '{title}'?"
    ]
}

samples = []
max_samples_per_play = 100  # Limit for each question type per play
collected = {play: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0} for play in plays}

for play in plays:
    path = input_dir / f"{play}.json"
    if not path.exists():
        print(f"Warning: {path} not found, skipping {play}")
        continue

    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {path}: {e}")
            continue

    title = data.get("title", play.replace("_", " ").title())

    for act_obj in data.get("acts", []):
        act = act_obj.get("act")
        for scene_obj in act_obj.get("scenes", []):
            scene = scene_obj.get("scene")
            dialogues = scene_obj.get("dialogues", [])
            characters = scene_obj.get("scene_characters", [])
            location = scene_obj.get("location")

            # Question 2: Scene characters (up to 50 per play)
            if collected[play][2] < max_samples_per_play:
                question = random.choice(question_types[2]).format(title=title, act=act, scene=scene)
                answer = ", ".join(characters) if characters else "No characters specified"
                samples.append({"prompt": question, "response": answer})
                collected[play][2] += 1

            # Question 5: Scene location (up to 50 per play)
            if location and collected[play][5] < max_samples_per_play:
                question = random.choice(question_types[5]).format(title=title, act=act, scene=scene)
                answer = location
                samples.append({"prompt": question, "response": answer})
                collected[play][5] += 1

            for i, dialogue in enumerate(dialogues):
                speaker = dialogue.get("speaker")
                line = dialogue.get("line")

                if not line or not speaker:
                    continue

                # Question 1: What is the next quote? (up to 50 per play)
                if i + 1 < len(dialogues) and collected[play][1] < max_samples_per_play:
                    next_dialogue = dialogues[i + 1]
                    prompt = random.choice(question_types[1]).format(line=line)
                    answer = f"{next_dialogue['speaker']}: {next_dialogue['line']}"
                    samples.append({"prompt": prompt, "response": answer})
                    collected[play][1] += 1

                # Question 3: Who said this line? (up to 50 per play)
                if collected[play][3] < max_samples_per_play:
                    prompt = random.choice(question_types[3]).format(line=line, title=title)
                    answer = f"{speaker} in Act {act}, Scene {scene}"
                    samples.append({"prompt": prompt, "response": answer})
                    collected[play][3] += 1

                # Question 4: Which play contains the line? (up to 50 per play)
                if collected[play][4] < max_samples_per_play:
                    prompt = random.choice(question_types[4]).format(line=line)
                    answer = title
                    samples.append({"prompt": prompt, "response": answer})
                    collected[play][4] += 1

# Save output as JSONL
with open(output_file, 'w', encoding='utf-8') as f:
    for sample in samples:
        json.dump(sample, f, ensure_ascii=False)
        f.write('\n')

# Print summary
total_samples = len(samples)
print(f"Saved {total_samples} samples to {output_file}")
print("Collected per play and type:")
for play in plays:
    if play in collected:
        print(f"{play}: {collected[play]}")