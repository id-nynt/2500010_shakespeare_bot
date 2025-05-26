import os
import json
import re

# Map of Roman numerals I-VIII (you can extend if needed)
ROMAN_TO_INT = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
}

def roman_to_int(roman):
    return ROMAN_TO_INT.get(roman.lower())

def clean_summary_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = {}
    cleaned["title"] = data.get("title", "")
    acts = data.get("acts", {})

    cleaned_acts = {}
    for key, summary in acts.items():
        # Extract roman numeral part from key, e.g. 'act_i' -> 'i'
        match = re.match(r"act_([ivx]+)", key, re.IGNORECASE)
        if match:
            roman = match.group(1)
            arabic = roman_to_int(roman)
            if arabic is not None:
                new_key = f"act_{arabic}"
            else:
                # fallback to original key if roman not found
                new_key = key
        else:
            new_key = key
        cleaned_acts[new_key] = summary

    cleaned["acts"] = cleaned_acts

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

def main():
    input_dir = "data/raw/summary_act"
    output_dir = "data/cleaned/summary_act"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            clean_summary_file(input_path, output_path)
            print(f"Processed {filename}")

if __name__ == "__main__":
    main()