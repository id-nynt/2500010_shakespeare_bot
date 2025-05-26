import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# === CONFIGURATION ===
model_name = "EleutherAI/gpt-neo-125m"
output_dir = "models/gpt-neo-lora-finetuned"  # replace with actual checkpoint
test_file = "data/fine_tuning/test.jsonl"
num_samples = 10
max_new_tokens = 100

# === LOAD MODEL & TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(model, output_dir)
model.eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# === LOAD TEST DATA ===
with open(test_file, "r", encoding="utf-8") as f:
    test_samples = [json.loads(line) for line in f]

# === RUN INFERENCE ON RANDOM TEST QUESTIONS ===
samples = random.sample(test_samples, num_samples)
print("=== EVALUATION ===\n")

for i, sample in enumerate(samples):
    prompt = sample["prompt"]
    reference = sample["response"]

    print(f"Q{i+1}: {prompt}")
    print("Expected Answer:", reference.strip())

    output = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    answer = output[0]["generated_text"].replace(prompt, "").strip()
    print("Model Answer:", answer)
    print("-" * 80)