from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

# === Load the base model and LoRA fine-tuned weights ===
base_model_name = "EleutherAI/gpt-neo-125m"
lora_model_path = "models/gpt-neo-lora-finetuned"  # Change this if you used a different output dir

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, lora_model_path)
model = model.merge_and_unload()  # optional: merge LoRA into base for faster inference
model.eval()

# Create pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# === Manual testing loop ===
print("💬 Ask your Shakespeare question (type 'exit' to quit):")
while True:
    user_input = input("\n🧠 You: ")
    if user_input.lower().strip() == "exit":
        break

    prompt = user_input.strip()
    outputs = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.95)

    print(f"\n🤖 Bot: {outputs[0]['generated_text'][len(prompt):].strip()}")