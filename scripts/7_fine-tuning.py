from datasets import load_dataset
from transformers import AutoTokenizer

# Load your dataset from JSONL
dataset = load_dataset('json', data_files='data/fine_tuning/combined_prompt.jsonl', split='train')

# Load tokenizer for GPT-Neo 125M
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")

# GPT-Neo tokenizer might not have pad token, so set it
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

max_length = 512  # or choose suitable max length

def preprocess(example):
    prompt = example['prompt']
    response = example['response']

    # Combine prompt and response
    full_text = prompt + " " + response

    # Tokenize full text
    tokenized_full = tokenizer(full_text, truncation=True, max_length=max_length, padding='max_length')

    # Tokenize prompt only to get prompt length
    tokenized_prompt = tokenizer(prompt, truncation=True, max_length=max_length, padding=False)
    prompt_len = len(tokenized_prompt['input_ids'])

    input_ids = tokenized_full['input_ids']
    attention_mask = tokenized_full['attention_mask']

    # Create labels: mask prompt tokens with -100
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    # Pad labels to max_length if needed
    if len(labels) < max_length:
        labels = labels + [-100] * (max_length - len(labels))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# Map preprocessing over dataset
train_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# Now train_dataset can be used with Trainer or PEFT fine-tuning pipeline

# FINE-TUNE
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType

# === CONFIG ===
model_name = "EleutherAI/gpt-neo-125m"
train_file = "data/fine_tuning/combined_prompt.jsonl"
max_length = 512
output_dir = "models/gpt-neo-lora-finetuned"

# === Load dataset ===
dataset = load_dataset("json", data_files={"train": train_file}, split="train")

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # GPT-Neo doesn't have pad_token, use eos_token instead

# === Preprocess function ===
def preprocess(example):
    prompt = example["prompt"]
    response = example["response"]
    full_text = prompt + " " + response

    tokenized_full = tokenizer(full_text, truncation=True, max_length=max_length, padding="max_length")
    tokenized_prompt = tokenizer(prompt, truncation=True, max_length=max_length, padding=False)

    prompt_len = len(tokenized_prompt["input_ids"])

    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]

    # labels: mask prompt tokens
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    if len(labels) < max_length:
        labels += [-100] * (max_length - len(labels))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# === Apply preprocessing ===
train_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# === Load model ===
model = AutoModelForCausalLM.from_pretrained(model_name)

# === Setup LoRA config ===
# Print the model's structure to identify target modules if needed
# print(model) # Uncomment this line to see the model's layer names

lora_config = LoraConfig(
    r=8,            # LoRA rank
    lora_alpha=32,  # LoRA alpha
    # Correct target modules for GPT-Neo: based on common naming conventions
    # Try these first. If they still cause an error, uncomment the line below and comment this one.
    target_modules=["q_proj", "k_proj", "v_proj"],
    # Alternatively, uncomment the line below to target all linear layers (except output)
    # target_modules="all-linear",
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# === Training arguments ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=3e-4,
    logging_dir=f"{output_dir}/logs",
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    eval_strategy="no",
    fp16=torch.cuda.is_available(),
    report_to="none",  # set to 'wandb' if you use Weights & Biases
    remove_unused_columns=False,
)

# === Data collator ===
data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

# Train the model using the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)
trainer.train()