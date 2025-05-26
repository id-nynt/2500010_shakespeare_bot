# Load the LoRA fine-tuned model
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_path = "models/gpt-neo-lora-finetuned"
config = PeftConfig.from_pretrained(peft_model_path)
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, peft_model_path)

# Integrate RAG
import os
import pickle
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === Paths ===
MODEL_PATH = "models/gpt-neo-lora-finetuned"
INDEX_PATH = "retrieval/index"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load RAG Components ===
with open(os.path.join(INDEX_PATH, "chunk_mapping.pkl"), "rb") as f:
    chunk_mapping = pickle.load(f)

with open(os.path.join(INDEX_PATH, "chunks_indexed.pkl"), "rb") as f:
    chunks_data = pickle.load(f)

embeddings = np.load(os.path.join(INDEX_PATH, "embeddings.npy"))

index = faiss.read_index(os.path.join(INDEX_PATH, "faiss_index.bin"))

# === Load Tokenizer & Model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if DEVICE == "cuda" else -1)

# === Embed Query Function ===
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # You can change this to match your index

def embed_query(query):
    return embedding_model.encode([query])[0]

# === Retrieval Function ===
def retrieve_chunks(query, top_k=3):
    query_vector = embed_query(query).astype("float32")
    _, indices = index.search(np.array([query_vector]), top_k)
    return [chunks_data[idx] for idx in indices[0]]

# === Prompt Builder ===
def build_prompt(query, contexts):
    context_text = "\n\n---\n\n".join(c.get("dialogue", "") if isinstance(c, dict) else c for c in contexts)
    prompt = (
        f"You are a helpful Shakespeare expert. Use the context below to answer the question.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\nAnswer:"
    )
    return prompt

# === Generate Answer ===
def answer_question(query, top_k=3, max_tokens=200):
    contexts = retrieve_chunks(query, top_k)
    prompt = build_prompt(query, contexts)
    outputs = generator(prompt, max_length=len(tokenizer(prompt)["input_ids"]) + max_tokens, do_sample=True, top_p=0.9)
    return outputs[0]["generated_text"].split("Answer:")[-1].strip()

# === CLI or Script Mode ===
if __name__ == "__main__":
    while True:
        user_query = input("Ask a Shakespeare question (or 'exit'): ")
        if user_query.lower() == "exit":
            break
        response = answer_question(user_query)
        print("\n💬 Response:\n", response, "\n")