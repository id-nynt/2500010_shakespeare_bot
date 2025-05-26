import gradio as gr
import os
import pickle
import faiss
import numpy as np
import torch
import functools
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from peft import PeftModel, PeftConfig

# === Configuration ===
MODEL_PATH = "models/gpt-neo-lora-finetuned"
INDEX_PATH = "retrieval/index"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load Model ===
def load_model():
    config = PeftConfig.from_pretrained(MODEL_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, MODEL_PATH).to(DEVICE)
    return model, tokenizer

# === Load RAG Components ===
def load_rag_components():
    with open(os.path.join(INDEX_PATH, "chunk_mapping.pkl"), "rb") as f:
        chunk_mapping = pickle.load(f)
    with open(os.path.join(INDEX_PATH, "chunks_indexed.pkl"), "rb") as f:
        chunks_data = pickle.load(f)
    embeddings = np.load(os.path.join(INDEX_PATH, "embeddings.npy"))
    index = faiss.read_index(os.path.join(INDEX_PATH, "faiss_index.bin"))
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return chunk_mapping, chunks_data, embeddings, index, embedding_model

# === Embed Query Function ===
def embed_query(query, embedding_model):
    return embedding_model.encode([query])[0]

# === Retrieval Function ===
def retrieve_chunks(query, embedding_model, index, chunks_data, top_k=3):
    query_vector = embed_query(query, embedding_model).astype("float32")
    _, indices = index.search(np.array([query_vector]), top_k)
    return [chunks_data[idx] for idx in indices[0]]

# === Prompt Builder ===
def build_prompt(query, contexts):
    context_text = "\n\n---\n\n".join(
        c.get("dialogue", c.get("content", "")) if isinstance(c, dict) else c for c in contexts
    )
    prompt = (
        f"You are a helpful Shakespeare expert. Use the context below to answer the question.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\nAnswer:"
    )
    return prompt

# === Generate Answer ===
def chatbot_interface(user_input, model, tokenizer, embedding_model, index, chunks_data):
    if not user_input or user_input.strip().lower() == "exit":
        return "Please enter a valid question or try again."
    chunks = retrieve_chunks(user_input, embedding_model, index, chunks_data)
    prompt = build_prompt(user_input, chunks)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if DEVICE == "cuda" else -1)
    output = generator(prompt, max_new_tokens=150, temperature=0.7)[0]["generated_text"]
    return output[len(prompt):].strip()

# === Initialize Components ===
model, tokenizer = load_model()
chunk_mapping, chunks_data, embeddings, index, embedding_model = load_rag_components()

# === Wrap chatbot function with fixed args ===
chat_fn = functools.partial(
    chatbot_interface,
    model=model,
    tokenizer=tokenizer,
    embedding_model=embedding_model,
    index=index,
    chunks_data=chunks_data
)

# === Gradio Interface ===
with gr.Blocks() as demo:
    gr.Markdown("# 🎭 Shakespeare Chatbot")
    user_input = gr.Textbox(label="Ask a question about Shakespeare")
    output = gr.Textbox(label="Chatbot Response")
    button = gr.Button("Submit")
    button.click(
        fn=chat_fn,
        inputs=user_input,
        outputs=output
    )

demo.launch()