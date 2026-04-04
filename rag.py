import numpy as np 
from sentence_transformers import SentenceTransformer

import os 
import subprocess


DATA_DIR = "data"
TOP_K = 2

OLLAMA_MODEL = "phi"
ST_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 125
OVERLAP = 25


embedder = SentenceTransformer(ST_MODEL)

def load_docs(data_dir):
    texts = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt") or file.endswith(".md"):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

def chunk_text(text):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+CHUNK_SIZE]
        chunks.append(" ".join(chunk))
        i += CHUNK_SIZE - OVERLAP
    return chunks

def build_index(chunks):
    X = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def retrive(query, X, chunks):
    q = embedder.encode([query], convert_to_numpy=True)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    
    sims = np.dot(q, X.T).flatten()

    top_idx = np.argsort(sims)[::-1][:TOP_K]
    return [chunks[i] for i in top_idx]


def ask_ollama(prompt):
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input = prompt.encode(),
        stdout = subprocess.PIPE
    )
    return result.stdout.decode()

def rag(query, X, chunks):
    top_chunks = retrive(query, X, chunks)

    context = "\n\n".join(top_chunks)

    prompt = f"""
Answer ONLY from the context. If not found, say "Not in context".

context:{context}

Question:{query}
    """
    return ask_ollama(prompt)

if __name__ == "__main__":
    print("loading documents...")
    texts = load_docs(DATA_DIR)

    chunks = []
    for t in texts:
        chunks.extend(chunk_text(t))

    print("Building Index...")
    X = build_index(chunks)

    print("Ready. Ask Questions (type 'exit' to quit)\n")

    while True:
        query = input(">> ")
        if query.lower() == "exit":
            break

        answer = rag(query, X, chunks)
        print(answer)
